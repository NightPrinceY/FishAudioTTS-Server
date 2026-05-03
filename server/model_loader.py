"""
Model loading and management for the Fish Audio S2-Pro TTS server.

Handles loading the DualAR Transformer model and DACA codec, distributing them
across available GPUs, and pre-encoding the reference voice for voice cloning.

GPU Strategy:
    - Model (DualARTransformer): Loaded on primary GPU (default: cuda:0)
    - Codec (DAC):              Loaded on secondary GPU (default: cuda:1)
    - Reference voice tokens:   Encoded once at startup, cached in CPU memory
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import torch
from loguru import logger

# Ensure fish-speech is on the Python path
FISH_SPEECH_DIR = str(Path(__file__).parent.parent / "fish-speech")
if FISH_SPEECH_DIR not in sys.path:
    sys.path.insert(0, FISH_SPEECH_DIR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fish_speech.models.text2semantic.inference import (
    init_model,
    generate_long,
)


class TTSModelManager:
    """
    Singleton manager for all TTS model resources.

    Loads the DualAR LLM and DAC codec onto specified GPUs, pre-encodes
    the fixed reference voice, and provides a clean interface for inference.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device_model: str = "cuda:0",
        device_codec: str = "cuda:1",
        reference_audio_path: Optional[str] = None,
        reference_text: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device_model = device_model
        self.device_codec = device_codec
        self.reference_audio_path = reference_audio_path
        self.reference_text = reference_text

        # Will be set during load()
        self.llama_model = None
        self.decode_one_token = None
        self.codec_model = None
        self.cached_prompt_tokens = None  # Pre-encoded reference voice VQ codes
        self.sample_rate: int = 44100
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """
        Load all models and prepare for inference.
        This is called once at server startup.
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING TTS MODEL MANAGER")
        logger.info("=" * 60)

        t_start = time.time()

        # Determine precision based on GPU capability
        precision = self._select_precision()

        # Step 1: Load the main LLM (DualARTransformer)
        self._load_llm(precision)

        # Step 2: Load the audio codec
        self._load_codec(precision)

        # Step 3: Pre-encode the reference voice
        if self.reference_audio_path:
            self._encode_reference_voice()

        self._is_loaded = True
        elapsed = time.time() - t_start
        logger.info("=" * 60)
        logger.info(f"MODEL MANAGER READY — Total init time: {elapsed:.1f}s")
        logger.info(f"  LLM device:   {self.device_model}")
        logger.info(f"  Codec device:  {self.device_codec}")
        logger.info(f"  Voice cached:  {self.cached_prompt_tokens is not None}")
        logger.info(f"  Sample rate:   {self.sample_rate} Hz")
        logger.info("=" * 60)

    def _select_precision(self) -> torch.dtype:
        """Select the best precision for the current GPU."""
        device_idx = int(self.device_model.split(":")[-1]) if ":" in self.device_model else 0
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported(device_idx):
            logger.info("Using bfloat16 precision (GPU supports BF16)")
            return torch.bfloat16
        else:
            logger.info("Using float16 precision (GPU does not support BF16)")
            return torch.float16

    def _load_llm(self, precision: torch.dtype) -> None:
        """Load the DualARTransformer model."""
        logger.info(f"Loading DualARTransformer from {self.checkpoint_dir}")
        logger.info(f"  Target device: {self.device_model}, precision: {precision}")

        t0 = time.time()
        self.llama_model, self.decode_one_token = init_model(
            checkpoint_path=str(self.checkpoint_dir),
            device=self.device_model,
            precision=precision,
            compile=False,
        )

        # Setup KV caches for inference
        # IMPORTANT: Use a reduced max_seq_len to fit in 11GB VRAM.
        # The model's default is 32768 which creates a ~4.8GB cache,
        # exceeding GPU memory when combined with model weights (~9.1GB).
        # 4096 tokens is more than sufficient for TTS generation
        # (~10 minutes of speech at 21Hz frame rate).
        # Inference code reserves 2048 tokens for output (max_length - 2048 >= prompt_size).
        # 2560 = 512 token prompt budget + 2048 output budget, uses ~10.5GB on RTX 2080 Ti.
        cache_seq_len = min(2560, self.llama_model.config.max_seq_len)
        logger.info(f"  Setting up KV cache with max_seq_len={cache_seq_len}")
        with torch.device(self.device_model):
            self.llama_model.setup_caches(
                max_batch_size=1,
                max_seq_len=cache_seq_len,
                dtype=next(self.llama_model.parameters()).dtype,
            )

        # Override the config so generate() respects the cache limit
        self.llama_model.config.max_seq_len = cache_seq_len

        elapsed = time.time() - t0
        logger.info(f"  LLM loaded in {elapsed:.1f}s")

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(self.device_model) / 1e9
            mem_reserved = torch.cuda.memory_reserved(self.device_model) / 1e9
            logger.info(f"  GPU memory: {mem_used:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    def _load_codec(self, precision: torch.dtype) -> None:
        """Load the DAC audio codec model."""
        codec_path = self.checkpoint_dir / "codec.pth"
        logger.info(f"Loading DAC codec from {codec_path}")
        logger.info(f"  Target device: {self.device_codec}")

        t0 = time.time()

        # We need to temporarily cd to fish-speech for hydra config resolution
        original_cwd = os.getcwd()
        os.chdir(FISH_SPEECH_DIR)

        try:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(
                Path(FISH_SPEECH_DIR) / "fish_speech" / "configs" / "modded_dac_vq.yaml"
            )
            self.codec_model = instantiate(cfg)

            state_dict = torch.load(str(codec_path), map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if any("generator" in k for k in state_dict):
                state_dict = {
                    k.replace("generator.", ""): v
                    for k, v in state_dict.items()
                    if "generator." in k
                }

            self.codec_model.load_state_dict(state_dict, strict=False)
            self.codec_model.eval()
            self.codec_model.to(device=self.device_codec, dtype=precision)

            self.sample_rate = self.codec_model.sample_rate
        finally:
            os.chdir(original_cwd)

        elapsed = time.time() - t0
        logger.info(f"  Codec loaded in {elapsed:.1f}s (sample_rate={self.sample_rate})")

        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated(self.device_codec) / 1e9
            logger.info(f"  Codec GPU memory: {mem_used:.1f}GB allocated")

    @torch.no_grad()
    def _encode_reference_voice(self) -> None:
        """Pre-encode the reference voice audio into VQ tokens."""
        audio_path = self.reference_audio_path
        logger.info(f"Encoding reference voice: {audio_path}")

        t0 = time.time()

        wav_np, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        wav = torch.from_numpy(wav_np).to(self.device_codec)
        model_dtype = next(self.codec_model.parameters()).dtype

        audios = wav[None, None, :].to(dtype=model_dtype)
        audio_lengths = torch.tensor(
            [wav.shape[0]], device=self.device_codec, dtype=torch.long
        )

        indices, feature_lengths = self.codec_model.encode(audios, audio_lengths)
        self.cached_prompt_tokens = indices[0, :, : feature_lengths[0]].cpu()

        elapsed = time.time() - t0
        logger.info(
            f"  Voice encoded in {elapsed:.1f}s — "
            f"shape: {self.cached_prompt_tokens.shape}"
        )

    @torch.no_grad()
    def decode_codes_to_audio(self, codes: torch.Tensor) -> np.ndarray:
        """
        Decode VQ codes to a PCM 16-bit audio waveform (numpy array).

        Args:
            codes: Tensor of shape (num_codebooks, T) with VQ indices

        Returns:
            numpy array of int16 PCM audio samples
        """
        codes = codes.to(self.device_codec)
        audio_waveform = self.codec_model.from_indices(codes[None])
        audio_np = audio_waveform[0, 0].cpu().float().numpy()
        audio_pcm = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_pcm

    def generate_speech(
        self,
        text: str,
        max_new_tokens: int = 1024,
        chunk_length: int = 200,
        top_p: float = 0.7,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
    ) -> np.ndarray:
        """
        Generate speech audio from text using the fixed reference voice.

        Args:
            text: The text to synthesize
            max_new_tokens: Maximum tokens to generate (0 = auto)
            chunk_length: Text chunk length for iterative prompting
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor

        Returns:
            numpy array of int16 PCM audio at self.sample_rate Hz

        Raises:
            RuntimeError: If models are not loaded or generation fails
        """
        if not self._is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        # Build prompt tokens list (the cached reference voice)
        prompt_tokens_list = None
        prompt_text = None

        if self.cached_prompt_tokens is not None and self.reference_text:
            prompt_tokens_list = [self.cached_prompt_tokens]
            prompt_text = [self.reference_text]

        # Generate VQ codes from text
        generator = generate_long(
            model=self.llama_model,
            device=self.device_model,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=30,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            compile=False,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens_list,
        )

        # Collect all generated code chunks
        codes_list = []
        for response in generator:
            if response.action == "sample":
                codes_list.append(response.codes)
            elif response.action == "next":
                break

        if not codes_list:
            raise RuntimeError(
                "No audio was generated. The model returned no samples. "
                "This may happen if the input text is too short or invalid."
            )

        # Merge all code chunks
        if len(codes_list) == 1:
            merged_codes = codes_list[0]
        else:
            merged_codes = torch.cat(codes_list, dim=1)

        # Decode VQ codes to audio
        audio_pcm = self.decode_codes_to_audio(merged_codes)

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_pcm
