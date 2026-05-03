"""
Thread-safe TTS inference engine.

Wraps the model manager with a thread-safe queue to ensure only one inference
runs at a time on the GPU. This prevents CUDA memory issues and race conditions
when multiple HTTP requests arrive concurrently.
"""

import io
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from server.model_loader import TTSModelManager


@dataclass
class InferenceRequest:
    """A request submitted to the inference queue."""
    text: str
    max_new_tokens: int = 1024
    chunk_length: int = 200
    top_p: float = 0.7
    temperature: float = 0.7
    repetition_penalty: float = 1.2
    format: str = "wav"


@dataclass
class InferenceResult:
    """Result from the inference engine."""
    success: bool
    audio_bytes: Optional[bytes] = None
    sample_rate: int = 44100
    format: str = "wav"
    duration_seconds: float = 0.0
    generation_time: float = 0.0
    error: Optional[str] = None


class TTSEngine:
    """
    Thread-safe TTS inference engine.

    Uses a producer-consumer pattern with a single worker thread to serialize
    GPU access. Requests are submitted via synthesize() and processed sequentially.
    """

    def __init__(self, model_manager: TTSModelManager, max_queue_size: int = 10):
        self.model_manager = model_manager
        self._request_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def start(self) -> None:
        """Start the inference worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="tts-inference-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("TTS inference engine started (worker thread active)")

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._shutdown_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=30)
        logger.info("TTS inference engine stopped")

    def synthesize(
        self,
        request: InferenceRequest,
        timeout: float = 300.0,
    ) -> InferenceResult:
        """
        Submit a TTS request and wait for the result.

        This method is safe to call from async/threaded contexts.
        The actual inference happens on the dedicated worker thread.

        Args:
            request: The inference request
            timeout: Maximum seconds to wait for a result

        Returns:
            InferenceResult with audio bytes or error

        Raises:
            queue.Full: If the inference queue is full (server overloaded)
            TimeoutError: If inference takes longer than timeout
        """
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        work_item = (request, result_queue)

        try:
            self._request_queue.put(work_item, timeout=5.0)
        except queue.Full:
            return InferenceResult(
                success=False,
                error="Server is overloaded. Please try again later.",
            )

        try:
            result = result_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return InferenceResult(
                success=False,
                error=f"Inference timed out after {timeout}s.",
            )

    def _worker_loop(self) -> None:
        """
        Main loop for the inference worker thread.
        Processes requests sequentially from the queue.
        """
        logger.info("Inference worker thread started")

        while not self._shutdown_event.is_set():
            try:
                work_item = self._request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            request, result_queue = work_item

            try:
                result = self._process_request(request)
                result_queue.put(result)
            except Exception as e:
                logger.error(f"Unhandled error in worker: {traceback.format_exc()}")
                result_queue.put(InferenceResult(
                    success=False,
                    error=f"Internal inference error: {str(e)}",
                ))

        logger.info("Inference worker thread exiting")

    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        """Process a single inference request."""
        logger.info(f"Processing TTS request: {len(request.text)} chars, format={request.format}")

        t_start = time.time()

        try:
            # Generate raw PCM audio
            audio_pcm = self.model_manager.generate_speech(
                text=request.text,
                max_new_tokens=request.max_new_tokens,
                chunk_length=request.chunk_length,
                top_p=request.top_p,
                temperature=request.temperature,
                repetition_penalty=request.repetition_penalty,
            )

            sample_rate = self.model_manager.sample_rate
            duration = len(audio_pcm) / sample_rate

            # Convert to requested format
            audio_bytes = self._encode_audio(
                audio_pcm,
                sample_rate=sample_rate,
                format=request.format,
            )

            generation_time = time.time() - t_start
            logger.info(
                f"TTS complete: {duration:.1f}s audio in {generation_time:.1f}s "
                f"(RTF={generation_time/duration:.2f})"
            )

            return InferenceResult(
                success=True,
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                format=request.format,
                duration_seconds=duration,
                generation_time=generation_time,
            )

        except Exception as e:
            generation_time = time.time() - t_start
            logger.error(f"Inference failed after {generation_time:.1f}s: {traceback.format_exc()}")
            return InferenceResult(
                success=False,
                error=str(e),
                generation_time=generation_time,
            )

    @staticmethod
    def _encode_audio(
        audio_pcm: np.ndarray,
        sample_rate: int,
        format: str = "wav",
    ) -> bytes:
        """Encode PCM audio samples to the requested format."""
        buffer = io.BytesIO()

        if format == "wav":
            sf.write(buffer, audio_pcm, sample_rate, format="WAV", subtype="PCM_16")
        else:
            # Default fallback to WAV
            sf.write(buffer, audio_pcm, sample_rate, format="WAV", subtype="PCM_16")

        buffer.seek(0)
        return buffer.read()
