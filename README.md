# FishAudioTTS-Server

A production-ready FastAPI inference server for [Fish Audio S2-Pro](https://github.com/fishaudio/fish-speech) — a high-quality voice cloning TTS model with support for Arabic and 80+ languages.

This server wraps the S2-Pro model with a REST API, a thread-safe inference queue, and GPU memory tuning to run stably on consumer-grade 11GB VRAM cards (RTX 2080 Ti / RTX 3080).

---

## Features

- Arabic-first TTS with a fixed high-quality reference voice (Mazen Lawand)
- Inline emotion tags: `[pause]`, `[emphasis]`, `[excited]`, `[whisper]`, `[angry]`, and more
- Thread-safe inference queue — handles concurrent requests without GPU race conditions
- WAV audio output at 44100 Hz
- `/health`, `/info`, and `/docs` endpoints out of the box
- Smoke test script included

---

## Hardware Requirements

| Component | Minimum | Tested On |
|-----------|---------|-----------|
| GPU (model) | 11GB VRAM | 8× RTX 2080 Ti |
| GPU (codec) | 5GB VRAM | Same machine, separate GPU |
| RAM | 16GB | — |
| CUDA | 11.8+ | CUDA 12.8 |

The LLM uses ~10.6GB on its GPU. The codec uses ~4.1GB on a second GPU. Both must be CUDA-capable.

---

## Dependencies

### 1. Clone fish-speech

This server depends on the fish-speech library. Clone it into the project root:

```bash
git clone https://github.com/fishaudio/fish-speech.git
```

### 2. Download the S2-Pro model

Download the model checkpoint into `s2-pro/`:

```
s2-pro/
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── chat_template.jinja
└── codec.pth
```

### 3. Add a reference voice

Place your reference voice audio file under `voice/`. Update the path and transcript in `server/main.py`:

```python
VOICE_PATH = PROJECT_ROOT / "voice" / "your-voice-file.mp3"
REFERENCE_TEXT = "The exact transcript of the reference audio."
```

### 4. Install Python dependencies

```bash
python -m venv ~/tts-env
source ~/tts-env/bin/activate
pip install fastapi uvicorn loguru librosa soundfile torch torchaudio
pip install -r fish-speech/requirements.txt
```

---

## Running the Server

```bash
bash server/run.sh
```

Or with explicit GPU assignment:

```bash
TTS_DEVICE_MODEL=cuda:0 TTS_DEVICE_CODEC=cuda:1 bash server/run.sh
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_DEVICE_MODEL` | `cuda:0` | GPU for the LLM (~10.6GB) |
| `TTS_DEVICE_CODEC` | `cuda:1` | GPU for the audio codec (~4.1GB) |
| `TTS_PORT` | `3008` | Server port |
| `TTS_HOST` | `0.0.0.0` | Bind address |

Server startup takes **~3 minutes** (model loading + voice encoding).

---

## API

Interactive docs available at `http://localhost:3008/docs` once the server is running.

### Health Check

```bash
curl http://localhost:3008/health
```

```json
{
  "status": "ready",
  "model_loaded": true,
  "codec_loaded": true,
  "voice_cached": true,
  "device_model": "cuda:0",
  "device_codec": "cuda:1"
}
```

### Text-to-Speech

```bash
curl -X POST http://localhost:3008/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحباً بكم في خدمتنا الصوتية"}' \
  --output speech.wav
```

Request body:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize (max 5000 chars) |
| `max_new_tokens` | int | 1024 | Max tokens to generate |
| `chunk_length` | int | 200 | Text chunk size for iterative prompting |
| `top_p` | float | 0.7 | Nucleus sampling threshold |
| `temperature` | float | 0.7 | Sampling temperature |
| `repetition_penalty` | float | 1.2 | Repetition penalty |

### Emotion Tags

Embed emotion tags anywhere in the text:

```
[emphasis] هذا مهم جداً [pause] استمع بعناية.
[excited] لقد فزنا! [whisper] لكن لا تخبر أحداً.
```

Supported tags: `[pause]`, `[short pause]`, `[emphasis]`, `[excited]`, `[angry]`, `[sad]`, `[whisper]`, `[loud]`, `[low voice]`, `[laughing]`, `[chuckle]`, `[sigh]`, `[inhale]`, `[singing]`, `[shouting]`, `[screaming]`, `[surprised]`, `[clearing throat]`

---

## Smoke Test

```bash
bash testing/test_tts.sh
```

Sends a short Arabic sentence to the server and verifies the WAV output.

---

## GPU Memory Tuning

The KV cache is capped at **2560 tokens** by default — the minimum that satisfies the inference code's internal 2048-token output buffer while keeping the model within 11GB VRAM. Change it in `server/model_loader.py`:

```python
cache_seq_len = min(2560, self.llama_model.config.max_seq_len)
```

| KV cache | VRAM used (cuda:0) | Max prompt tokens |
|----------|--------------------|-------------------|
| 2560 | ~10.6GB | 512 |
| 4096 | ~10.8GB | 2048 (OOM on 11GB during inference) |

---

## Known Limitations

- **Not real-time**: RTF ≈ 6× on RTX 2080 Ti (3.9s of audio takes ~25s to generate). Not suitable for live voice agents.
- **No streaming**: Returns the complete WAV file after generation finishes.
- **Single-threaded inference**: One request at a time; others queue.
- **Fixed voice**: The server uses one reference voice for all requests.

---

## Project Structure

```
FishAudioTTS-Server/
├── server/
│   ├── main.py          # FastAPI application and endpoints
│   ├── model_loader.py  # Model loading, KV cache setup, inference
│   ├── tts_engine.py    # Thread-safe inference queue
│   ├── schemas.py       # Pydantic request/response models
│   └── run.sh           # Startup script
├── fish-s2-pro-gradioApp/  # Gradio web UI (optional)
├── testing/
│   └── test_tts.sh      # Smoke test
├── fish-speech/         # Clone from fishaudio/fish-speech (not tracked)
├── s2-pro/              # Model weights (not tracked, ~11GB)
└── voice/               # Reference voice audio (not tracked)
```

---

## License

Model weights and fish-speech library are subject to the [Fish Audio License](https://github.com/fishaudio/fish-speech/blob/main/LICENSE).
