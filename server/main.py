"""
Fish Audio S2-Pro TTS Server — FastAPI Application

A production-grade Text-to-Speech API server powered by Fish Audio's S2-Pro model.
Uses a fixed Arabic voice (Mazen Lawand - deep and professional) for all synthesis.

Usage:
    source ~/CollegeX/bin/activate
    PYTHONPATH="/path/to/fish-speech:$PYTHONPATH" python -m server.main

    Or simply:
    bash server/run.sh
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from loguru import logger

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
FISH_SPEECH_DIR = PROJECT_ROOT / "fish-speech"
CHECKPOINT_DIR = PROJECT_ROOT / "s2-pro"
VOICE_PATH = PROJECT_ROOT / "voice" / "voice_preview_mazen lawand - deep and professional.mp3"

# The exact transcript of the reference voice
REFERENCE_TEXT = "صوت يعكس الثقة والدفء في الوقت نفسه. هذا الأداء العربي يناسب المشاريع التي تحتاج إلى نبرة مؤثرة تنقل الرسالة بصدق واحتراف."

# GPU configuration (selected based on availability analysis)
DEVICE_MODEL = os.environ.get("TTS_DEVICE_MODEL", "cuda:0")
DEVICE_CODEC = os.environ.get("TTS_DEVICE_CODEC", "cuda:1")

SERVER_HOST = os.environ.get("TTS_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("TTS_PORT", "3008"))

# ──────────────────────────────────────────────────────────────
# Ensure fish-speech is importable
# ──────────────────────────────────────────────────────────────

if str(FISH_SPEECH_DIR) not in sys.path:
    sys.path.insert(0, str(FISH_SPEECH_DIR))

from server.model_loader import TTSModelManager
from server.tts_engine import TTSEngine, InferenceRequest
from server.schemas import (
    TTSRequest,
    TTSHealthResponse,
    TTSInfoResponse,
    TTSErrorResponse,
)

# ──────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────

model_manager: TTSModelManager = None
engine: TTSEngine = None


# ──────────────────────────────────────────────────────────────
# Application lifecycle
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    global model_manager, engine

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Fish Audio S2-Pro TTS Server Starting   ║")
    logger.info("╚══════════════════════════════════════════╝")

    # Validate paths
    if not CHECKPOINT_DIR.exists():
        logger.error(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    if not VOICE_PATH.exists():
        logger.error(f"Reference voice not found: {VOICE_PATH}")
        raise FileNotFoundError(f"Reference voice not found: {VOICE_PATH}")

    # Initialize model manager
    model_manager = TTSModelManager(
        checkpoint_dir=str(CHECKPOINT_DIR),
        device_model=DEVICE_MODEL,
        device_codec=DEVICE_CODEC,
        reference_audio_path=str(VOICE_PATH),
        reference_text=REFERENCE_TEXT,
    )

    # Load models (this takes time — ~60-120s for the large model)
    model_manager.load()

    # Start the inference engine
    engine = TTSEngine(model_manager=model_manager, max_queue_size=20)
    engine.start()

    logger.info(f"Server ready at http://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"API docs at http://{SERVER_HOST}:{SERVER_PORT}/docs")

    yield  # Server is running

    # Shutdown
    logger.info("Shutting down TTS engine...")
    if engine:
        engine.stop()
    logger.info("Server shutdown complete")


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fish Audio S2-Pro TTS Server",
    description=(
        "Production-grade Arabic Text-to-Speech API powered by Fish Audio S2-Pro. "
        "Uses a fixed Arabic voice (Mazen Lawand - deep and professional) for all synthesis. "
        "Supports 80+ languages with inline emotion tags."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    return response


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return JSONResponse(
        content={
            "message": "Fish Audio S2-Pro TTS Server",
            "docs": "/docs",
            "health": "/health",
        }
    )


@app.get(
    "/health",
    response_model=TTSHealthResponse,
    summary="Health Check",
    description="Check the operational status of the TTS server and all loaded models.",
    tags=["System"],
)
async def health_check():
    """Health check endpoint for monitoring and load balancers."""
    is_ready = model_manager is not None and model_manager.is_loaded
    return TTSHealthResponse(
        status="ready" if is_ready else "loading",
        model_loaded=model_manager.is_loaded if model_manager else False,
        codec_loaded=model_manager.codec_model is not None if model_manager else False,
        voice_cached=model_manager.cached_prompt_tokens is not None if model_manager else False,
        device_model=DEVICE_MODEL,
        device_codec=DEVICE_CODEC,
    )


@app.get(
    "/info",
    response_model=TTSInfoResponse,
    summary="Server Information",
    description="Get information about the TTS model, supported features, and configuration.",
    tags=["System"],
)
async def server_info():
    """Return server configuration and capabilities."""
    return TTSInfoResponse(
        sample_rate=model_manager.sample_rate if model_manager else 44100,
    )


@app.post(
    "/v1/tts",
    summary="Text-to-Speech Synthesis",
    description=(
        "Convert text to speech using the Fish Audio S2-Pro model with the fixed "
        "Mazen Lawand Arabic voice. Returns audio in WAV format.\n\n"
        "**Emotion tags**: You can embed emotion/prosody tags anywhere in the text:\n"
        "- `[pause]`, `[emphasis]`, `[excited]`, `[whisper]`, `[angry]`, `[sad]`\n"
        "- Or free-form: `[professional broadcast tone]`, `[low voice]`\n\n"
        "**Example**: `[emphasis] مرحباً بكم [pause] في خدمتنا الصوتية`"
    ),
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Synthesized audio in WAV format",
        },
        422: {"model": TTSErrorResponse, "description": "Validation error"},
        500: {"model": TTSErrorResponse, "description": "Inference error"},
        503: {"model": TTSErrorResponse, "description": "Server not ready"},
    },
    tags=["TTS"],
)
async def synthesize(request: TTSRequest):
    """
    Synthesize speech from text.

    Returns raw WAV audio bytes with Content-Type: audio/wav.
    The response can be directly played or saved as a .wav file.
    """
    # Ensure server is ready
    if not model_manager or not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Server is still loading models. Please wait and retry.",
        )

    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not initialized.",
        )

    # Log the request
    text_preview = request.text[:80] + "..." if len(request.text) > 80 else request.text
    logger.info(f"TTS request: \"{text_preview}\" ({len(request.text)} chars)")

    # Submit to inference engine
    inference_req = InferenceRequest(
        text=request.text,
        max_new_tokens=request.max_new_tokens,
        chunk_length=request.chunk_length,
        top_p=request.top_p,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
        format=request.format.value,
    )

    result = engine.synthesize(inference_req, timeout=300.0)

    if not result.success:
        logger.error(f"TTS failed: {result.error}")
        raise HTTPException(
            status_code=500,
            detail=result.error or "Unknown inference error",
        )

    # Return audio response
    content_type = "audio/wav"
    filename = "speech.wav"

    return Response(
        content=result.audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Audio-Duration": f"{result.duration_seconds:.2f}s",
            "X-Generation-Time": f"{result.generation_time:.2f}s",
            "X-Sample-Rate": str(result.sample_rate),
        },
    )


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {SERVER_HOST}:{SERVER_PORT}")

    uvicorn.run(
        "server.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        workers=1,  # Must be 1 — GPU models cannot be shared across processes
        log_level="info",
        access_log=True,
    )
