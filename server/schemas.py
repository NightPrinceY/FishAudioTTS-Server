"""
Pydantic schemas for the Fish Audio S2-Pro TTS API.

This module defines the request and response models used by the API endpoints.
All models are designed for production use with proper validation and documentation.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AudioFormat(str, Enum):
    """Supported audio output formats."""
    WAV = "wav"
    # Future: MP3, OGG, FLAC


class TTSRequest(BaseModel):
    """
    Text-to-Speech synthesis request.

    The server uses a fixed voice (Mazen Lawand - deep and professional Arabic voice).
    Only the text field is required — all other parameters have sensible defaults
    optimized for Arabic speech synthesis.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The text to synthesize into speech. Supports Arabic, English, and 80+ languages. "
                    "Emotion tags like [pause], [emphasis], [excited] can be embedded inline.",
        examples=[
            "مرحباً بكم في خدمتنا الصوتية",
            "Hello, welcome to our service.",
            "[excited] هذا رائع حقاً!",
        ],
    )

    max_new_tokens: int = Field(
        default=1024,
        ge=0,
        le=2048,
        description="Maximum number of new tokens to generate. 0 means automatic (model decides). "
                    "Higher values allow longer audio output.",
    )

    chunk_length: int = Field(
        default=200,
        ge=100,
        le=500,
        description="Chunk length for iterative prompt processing. Controls how the text "
                    "is split for generation. Lower values = more stable but slower.",
    )

    top_p: float = Field(
        default=0.7,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling probability. Controls diversity of the output. "
                    "Lower values = more deterministic.",
    )

    temperature: float = Field(
        default=0.7,
        gt=0.0,
        lt=2.0,
        description="Sampling temperature. Controls randomness. "
                    "Lower = more consistent, higher = more expressive.",
    )

    repetition_penalty: float = Field(
        default=1.2,
        ge=0.9,
        le=2.0,
        description="Repetition penalty to reduce repeated patterns in generated speech.",
    )

    format: AudioFormat = Field(
        default=AudioFormat.WAV,
        description="Output audio format.",
    )


class TTSHealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Server status")
    model_loaded: bool = Field(description="Whether the TTS model is loaded and ready")
    codec_loaded: bool = Field(description="Whether the audio codec is loaded and ready")
    voice_cached: bool = Field(description="Whether the reference voice tokens are cached")
    device_model: str = Field(description="GPU device used for the main model")
    device_codec: str = Field(description="GPU device used for the codec")


class TTSErrorResponse(BaseModel):
    """Error response returned when synthesis fails."""
    error: str = Field(description="Error type")
    detail: str = Field(description="Human-readable error description")


class TTSInfoResponse(BaseModel):
    """Server information response."""
    name: str = "Fish Audio S2-Pro TTS Server"
    version: str = "1.0.0"
    model: str = "fishaudio/s2-pro"
    voice: str = "Mazen Lawand - Deep and Professional (Arabic)"
    sample_rate: int = 44100
    supported_formats: list[str] = ["wav"]
    supported_tags: list[str] = [
        "[pause]", "[emphasis]", "[laughing]", "[inhale]", "[chuckle]",
        "[singing]", "[excited]", "[angry]", "[low volume]", "[sigh]",
        "[low voice]", "[whisper]", "[screaming]", "[shouting]", "[loud]",
        "[surprised]", "[short pause]", "[sad]", "[clearing throat]",
    ]
    max_text_length: int = 5000
