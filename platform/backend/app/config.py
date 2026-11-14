"""Application configuration.

All values can be overridden with environment variables so the same image can be
deployed across dev / staging / production.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "CardioPulmo AI"
    api_prefix: str = "/api"
    version: str = "0.1.0"

    # Upload limits
    max_upload_mb: int = 50
    allowed_audio_suffixes: tuple[str, ...] = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a",
    )

    # Signal processing
    sample_rate: int = 8000
    hss_band: tuple[float, float] = (20.0, 200.0)
    lss_cutoff: float = 150.0

    # AI agent (free-tier, pluggable). Order = priority.
    # ollama -> local & unlimited; groq/gemini/openrouter -> free tier keys.
    agent_provider_order: tuple[str, ...] = ("ollama", "groq", "gemini", "openrouter", "offline")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

    # Model weight cache
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/tmp/cardiopulmo_models")

    class Config:
        frozen = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
