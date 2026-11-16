"""Centralized Multi-Model LLM Cascading & Resilience Engine.

Provides unified execution wrappers for Google GenAI SDK calls with:
- Automatic model priority cascading (e.g. gemini-3.5-flash -> gemini-3.5-flash-lite -> gemini-3.8-flash)
- Transparent failover on transient errors (429 Quota Exhausted, 503 Service Unavailable, 504 Gateway Timeout, 404 Model Retired)
- Strict client-side and server-side deadline management (meeting Google's >= 10s constraint)
- Both async (standard/streaming) and synchronous cascading interfaces
"""

from __future__ import annotations
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from google.genai import types

from app.core.config import genAI
from app.core.env import settings

logger = logging.getLogger("uvicorn.error")

DEFAULT_MODELS: List[str] = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]


def get_candidate_models(explicit_model: Optional[str] = None) -> List[str]:
    """Returns an ordered, deduplicated list of candidate Gemini models."""
    configured = list(settings.GEMINI_MODELS) if getattr(settings, "GEMINI_MODELS", None) else DEFAULT_MODELS
    candidates: List[str] = []
    seen = set()

    if explicit_model and explicit_model.strip():
        m = explicit_model.strip()
        candidates.append(m)
        seen.add(m)

    for m in configured:
        if m and m not in seen:
            candidates.append(m)
            seen.add(m)

    return candidates or DEFAULT_MODELS


def _ensure_config_timeout(
    config: Optional[types.GenerateContentConfig],
    timeout_seconds: float
) -> types.GenerateContentConfig:
    """Clones or updates GenerateContentConfig with valid server-side HTTP deadline >= 10s."""
    timeout_ms = max(10000, int(timeout_seconds * 1000))
    if config is None:
        return types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=timeout_ms)
        )

    # If http_options missing or timeout < 10000ms, set valid timeout
    if not config.http_options or (config.http_options.timeout and config.http_options.timeout < 10000):
        config.http_options = types.HttpOptions(timeout=timeout_ms)

    return config


async def generate_content_cascade(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    candidate_models: Optional[List[str]] = None,
    timeout_seconds: float = 20.0,
    max_models: int = 3,
) -> Any:
    """
    Executes an async Gemini generate_content call with automatic model cascading.
    Tries each candidate model sequentially; if a model encounters 429, 503, 504,
    or times out, it gracefully cascades to the next candidate model.
    """
    models = candidate_models or get_candidate_models()
    models_to_try = models[:max_models]
    call_config = _ensure_config_timeout(config, timeout_seconds)

    last_error: Optional[Exception] = None

    for idx, model_name in enumerate(models_to_try, 1):
        try:
            coro = genAI.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=call_config,
            )
            response = await asyncio.wait_for(coro, timeout=timeout_seconds)
            return response
        except Exception as e:
            last_error = e
            logger.warning(
                f"[AI Cascade] Model '{model_name}' (attempt {idx}/{len(models_to_try)}) failed ({e}). "
                f"{'Cascading to next candidate...' if idx < len(models_to_try) else 'All candidates exhausted.'}"
            )
            continue

    logger.error(f"[AI Cascade] All candidate models failed: {models_to_try}. Final error: {last_error}")
    if last_error:
        raise last_error
    raise RuntimeError("All candidate models failed to generate content.")


def generate_content_sync_cascade(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    candidate_models: Optional[List[str]] = None,
    timeout_seconds: float = 20.0,
    max_models: int = 3,
) -> Any:
    """
    Synchronous version of generate_content_cascade for sync nodes and background threads.
    """
    models = candidate_models or get_candidate_models()
    models_to_try = models[:max_models]
    call_config = _ensure_config_timeout(config, timeout_seconds)

    last_error: Optional[Exception] = None

    for idx, model_name in enumerate(models_to_try, 1):
        try:
            response = genAI.models.generate_content(
                model=model_name,
                contents=contents,
                config=call_config,
            )
            return response
        except Exception as e:
            last_error = e
            logger.warning(
                f"[AI Cascade Sync] Model '{model_name}' (attempt {idx}/{len(models_to_try)}) failed ({e}). "
                f"{'Cascading to next candidate...' if idx < len(models_to_try) else 'All candidates exhausted.'}"
            )
            continue

    logger.error(f"[AI Cascade Sync] All candidate models failed: {models_to_try}. Final error: {last_error}")
    if last_error:
        raise last_error
    raise RuntimeError("All candidate models failed to generate content synchronously.")


async def generate_content_stream_cascade(
    contents: Any,
    config: Optional[types.GenerateContentConfig] = None,
    candidate_models: Optional[List[str]] = None,
    timeout_first_chunk: float = 25.0,
    max_models: int = 3,
) -> AsyncGenerator[Tuple[Any, str], None]:
    """
    Streams content chunks with automatic model cascading on startup / first-token timeout.
    Yields (chunk, active_model_name).
    """
    models = candidate_models or get_candidate_models()
    models_to_try = models[:max_models]
    call_config = _ensure_config_timeout(config, timeout_first_chunk)

    last_error: Optional[Exception] = None
    stream_successful = False

    for idx, model_name in enumerate(models_to_try, 1):
        response_stream = None
        try:
            stream_coro = genAI.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=call_config,
            )
            # Bound first token arrival
            response_stream = await asyncio.wait_for(stream_coro, timeout=timeout_first_chunk)

            chunk_iter = response_stream.__aiter__()
            first_chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=timeout_first_chunk)
            yield first_chunk, model_name
            stream_successful = True

            # Stream remaining chunks with bounded per-chunk timeout
            while True:
                try:
                    chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=15.0)
                    yield chunk, model_name
                except StopAsyncIteration:
                    break
            return

        except StopAsyncIteration:
            return
        except Exception as e:
            last_error = e
            if not stream_successful:
                logger.warning(
                    f"[AI Cascade Stream] Model '{model_name}' failed before streaming ({e}). "
                    f"{'Cascading to next candidate...' if idx < len(models_to_try) else 'All candidates exhausted.'}"
                )
            else:
                logger.error(f"[AI Cascade Stream] Mid-stream failure on '{model_name}': {e}")
                raise e
        finally:
            if response_stream is not None and hasattr(response_stream, "aclose"):
                try:
                    await response_stream.aclose()
                except Exception:
                    pass

    if not stream_successful:
        logger.error(f"[AI Cascade Stream] All candidate models failed to stream: {models_to_try}. Error: {last_error}")
        if last_error:
            raise last_error
        raise RuntimeError("All candidate models failed to stream content.")
