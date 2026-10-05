import logging
import asyncio
import random
from typing import Any, List, Optional
from google.genai import errors
from app.core.config import genAI
from app.core.env import settings
from app.core.progress import publish_progress

logger = logging.getLogger("uvicorn.error")

def _is_transient_error(e: Exception) -> bool:
    """Check if an exception represents a temporary, retryable server-side capacity/rate-limit error."""
    err_str = str(e).lower()
    
    # Check for HTTP status codes / errors from Google GenAI SDK
    if isinstance(e, errors.APIError):
        # 503 Service Unavailable, 429 Too Many Requests, 500 Internal Server Error
        if e.code in [503, 429, 500]:
            return True
            
    # Check error message strings for high demand or quota
    transient_indicators = [
        "503",
        "service_unavailable",
        "high demand",
        "spikes in demand",
        "try again later",
        "429",
        "rate limit",
        "resource_exhausted",
        "temporarily unavailable",
        "connection reset",
        "timed out",
    ]
    return any(indicator in err_str for indicator in transient_indicators)


async def call_gemini_with_fallback(
    input_contents: list,
    response_schema: Any,
    doc_id: Optional[str] = None,
    step_name: str = "agent_inference",
    temperature: float = 0.2,
    max_retries_per_model: int = 2,
    base_delay: float = 2.0,
) -> str:
    """
    Executes a structured Gemini interaction with:
    1. Automatic model fallback cascade from settings.GEMINI_MODELS (primary -> secondary -> tertiary).
    2. Exponential backoff with jitter on transient errors (503 High Demand, 429 Rate Limits, 500 Server Errors).
    3. Real-time progress updates published to Redis via publish_progress.
    """
    models = settings.GEMINI_MODELS if settings.GEMINI_MODELS else ["gemini-3.8-flash", "gemini-2.5-flash"]
    last_error: Optional[Exception] = None
    
    for model_index, model_name in enumerate(models):
        is_fallback = model_index > 0
        if is_fallback and doc_id:
            logger.warning(f"Promoting fallback model: '{model_name}' for document {doc_id}")
            await publish_progress(
                doc_id,
                f"Previous model experiencing high traffic. Switching to fallback model '{model_name}'...",
                step_name
            )
            
        for attempt in range(1, max_retries_per_model + 1):
            try:
                logger.info(f"Invoking model '{model_name}' (attempt {attempt}/{max_retries_per_model}) for doc={doc_id}")
                response = await genAI.aio.interactions.create(
                    model=model_name,
                    input=input_contents,
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": response_schema.model_json_schema()
                    }
                )
                return response.output_text
            except Exception as e:
                last_error = e
                is_transient = _is_transient_error(e)
                
                if not is_transient:
                    # Non-transient error (e.g. 400 Bad Request, schema validation failure), do not retry this model
                    logger.error(f"Non-transient error calling model '{model_name}': {e}")
                    break
                    
                logger.warning(
                    f"Model '{model_name}' encountered transient error (attempt {attempt}/{max_retries_per_model}): {e}"
                )
                
                if attempt < max_retries_per_model:
                    # Exponential backoff with random jitter: base_delay * 2^(attempt-1) + jitter(0, 1)
                    jitter = random.uniform(0.1, 1.0)
                    delay = (base_delay * (2 ** (attempt - 1))) + jitter
                    
                    if doc_id:
                        await publish_progress(
                            doc_id,
                            f"Model '{model_name}' is experiencing high demand. Retrying in {delay:.1f}s (attempt {attempt}/{max_retries_per_model})...",
                            step_name
                        )
                    await asyncio.sleep(delay)
                else:
                    logger.warning(f"Exhausted {max_retries_per_model} retries for model '{model_name}'.")
                    
    # If all models in the cascade failed
    error_msg = f"All Gemini models in cascade failed ({', '.join(models)}). Last error: {last_error}"
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error
