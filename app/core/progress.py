import json
import logging
from app.core import queue

logger = logging.getLogger("uvicorn.error")

async def publish_progress(document_id: str, message: str, step: str, status: str = "PROCESSING") -> None:
    """Publish a progress update message to the Redis Pub/Sub channel for a document."""
    try:
        if queue.redis_pool is None:
            await queue.connect_redis()
            
        channel = f"document_analysis:{document_id}"
        payload = {
            "documentId": document_id,
            "message": message,
            "step": step,
            "status": status
        }
        
        # Publish payload as JSON string using the live queue pool reference
        await queue.redis_pool.publish(channel, json.dumps(payload))
        logger.info(f"Published progress to {channel}: {message}")
    except Exception as e:
        logger.error(f"Error publishing progress to Redis: {e}")
