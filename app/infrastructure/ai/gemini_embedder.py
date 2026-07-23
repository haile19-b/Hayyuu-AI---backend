import logging
from typing import List
from google.genai import types

from app.core.config import genAI

logger = logging.getLogger("uvicorn.error")


class GeminiEmbedder:
    """Gemini gemini-embedding-2 embedding client."""

    def __init__(self):
        self.client = genAI
        self.model = "gemini-embedding-2"

    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding (768-dim) for a single query text string."""
        if not text.strip():
            return [0.0] * 768

        # Task prefix for search query
        prefixed_text = f"task: search query | {text}"
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=prefixed_text,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            # Access embeddings as returned result list
            if hasattr(response, "embeddings") and response.embeddings:
                return list(response.embeddings[0].values)
            elif hasattr(response, "embedding") and response.embedding:
                return list(response.embedding.values)
            else:
                raise ValueError(f"Unexpected response format from Gemini: {response}")
        except Exception as e:
            logger.error(f"Error generating embedding with model {self.model}: {e}")
            raise e

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings (768-dim) for a list of document texts."""
        if not texts:
            return []

        # Prefix all texts for retrieval document task
        prefixed_texts = [f"task: search document | {t}" for t in texts]

        # Wrap each prefixed text as a Content object to ensure separate embeddings are generated
        contents = [
            types.Content(parts=[types.Part.from_text(text=t)])
            for t in prefixed_texts
        ]

        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
            if hasattr(response, "embeddings") and response.embeddings:
                return [list(e.values) for e in response.embeddings]
            else:
                raise ValueError(f"Unexpected response format from batch embedding: {response}")
        except Exception as e:
            logger.warning(f"Batch embedding fallback to single item processing: {e}")
            # Fallback using search document prefix format
            fallback_embeddings = []
            for t in texts:
                fallback_embeddings.append(self.embed_text(t))
            return fallback_embeddings


# Global singleton instance
gemini_embedder = GeminiEmbedder()
