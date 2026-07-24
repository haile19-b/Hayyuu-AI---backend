import logging
import time
import re
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
        """Generate vector embeddings (768-dim) for a list of document texts, chunking batches to max 100 items."""
        if not texts:
            return []

        # Prefix all texts for retrieval document task
        prefixed_texts = [f"task: search document | {t}" for t in texts]

        # Chunk the list of inputs into batches of 100 to comply with Gemini batch limit
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i : i + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in batch_texts
            ]

            response = None
            max_attempts = 4
            for attempt in range(max_attempts):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=contents,
                        config=types.EmbedContentConfig(output_dimensionality=768),
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_attempts - 1:
                        sleep_time = 15
                        # Attempt to parse specific retry delay from Gemini response
                        match = re.search(r"retry in ([\d\.]+)s", err_str)
                        if match:
                            sleep_time = int(float(match.group(1))) + 1
                        logger.warning(f"Rate limit (429) hit on batch {i // batch_size}. Sleeping for {sleep_time}s before retry (attempt {attempt + 1}/{max_attempts})...")
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Batch embedding failed on attempt {attempt + 1}: {e}")
                        if attempt == max_attempts - 1:
                            # Fallback sequentially only if all batch retry attempts failed, with pacing delays
                            logger.warning("Falling back to paced sequential embedding.")
                            for t in batch_texts:
                                for seq_attempt in range(3):
                                    try:
                                        res = self.client.models.embed_content(
                                            model=self.model,
                                            contents=t,
                                            config=types.EmbedContentConfig(output_dimensionality=768),
                                        )
                                        if hasattr(res, "embeddings") and res.embeddings:
                                            all_embeddings.append(list(res.embeddings[0].values))
                                        elif hasattr(res, "embedding") and res.embedding:
                                            all_embeddings.append(list(res.embedding.values))
                                        else:
                                            all_embeddings.append([0.0] * 768)
                                        # Pace requests to avoid rate limits
                                        time.sleep(0.3)
                                        break
                                    except Exception as seq_err:
                                        if ("429" in str(seq_err) or "RESOURCE_EXHAUSTED" in str(seq_err)) and seq_attempt < 2:
                                            logger.warning("Rate limit hit during sequential fallback. Sleeping 12s...")
                                            time.sleep(12)
                                        else:
                                            logger.error(f"Sequential fallback failed: {seq_err}")
                                            all_embeddings.append([0.0] * 768)
                                            break
                            response = None
                            break
                        raise e

            if response:
                if hasattr(response, "embeddings") and response.embeddings:
                    all_embeddings.extend([list(e.values) for e in response.embeddings])
                else:
                    raise ValueError(f"Unexpected response format from batch embedding: {response}")

        return all_embeddings


# Global singleton instance
gemini_embedder = GeminiEmbedder()
