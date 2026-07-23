import sys
import asyncio
import logging
import json
from typing import Any, Dict, List, Optional
import psycopg
from pgvector.psycopg import register_vector_async

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.core.env import settings

logger = logging.getLogger("uvicorn.error")


def _get_clean_db_url() -> str:
    """Ensure database URL is compatible with psycopg."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+psycopg://"):
        url = url.replace("postgresql+psycopg://", "postgresql://", 1)
    elif url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


class PGVectorStore:
    """Async PGVector Store driver for document chunk embeddings."""

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or _get_clean_db_url()

    async def get_connection(self) -> psycopg.AsyncConnection:
        """Establish async connection and register pgvector extension."""
        conn = await psycopg.AsyncConnection.connect(self.db_url)
        try:
            await register_vector_async(conn)
        except psycopg.ProgrammingError:
            # Create extension if not present yet, then register
            async with conn.cursor() as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.commit()
            await register_vector_async(conn)
        return conn

    async def init_vector_store(self) -> None:
        """Initialize PGVector extension, document_chunks table, and indexes."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                # Enable vector extension
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create table for document chunks
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        document_id VARCHAR(255) NOT NULL,
                        project_id VARCHAR(255) NOT NULL,
                        chunk_index INT NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector(768),
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                
                # Create indexes
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);"
                )
                await cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunks_project_id ON document_chunks(project_id);"
                )
                
                await conn.commit()
                logger.info("✅ PGVector store initialized successfully.")

    async def upsert_chunks(
        self,
        document_id: str,
        project_id: str,
        chunks: List[Dict[str, Any]],
    ) -> int:
        """
        Delete existing chunks for a document and insert new ones.
        
        Each chunk dict should contain:
        - chunk_index: int
        - content: str
        - embedding: List[float] (768 dimensions)
        - metadata: Optional[dict]
        """
        if not chunks:
            return 0

        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                # Idempotency: delete old chunks for this document
                await cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s;",
                    (document_id,),
                )

                # Prepare insert batch
                for chunk in chunks:
                    chunk_index = chunk.get("chunk_index", 0)
                    content = chunk.get("content", "")
                    embedding = chunk.get("embedding", [])
                    metadata = json.dumps(chunk.get("metadata", {}))

                    await cur.execute(
                        """
                        INSERT INTO document_chunks (document_id, project_id, chunk_index, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s::jsonb);
                        """,
                        (document_id, project_id, chunk_index, content, embedding, metadata),
                    )

                await conn.commit()
                logger.info(f" Successfully upserted {len(chunks)} chunks for document {document_id}")
                return len(chunks)

    async def search_similar(
        self,
        query_vector: List[float],
        project_id: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k most similar document chunks using cosine distance (<=>).
        Optionally filter by document_id and project_id.
        """
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                if document_id:
                    query = """
                        SELECT id, document_id, project_id, chunk_index, content, metadata,
                               (embedding <=> %s::vector) AS distance
                        FROM document_chunks
                        WHERE project_id = %s AND document_id = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                    """
                    params = (query_vector, project_id, document_id, top_k)
                else:
                    query = """
                        SELECT id, document_id, project_id, chunk_index, content, metadata,
                               (embedding <=> %s::vector) AS distance
                        FROM document_chunks
                        WHERE project_id = %s
                        ORDER BY distance ASC
                        LIMIT %s;
                    """
                    params = (query_vector, project_id, top_k)

                await cur.execute(query, params)
                rows = await cur.fetchall()

                results = []
                for row in rows:
                    distance = float(row[6])
                    similarity = max(0.0, 1.0 - distance)
                    results.append(
                        {
                            "id": str(row[0]),
                            "document_id": row[1],
                            "project_id": row[2],
                            "chunk_index": row[3],
                            "content": row[4],
                            "metadata": row[5],
                            "distance": distance,
                            "similarity": similarity,
                        }
                    )
                return results

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_chunks WHERE document_id = %s;",
                    (document_id,),
                )
                deleted_count = cur.rowcount
                await conn.commit()
                logger.info(f" Deleted {deleted_count} chunks for document {document_id}")
                return deleted_count

    async def delete_project_chunks(self, project_id: str) -> int:
        """Delete all chunks belonging to a project."""
        async with await self.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_chunks WHERE project_id = %s;",
                    (project_id,),
                )
                deleted_count = cur.rowcount
                await conn.commit()
                logger.info(f" Deleted {deleted_count} chunks for project {project_id}")
                return deleted_count


# Global singleton instance
pgvector_store = PGVectorStore()
