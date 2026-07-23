import logging
from typing import Any, Dict, List, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession

from app.core.env import settings

logger = logging.getLogger("uvicorn.error")


class Neo4jGraphStore:
    """Async Neo4j Graph Database Store driver."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self._driver: Optional[AsyncDriver] = None

    async def connect(self) -> AsyncDriver:
        """Initialize and return the async Neo4j driver connection."""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            logger.info(f" Neo4j driver connected to {self.uri}")
        return self._driver

    async def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("🛑 Neo4j driver closed successfully.")

    async def check_connection(self) -> bool:
        """Check if Neo4j instance is accessible."""
        try:
            driver = await self.connect()
            await driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"⚠️ Neo4j connection check failed: {e}")
            return False

    async def init_graph_store(self) -> None:
        """Create initial unique constraints for nodes in Neo4j schema."""
        driver = await self.connect()
        constraints = [
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;",
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE;",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;",
            "CREATE CONSTRAINT requirement_id_unique IF NOT EXISTS FOR (r:Requirement) REQUIRE r.id IS UNIQUE;",
            "CREATE CONSTRAINT task_id_unique IF NOT EXISTS FOR (t:Task) REQUIRE t.id IS UNIQUE;",
            "CREATE CONSTRAINT conflict_id_unique IF NOT EXISTS FOR (c:Conflict) REQUIRE c.id IS UNIQUE;",
        ]

        async with driver.session() as session:
            for constraint_query in constraints:
                try:
                    await session.run(constraint_query)
                except Exception as e:
                    logger.error(f"Error creating constraint query '{constraint_query}': {e}")
                    raise e
        logger.info("✅ Neo4j unique constraints created successfully.")

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return list of records as dicts."""
        driver = await self.connect()
        async with driver.session(database=db) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write_batch(
        self,
        query: str,
        batch: List[Dict[str, Any]],
        db: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a write query using UNWIND batching for duplicate-free bulk writes."""
        if not batch:
            return []
        driver = await self.connect()
        async with driver.session(database=db) as session:
            result = await session.run(query, {"batch": batch})
            records = await result.data()
            return records


# Global singleton instance
neo4j_graph_store = Neo4jGraphStore()
