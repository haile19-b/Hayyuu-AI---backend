"""
Backward-compatibility entrypoint for Hayyuu DB & Graph MCP Server.
Redirects to the modular app.infrastructure.mcp.server implementation.
"""
import sys
import logging

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.infrastructure.mcp.server import mcp

logger = logging.getLogger("uvicorn.error")

if __name__ == "__main__":
    logger.info("🔌 Launching Hayyuu MCP Server via legacy db_graph_server entrypoint...")
    mcp.run(transport="stdio")
