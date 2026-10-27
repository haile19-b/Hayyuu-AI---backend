import sys
import os
import logging
import argparse
from typing import Optional
from contextlib import asynccontextmanager

# Ensure asyncio Windows compatibility for psycopg/neo4j
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from mcp.server import MCPServer
from mcp.server.sse import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware.cors import CORSMiddleware

from app.core.database import connect_db, disconnect_db
from app.infrastructure.graph_store.neo4j import neo4j_graph_store

from app.infrastructure.mcp.tools.db_graph_tools import register_db_graph_tools
from app.infrastructure.mcp.tools.curated_backend_tools import register_curated_backend_tools
from app.infrastructure.mcp.tools.github_tools import register_github_tools
from app.infrastructure.mcp.resources.project_resources import register_project_resources
from app.infrastructure.mcp.prompts.agent_prompts import register_agent_prompts

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def app_lifespan(server_or_app=None):
    """Initializes and tears down backing database and graph store connections."""
    logger.info("[mcp: server] Initializing database connections (Prisma + Neo4j)...")
    await connect_db()
    await neo4j_graph_store.connect()
    try:
        yield
    finally:
        logger.info("[mcp: server] Shutting down database connections...")
        await disconnect_db()
        await neo4j_graph_store.close()


def create_mcp_server(name: str = "hayyuu-mcp") -> MCPServer:
    """
    Factory function instantiating and configuring a production-ready MCPServer
    with Tools, Resources, and Prompts.
    """
    server = MCPServer(name, lifespan=app_lifespan)

    # 1. Register Database & Graph Tools
    register_db_graph_tools(server)

    # 2. Register Curated FastAPI Backend Tools
    register_curated_backend_tools(server)

    # 3. Register Multi-Tenant GitHub Tools Suite (21 tools)
    register_github_tools(server)

    # 4. Register Contextual Resources
    register_project_resources(server)

    # 5. Register Agent Prompts
    register_agent_prompts(server)

    logger.info(f"Initialized MCPServer '{name}' with Tools, Resources, and Prompts.")
    return server


# Default singleton instance
mcp = create_mcp_server()


def create_sse_app(server: Optional[MCPServer] = None) -> Starlette:
    """
    Wraps the MCP server's SSE application with a health check endpoint,
    CORS middleware, and database lifecycle management for standalone Docker execution.
    """
    target_server = server or mcp
    # Disable DNS rebinding check for container networking (Host: mcp:8001)
    security_settings = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    )
    sse_app = target_server.sse_app(transport_security=security_settings)

    async def health_check(request):
        tools = await target_server.list_tools() if hasattr(target_server, "list_tools") else []
        return JSONResponse({
            "status": "healthy",
            "service": "hayyuu-mcp",
            "version": "2.0.0",
            "tools_count": len(tools),
        })

    # Add health route to Starlette app
    sse_app.routes.append(Route("/health", health_check, methods=["GET"]))

    # Attach database lifespan context to Starlette router
    sse_app.router.lifespan_context = app_lifespan

    # Add CORS middleware
    sse_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return sse_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hayyuu AI Standalone MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport layer protocol: 'stdio' (subprocess) or 'sse' (HTTP server)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HOST", "0.0.0.0"),
        help="Host interface for SSE server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_PORT", "8001")),
        help="Port for SSE server (default 8001)"
    )

    args = parser.parse_args()

    if args.transport == "sse":
        import uvicorn
        logger.info(f"🚀 Starting Hayyuu MCP Server on http://{args.host}:{args.port}/sse")
        app = create_sse_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.info("🔌 Starting Hayyuu MCP Server using stdio transport")
        mcp.run(transport="stdio")
