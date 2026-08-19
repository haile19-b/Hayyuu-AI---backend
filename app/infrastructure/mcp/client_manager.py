import os
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("uvicorn.error")

class MCPClientManager:
    """
    Manages connections to multiple MCP servers as subprocesses,
    discovering their tools and routing calls dynamically.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = config_path
        else:
            # Resolve default path relative to this file:
            # app/infrastructure/mcp/client_manager.py -> app/core/mcp_config.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.abspath(os.path.join(current_dir, "..", "..", "core", "mcp_config.json"))
            
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[Dict[str, Any]] = []

    async def connect_all(self) -> None:
        """Reads configuration and connects to all declared MCP servers."""
        logger.info(f"[mcp: client] Loading MCP server configuration from: {self.config_path}")
        
        if not os.path.exists(self.config_path):
            logger.error(f"[mcp: client] Configuration file not found at {self.config_path}")
            return

        try:
            with open(self.config_path, "r") as file:
                config = json.load(file)
            
            servers = config.get("mcpServers", {})
            if not servers:
                logger.warning("[mcp: client] No servers configured in mcp_config.json.")
                return

            for server_name, server_config in servers.items():
                await self.connect_server(server_name, server_config)
                
            logger.info(f"✅ Registered {len(self.available_tools)} MCP tools dynamically.")
        except Exception as e:
            logger.error(f"[mcp: client] Error loading server configuration: {e}")
            raise

    async def connect_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        """Connects to a single MCP server using stdio transport."""
        logger.info(f"[mcp: client] Connecting to server '{server_name}'...")
        try:
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env")
            
            # Resolve environment variable mappings
            if env:
                # Merge with current system environment
                merged_env = os.environ.copy()
                merged_env.update(env)
                env = merged_env

            if not command:
                logger.error(f"[mcp: client] Server '{server_name}' configuration is missing 'command'.")
                return

            # Instantiate server parameters
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )

            # Connect transport
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            
            # Create client session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Initialize connection handshake
            await session.initialize()
            self.sessions[server_name] = session
            
            # Retrieve tools list
            response = await session.list_tools()
            tools = response.tools
            logger.info(f"🔌 Connected to '{server_name}' exposing tools: {[t.name for t in tools]}")
            
            for tool in tools:
                # Store dynamic mapping for routing calls
                self.tool_to_session[tool.name] = session
                
                # Normalize tool input schema to support multiple SDK versions
                input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
                
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": input_schema or {"type": "object", "properties": {}}
                })
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server '{server_name}': {e}")
            # Do not crash the entire app if one server fails (per FR-K-001 grace-degrade)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Calls a tool by name on the corresponding MCP server session."""
        session = self.tool_to_session.get(tool_name)
        if not session:
            raise ValueError(f"Tool '{tool_name}' is not registered with any active MCP server.")
            
        logger.info(f"[mcp: client] Invoking tool '{tool_name}' with args {arguments}")
        result = await session.call_tool(tool_name, arguments=arguments)
        
        # Check result payload structure and extract content
        if hasattr(result, "content") and result.content:
            # Standard MCP text block response
            text_contents = [c.text for c in result.content if hasattr(c, "text") and c.text]
            if text_contents:
                return "\n".join(text_contents)
            return str(result.content)
            
        return str(result)

    async def close(self) -> None:
        """Cleanly close all transport and sessions using ExitStack."""
        logger.info("[mcp: client] Closing all MCP server connection sessions...")
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tool_to_session.clear()
        self.available_tools.clear()
        logger.info("🛑 All MCP connections closed successfully.")
