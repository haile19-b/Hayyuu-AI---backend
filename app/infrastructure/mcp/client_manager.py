import os
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger("uvicorn.error")


class MCPClientManager:
    """
    Manages connections to multiple MCP servers across transports:
    - SSE (Remote/Docker HTTP microservices: mcp.client.sse)
    - Stdio (Subprocess commands: mcp.client.stdio)

    Discovers Tools, Resources, and Prompts dynamically and routes calls.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config_path = config_path
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.abspath(os.path.join(current_dir, "..", "..", "core", "mcp_config.json"))

        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.resource_to_session: Dict[str, ClientSession] = {}
        self.prompt_to_session: Dict[str, ClientSession] = {}

        self.available_tools: List[Dict[str, Any]] = []
        self.available_resources: List[Dict[str, Any]] = []
        self.available_resource_templates: List[Dict[str, Any]] = []
        self.available_prompts: List[Dict[str, Any]] = []

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

            # 1. Check for standalone container / remote URL environment overrides
            remote_url = os.getenv("HAYYUU_MCP_URL") or os.getenv("MCP_SERVER_URL")
            if remote_url:
                logger.info(f"[mcp: client] Discovered remote MCP environment override: {remote_url}")
                await self.connect_server("hayyuu-mcp", {
                    "transport": "sse",
                    "url": remote_url
                })

            # 2. Connect configured servers from mcp_config.json
            for server_name, server_config in servers.items():
                if server_name in self.sessions:
                    continue
                await self.connect_server(server_name, server_config)

            logger.info(
                f"✅ Connected to MCP: {len(self.available_tools)} tools, "
                f"{len(self.available_resources) + len(self.available_resource_templates)} resources, "
                f"{len(self.available_prompts)} prompts discovered."
            )
        except Exception as e:
            logger.error(f"[mcp: client] Error loading server configuration: {e}")
            raise

    async def connect_server(self, server_name: str, server_config: Dict[str, Any]) -> None:
        """Connects to a single MCP server using SSE or Stdio transport."""
        logger.info(f"[mcp: client] Connecting to server '{server_name}'...")
        try:
            transport_type = server_config.get("transport")
            url = server_config.get("url")
            command = server_config.get("command")

            # Expand environment variables in URL if any (e.g. ${MCP_SERVER_URL})
            if url:
                url = os.path.expandvars(url)

            # Auto-detect transport type if not explicitly set
            if not transport_type:
                transport_type = "sse" if url else "stdio"

            if transport_type.lower() == "sse":
                if not url:
                    logger.error(f"[mcp: client] Server '{server_name}' configured for SSE but missing 'url'.")
                    return

                logger.info(f"[mcp: client] Connecting to '{server_name}' via SSE at: {url}")
                headers = server_config.get("headers")
                sse_transport = await self.exit_stack.enter_async_context(
                    sse_client(url, headers=headers)
                )
                read, write = sse_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )

            elif transport_type.lower() == "stdio":
                if not command:
                    logger.error(f"[mcp: client] Server '{server_name}' configured for Stdio but missing 'command'.")
                    return

                args = server_config.get("args", [])
                env = server_config.get("env")
                if env:
                    merged_env = os.environ.copy()
                    merged_env.update(env)
                    env = merged_env

                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                read, write = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
            else:
                logger.error(f"[mcp: client] Unknown transport '{transport_type}' for server '{server_name}'.")
                return

            # Initialize protocol handshake
            await session.initialize()
            self.sessions[server_name] = session

            # 1. Discover Tools
            try:
                response = await session.list_tools()
                tools = getattr(response, "tools", [])
                logger.info(f"🔌 Connected to '{server_name}' exposing tools: {[t.name for t in tools]}")
                for tool in tools:
                    self.tool_to_session[tool.name] = session
                    input_schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None)
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": input_schema or {"type": "object", "properties": {}},
                        "server": server_name,
                    })
            except Exception as tool_err:
                logger.warning(f"[mcp: client] Could not list tools for '{server_name}': {tool_err}")

            # 2. Discover Static Resources & Dynamic Resource Templates
            try:
                res_resp = await session.list_resources()
                for r in getattr(res_resp, "resources", []):
                    self.resource_to_session[r.uri] = session
                    self.available_resources.append({
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description or "",
                        "mime_type": r.mime_type,
                        "server": server_name,
                    })
            except Exception:
                pass

            try:
                tmpl_resp = await session.list_resource_templates()
                for t in getattr(tmpl_resp, "resource_templates", []):
                    self.resource_to_session[t.uri_template] = session
                    self.available_resource_templates.append({
                        "uri_template": t.uri_template,
                        "name": t.name,
                        "description": t.description or "",
                        "mime_type": t.mime_type,
                        "server": server_name,
                    })
            except Exception:
                pass

            # 3. Discover Prompts
            try:
                prompts_resp = await session.list_prompts()
                for p in getattr(prompts_resp, "prompts", []):
                    self.prompt_to_session[p.name] = session
                    self.available_prompts.append({
                        "name": p.name,
                        "description": p.description or "",
                        "server": server_name,
                    })
            except Exception:
                pass

        except Exception as e:
            logger.error(f"❌ Failed to connect to MCP server '{server_name}': {e}")
            # Graceful degradation per FR-K-001

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Calls a tool by name on the corresponding MCP server session."""
        session = self.tool_to_session.get(tool_name)
        if not session:
            raise ValueError(f"Tool '{tool_name}' is not registered with any active MCP server.")

        logger.info(f"[mcp: client] Invoking tool '{tool_name}' with args {arguments}")
        result = await session.call_tool(tool_name, arguments=arguments)

        if hasattr(result, "content") and result.content:
            text_contents = [c.text for c in result.content if hasattr(c, "text") and c.text]
            if text_contents:
                return "\n".join(text_contents)
            return str(result.content)

        return str(result)

    async def read_resource(self, uri: str) -> str:
        """Reads a resource by URI across connected MCP sessions."""
        session = self.resource_to_session.get(uri)

        # Fallback search across all active sessions if uri has templated parameters
        if not session:
            for s in self.sessions.values():
                try:
                    result = await s.read_resource(uri)
                    if hasattr(result, "contents") and result.contents:
                        text_blocks = [
                            c.text if hasattr(c, "text") else str(c)
                            for c in result.contents
                        ]
                        return "\n".join(text_blocks)
                except Exception:
                    continue
            raise ValueError(f"Resource '{uri}' not found on any active MCP server.")

        logger.info(f"[mcp: client] Reading resource '{uri}'")
        result = await session.read_resource(uri)
        if hasattr(result, "contents") and result.contents:
            text_blocks = [
                c.text if hasattr(c, "text") else str(c)
                for c in result.contents
            ]
            return "\n".join(text_blocks)
        return str(result)

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Renders an MCP prompt template with arguments across connected sessions."""
        # Normalize prompt argument values to strings per MCP protocol spec (dict[str, str])
        normalized_args: Dict[str, str] = {}
        if arguments:
            for k, v in arguments.items():
                normalized_args[str(k)] = str(v)

        session = self.prompt_to_session.get(prompt_name)
        if not session:
            for s in self.sessions.values():
                try:
                    result = await s.get_prompt(prompt_name, arguments=normalized_args)
                    if hasattr(result, "messages") and result.messages:
                        text_parts = []
                        for m in result.messages:
                            content = getattr(m, "content", None)
                            if hasattr(content, "text"):
                                text_parts.append(content.text)
                            else:
                                text_parts.append(str(content))
                        return "\n".join(text_parts)
                except Exception:
                    continue
            raise ValueError(f"Prompt '{prompt_name}' not found on any active MCP server.")

        logger.info(f"[mcp: client] Rendering prompt '{prompt_name}' with args {normalized_args}")
        result = await session.get_prompt(prompt_name, arguments=normalized_args)
        if hasattr(result, "messages") and result.messages:
            text_parts = []
            for m in result.messages:
                content = getattr(m, "content", None)
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                else:
                    text_parts.append(str(content))
            return "\n".join(text_parts)
        return str(result)

    async def close(self) -> None:
        """Cleanly close all transport and sessions using ExitStack."""
        logger.info("[mcp: client] Closing all MCP server connection sessions...")
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tool_to_session.clear()
        self.resource_to_session.clear()
        self.prompt_to_session.clear()
        self.available_tools.clear()
        self.available_resources.clear()
        self.available_resource_templates.clear()
        self.available_prompts.clear()
        logger.info("🛑 All MCP connections closed successfully.")
