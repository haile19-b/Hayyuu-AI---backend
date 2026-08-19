import asyncio
import logging
import sys
import os

# Set Windows Selector Event Loop Policy for database / subprocess compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_test")

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from app.infrastructure.mcp.client_manager import MCPClientManager

async def main():
    logger.info("Initializing MCP Integration Test...")
    
    # Instantiate client manager
    manager = MCPClientManager()
    
    try:
        # Start connection to servers
        await manager.connect_all()
        
        logger.info("\n=== Discovering Available Tools ===")
        tools = manager.available_tools
        logger.info(f"Total Discovered Tools: {len(tools)}")
        
        for tool in tools:
            logger.info(f" - Tool Name: {tool['name']}")
            logger.info(f"   Description: {tool['description'][:80]}...")
            logger.info(f"   Schema: {tool['input_schema']}")
            logger.info("-" * 40)
            
        # Verify specific tools are present
        discovered_names = {t["name"] for t in tools}
        required_tools = [
            "list_project_documents", 
            "list_project_requirements", 
            "list_project_tasks",
            "list_project_conflicts",
            "list_project_suggestions",
            "get_project_graph_summary",
            "traverse_project_subgraph",
            "create_github_issue"
        ]
        
        missing_tools = [t for t in required_tools if t not in discovered_names]
        if missing_tools:
            logger.error(f"❌ Missing expected tools: {missing_tools}")
        else:
            logger.info("✅ All core database, graph, and GitHub tools successfully discovered!")

        # Execute a dummy test call
        logger.info("\n=== Testing Live Tool Execution ===")
        mock_project_id = "00000000-0000-0000-0000-000000000000"
        
        try:
            logger.info(f"Calling 'list_project_documents' for mock project ID '{mock_project_id}'...")
            result = await manager.call_tool("list_project_documents", {"project_id": mock_project_id})
            logger.info(f"Tool Result Output:\n{result}")
            logger.info("✅ Live tool execution completed successfully!")
        except Exception as exec_err:
            logger.error(f"❌ Tool execution failed: {exec_err}")
            
    except Exception as err:
        logger.error(f"❌ Setup error: {err}")
    finally:
        # Clean up connection stacking
        await manager.close()
        logger.info("Integration Test Finished.")

if __name__ == "__main__":
    asyncio.run(main())
