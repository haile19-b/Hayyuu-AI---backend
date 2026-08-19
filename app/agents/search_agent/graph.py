import logging
import uuid
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.env import settings

from app.agents.search_agent.state import SearchAgentState
from app.agents.search_agent.nodes import (
    query_analysis_node,
    graph_retrieval_node,
    agent_loop_node,
    execute_db_tool_node,
    validate_response_node,
)

logger = logging.getLogger("uvicorn.error")


def router_conditional_edge(state: SearchAgentState) -> str:
    """Decides if the model wants to call database tools or proceed to validation/end."""
    status = state.status
    if status == "tool_call":
        return "execute_db_tool"
    elif status == "synthesized":
        return "validate"
    return END


# 1. Initialize StateGraph
workflow = StateGraph(SearchAgentState)

# 2. Add workflow nodes
workflow.add_node("query_analysis", query_analysis_node)
workflow.add_node("graph_retrieval", graph_retrieval_node)
workflow.add_node("agent_loop", agent_loop_node)
workflow.add_node("execute_db_tool", execute_db_tool_node)
workflow.add_node("validate", validate_response_node)

# 3. Set Entry Point
workflow.set_entry_point("query_analysis")

# 4. Connect static edges
workflow.add_edge("query_analysis", "graph_retrieval")
workflow.add_edge("graph_retrieval", "agent_loop")
workflow.add_edge("execute_db_tool", "agent_loop")
workflow.add_edge("validate", END)

# 5. Connect conditional routing edge
workflow.add_conditional_edges(
    "agent_loop",
    router_conditional_edge,
    {
        "execute_db_tool": "execute_db_tool",
        "validate": "validate",
        "__end__": END
    }
)

# 6. Compile the Search Agent Graph statically (fallback)
search_agent_graph = workflow.compile()


async def run_search_agent(state: SearchAgentState, mcp_manager: Any = None) -> Dict[str, Any]:
    """Runs the Search Agent with persistent Postgres state checkpointing."""
    run_id = str(uuid.uuid4())
    thread_id = f"{state.project_id}-search-{run_id}"
    logger.info(f"Running Search Agent checkpointed workflow for thread {thread_id}")
    
    # Initialize connection to PostgreSQL for state checkpoints
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        
        # Compile graph with checkpointing
        graph = workflow.compile(checkpointer=checkpointer)
        config = {
            "configurable": {
                "thread_id": thread_id,
                "mcp_manager": mcp_manager
            }
        }
        
        # Start execution
        result_state = await graph.ainvoke(state.model_dump(), config=config)
        return result_state
