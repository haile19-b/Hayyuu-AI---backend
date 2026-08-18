import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.env import settings

from app.agents.knowledge_builder.state import KnowledgeBuilderState
from app.agents.knowledge_builder.nodes import (
    preprocess_chunk_node,
    generate_embeddings_node,
    extract_graph_node,
    store_vectors_node,
    store_graph_node,
    save_results_node,
)

logger = logging.getLogger("uvicorn.error")

# Define workflow
workflow = StateGraph(KnowledgeBuilderState)

# Add all agent workflow nodes
workflow.add_node("preprocess", preprocess_chunk_node)
workflow.add_node("embed", generate_embeddings_node)
workflow.add_node("extract", extract_graph_node)
workflow.add_node("store_vectors", store_vectors_node)
workflow.add_node("store_graph", store_graph_node)
workflow.add_node("save_results", save_results_node)

# Set entry point
workflow.set_entry_point("preprocess")

# Connect nodes sequentially
workflow.add_edge("preprocess", "embed")
workflow.add_edge("embed", "extract")
workflow.add_edge("extract", "store_vectors")
workflow.add_edge("store_vectors", "store_graph")
workflow.add_edge("store_graph", "save_results")
workflow.add_edge("save_results", END)

# Compile the agent graph (for static/backward compatibility reference)
knowledge_builder_graph = workflow.compile()

async def run_knowledge_builder(state: KnowledgeBuilderState) -> None:
    """Executes the Knowledge Builder workflow with persistent Postgres checkpointing."""
    thread_id = f"{state.document_id}-kb" if state.document_id else f"{state.project_id}-kb"
    logger.info(f"Running Knowledge Builder checkpointed workflow for thread {thread_id}")
    
    # Initialize persistent state checkpointer
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        
        # Compile graph with saver checkpointer
        graph = workflow.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Verify if checkpoint exists to resume
        kb_state = await graph.aget_state(config)
        if kb_state and kb_state.next:
            logger.info(f"Resuming Knowledge Builder for {thread_id} from node: {kb_state.next}")
            await graph.ainvoke(None, config=config)
        elif kb_state and not kb_state.next and kb_state.values:
            logger.info(f"Knowledge Builder already completed for {thread_id}. Skipping.")
        else:
            logger.info(f"Starting fresh Knowledge Builder execution for {thread_id}")
            await graph.ainvoke(state.model_dump(), config=config)
