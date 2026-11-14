import logging
from langgraph.graph import StateGraph, START, END
from app.core.env import settings
from app.core.progress import publish_progress
from app.core.database import prisma, ensure_db_connected, get_checkpointer

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
    """Executes the Knowledge Builder workflow with persistent Postgres checkpointing and robust idempotency."""
    doc_id = state.document_id
    proj_id = state.project_id
    thread_id = f"{doc_id}-kb" if doc_id else f"{proj_id}-kb"
    logger.info(f"Running Knowledge Builder checkpointed workflow for thread {thread_id}")

    await ensure_db_connected()

    # 1. State & Database Idempotency Check: Check if completion marker exists in DB
    if doc_id:
        existing_marker = await prisma.aisuggestion.find_first(
            where={
                "projectId": proj_id,
                "type": "doc_kb_indexed",
            }
        )
        if existing_marker and isinstance(existing_marker.content, dict) and existing_marker.content.get("documentId") == doc_id:
            logger.info(f"[Knowledge Builder] Document {doc_id} already marked as indexed in DB. Skipping knowledge base re-construction.")
            await publish_progress(doc_id, "Knowledge base construction already completed. Reusing existing vector and graph indices.", "knowledge_builder")
            return

    # 2. Graph Store Idempotency Check: Check if Neo4j already contains chunks for this document
    if doc_id:
        try:
            from app.infrastructure.graph_store.neo4j import neo4j_graph_store
            check_query = "MATCH (c:Chunk {document_id: $doc_id}) RETURN count(c) as cnt"
            records = await neo4j_graph_store.execute_query(check_query, {"doc_id": doc_id})
            if records and records[0].get("cnt", 0) > 0:
                cnt = records[0]["cnt"]
                logger.info(f"[Knowledge Builder] Found {cnt} existing indexed chunks in Neo4j for doc {doc_id}. Skipping re-indexing.")
                await publish_progress(
                    doc_id,
                    f"Knowledge base already built ({cnt} vector chunks present in Neo4j). Reusing existing indices.",
                    "knowledge_builder"
                )
                return
        except Exception as n4j_err:
            logger.debug(f"Neo4j idempotency check error (non-fatal): {n4j_err}")

    # 3. Use pooled checkpointer to avoid idle socket drops during long LLM calls
    checkpointer = await get_checkpointer()
    graph = workflow.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    
    # Verify if checkpoint exists to resume
    kb_state = await graph.aget_state(config)
    if kb_state and kb_state.next:
        next_node = kb_state.next[0] if isinstance(kb_state.next, (list, tuple)) else kb_state.next
        logger.info(f"Resuming Knowledge Builder for {thread_id} from node: {next_node}")
        if doc_id:
            await publish_progress(doc_id, f"Resuming knowledge base construction from stage '{next_node}'...", "knowledge_builder")
        await graph.ainvoke(None, config=config)
    elif kb_state and not kb_state.next and kb_state.values:
        status_val = kb_state.values.get("status")
        has_errors = bool(kb_state.values.get("errors"))
        if status_val == "completed" and not has_errors:
            logger.info(f"Knowledge Builder already completed for {thread_id}. Skipping.")
            if doc_id:
                await publish_progress(doc_id, "Knowledge base construction already completed.", "knowledge_builder")
        else:
            logger.warning(f"Previous Knowledge Builder run for {thread_id} finished in status='{status_val}'. Re-executing...")
            if doc_id:
                await publish_progress(doc_id, "Re-running knowledge base construction...", "knowledge_builder")
            await graph.ainvoke(state.model_dump(), config=config)
    else:
        logger.info(f"Starting fresh Knowledge Builder execution for {thread_id}")
        if doc_id:
            await publish_progress(doc_id, "Initializing knowledge base construction (vectors and knowledge graph)...", "knowledge_builder")
        await graph.ainvoke(state.model_dump(), config=config)

    # Check final execution result
    final_state = await graph.aget_state(config)
    if final_state and final_state.values:
        if final_state.values.get("errors"):
            errors = final_state.values["errors"]
            logger.error(f"Knowledge builder execution finished with errors: {errors}")
            raise RuntimeError(f"Knowledge builder failed: {'; '.join(errors)}")
        if final_state.values.get("status") == "failed":
            raise RuntimeError("Knowledge builder execution finished in failed status.")
