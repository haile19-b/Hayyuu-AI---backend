import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.env import settings
from app.core.database import prisma
from app.core.progress import publish_progress
from prisma.enums import DocumentStatus

from app.agents.requirements_extractor.state import DocumentAnalysisState
from app.agents.requirements_extractor.nodes import (
    extract_text_node,
    extract_requirements_node,
    chunk_and_embed_node,
    detect_conflicts_node,
    create_tasks_node,
    generate_suggestions_node,
)

logger = logging.getLogger("uvicorn.error")

# Build Graph
builder = StateGraph(DocumentAnalysisState)
builder.add_node("extract_text", extract_text_node)
builder.add_node("extract_requirements", extract_requirements_node)
builder.add_node("chunk_and_embed", chunk_and_embed_node)
builder.add_node("detect_conflicts", detect_conflicts_node)
builder.add_node("create_tasks", create_tasks_node)
builder.add_node("generate_suggestions", generate_suggestions_node)

builder.add_edge(START, "extract_text")
builder.add_edge("extract_text", "extract_requirements")
builder.add_edge("extract_requirements", "chunk_and_embed")
builder.add_edge("chunk_and_embed", "detect_conflicts")
builder.add_edge("detect_conflicts", "create_tasks")
builder.add_edge("create_tasks", "generate_suggestions")
builder.add_edge("generate_suggestions", END)

async def run_workflow(document_id: str, project_id: str) -> None:
    """Run the compiled LangGraph document analysis pipeline, enabling automatic resume."""
    logger.info(f"Running LangGraph workflow for document {document_id}")
    
    # Initialize connection to PostgreSQL for state checkpoints
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        # Create checkpoint tables if they do not exist
        await checkpointer.setup()
        
        # Compile graph with checkpointing
        graph = builder.compile(checkpointer=checkpointer)
        
        # Use document ID as thread ID to guarantee identical thread recovery
        config = {"configurable": {"thread_id": document_id}}
        
        # Check if there is an existing state checkpoint to resume from
        state = await graph.aget_state(config)
        if state and state.next:
            logger.info(f"Resuming document analysis workflow for {document_id} from node: {state.next}")
            await graph.ainvoke(None, config=config)
        else:
            logger.info(f"Starting new document analysis workflow execution for {document_id}")
            initial_state = {
                "project_id": project_id,
                "document_id": document_id,
                "extracted_text": ""
            }
            await graph.ainvoke(initial_state, config=config)
            
        # Update document status in the database to INDEXED upon successful completion
        await prisma.document.update(
            where={"id": document_id},
            data={"status": DocumentStatus.INDEXED}
        )
        
        # Publish final progress completion status
        await publish_progress(
            document_id=document_id,
            message="Document analysis pipeline completed successfully!",
            step="finished",
            status="COMPLETED"
        )

