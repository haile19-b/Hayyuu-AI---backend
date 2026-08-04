import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.env import settings
from app.core.database import prisma
from app.core.progress import publish_progress
from prisma.enums import DocumentStatus

from app.agents.requirements_extractor.state import DocumentAnalysisState
from app.agents.requirements_extractor.nodes import (
    ingest_document_node,
    extract_requirements_node,
    generate_suggestions_node,
)

logger = logging.getLogger("uvicorn.error")

# Build Graph
builder = StateGraph(DocumentAnalysisState)
builder.add_node("ingest_document", ingest_document_node)
builder.add_node("extract_requirements", extract_requirements_node)
builder.add_node("generate_suggestions", generate_suggestions_node)

builder.add_edge(START, "ingest_document")
builder.add_edge("ingest_document", "extract_requirements")
builder.add_edge("extract_requirements", "generate_suggestions")
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
        elif state and not state.next and state.values:
            logger.info(f"Document analysis workflow already completed for {document_id}. Skipping requirements extraction.")
        else:
            logger.info(f"Starting new document analysis workflow execution for {document_id}")
            initial_state = {
                "project_id": project_id,
                "document_id": document_id
            }
            await graph.ainvoke(initial_state, config=config)
            
        # 1. Fetch document name for filename reference
        document = await prisma.document.find_unique(where={"id": document_id})
        filename = document.name if document else "document"

        # 2. Extract the Gemini file URI if available from the requirements extractor checkpoint
        state_after = await graph.aget_state(config)
        gemini_file_uri = state_after.values.get("gemini_file_uri") if state_after else None
        gemini_file_mime_type = state_after.values.get("gemini_file_mime_type") if state_after else None

        # 3. Trigger Agent Knowledge Builder dynamically to build PGVector and Neo4j indices
        from app.agents.knowledge_builder.graph import knowledge_builder_graph
        from app.agents.knowledge_builder.state import KnowledgeBuilderState
        
        logger.info(f"Triggering Agent Knowledge Builder for document {document_id}")
        await publish_progress(document_id, "Building knowledge graph and vector indices...", "knowledge_builder")
        
        kb_state = KnowledgeBuilderState(
            project_id=project_id,
            document_id=document_id,
            filename=filename,
            gemini_file_uri=gemini_file_uri,
            gemini_file_mime_type=gemini_file_mime_type,
        )
        await knowledge_builder_graph.ainvoke(kb_state)
            
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

