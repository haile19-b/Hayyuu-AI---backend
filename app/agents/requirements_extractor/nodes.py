import logging
import json
import os
import anyio
from io import BytesIO
from pydantic import BaseModel, Field
from google import genai

from app.agents.requirements_extractor.schema import ExtractionResponse, SuggestionsResponse
from app.core.database import prisma
from app.core.storage import storage_utility
from app.core.progress import publish_progress
from prisma.enums import (
    RequirementType, RequirementPriority, RequirementStatus,
    TaskSource, TaskApprovalStatus, TaskStatus, TaskPriority,
    ConflictSeverity, ConflictStatus, SuggestionStatus
)
from app.core.env import settings
from app.agents.requirements_extractor.state import DocumentAnalysisState
from app.core.config import genAI

logger = logging.getLogger("uvicorn.error")

# Node 1: Ingest and Upload Original Document to Gemini Files API
async def ingest_document_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: ingest_document] Uploading document directly to Gemini Files API for {doc_id}")
    await publish_progress(doc_id, "Preparing document ingestion...", "ingest_document")
    
    # 1. Fetch document metadata
    document = await prisma.document.find_unique(where={"id": doc_id})
    if not document:
        raise ValueError(f"Document with ID {doc_id} not found in database")
        
    # 2. Download raw file content
    await publish_progress(doc_id, f"Downloading file '{document.name}' from storage...", "ingest_document")
    file_bytes = await storage_utility.download_file(document.filePath)
    
    # 3. Upload file to Gemini Files API for native multi-page document understanding
    await publish_progress(doc_id, "Uploading document to Gemini Files API...", "ingest_document")
    def _upload():
        return genAI.files.upload(
            file=BytesIO(file_bytes),
            config=dict(mime_type=document.fileType)
        )
    gemini_file = await anyio.to_thread.run_sync(_upload)
    logger.info(f"Successfully uploaded document to Gemini. URI: {gemini_file.uri}")
    await publish_progress(doc_id, "Document successfully uploaded to Gemini.", "ingest_document")
    
    return {
        "gemini_file_uri": gemini_file.uri,
        "gemini_file_mime_type": gemini_file.mime_type
    }

# Node 2: Extract Requirements + Tasks + Conflicts (Joint Gemini Processing)
async def extract_requirements_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    proj_id = state["project_id"]
    file_uri = state.get("gemini_file_uri")
    file_mime_type = state.get("gemini_file_mime_type")
    
    logger.info(f"[Node: extract_requirements] Performing joint extraction of requirements, tasks, and conflicts for {doc_id}")
    
    if not file_uri:
        raise ValueError("Gemini file URI is missing. Ingestion node must run first.")
        
    # 1. Retrieve existing project requirements from DB to check for conflicts
    await publish_progress(doc_id, "Checking existing project requirements for conflicts...", "extract_requirements")
    existing_reqs = await prisma.requirement.find_many(where={"projectId": proj_id})
    existing_reqs_text = ""
    if existing_reqs:
        existing_reqs_text = "Here is the list of existing requirements in the database for this project:\n"
        for r in existing_reqs:
            existing_reqs_text += f"Database UUID: {r.id}\nTitle: {r.title}\nDescription: {r.description}\n---\n"
    else:
        existing_reqs_text = "There are no existing requirements in the database for this project."
        
    prompt = (
        "You are an expert systems analyst. Analyze the provided document and perform four tasks:\n"
        "1. Extract all functional and non-functional software requirements mentioned in it. Assign each a unique temporary ID (e.g., 'new_req_1', 'new_req_2').\n"
        "2. For each requirement, automatically generate associated specific developer tasks needed to implement that requirement.\n"
        "3. Compare the document against the list of existing requirements in the database (provided below) "
        "and identify any conflicts, contradictions, or duplicate requirements between the document and the existing requirements. "
        "For these conflicts, classify them as 'EXISTING_VS_NEW' and reference the database UUID of the existing requirement.\n"
        "4. Detect contradictions or inconsistencies among the newly extracted requirements within the uploaded document itself. "
        "For these conflicts, classify them as 'NEW_VS_NEW' and reference the temporary IDs of both requirements in conflict.\n\n"
        f"{existing_reqs_text}\n\n"
        "Return a structured JSON response matching the extraction schema."
    )
    
    await publish_progress(doc_id, "Extracting requirements, tasks, and conflicts using Gemini AI...", "extract_requirements")
    
    # Run API request in thread pool using the Interactions API
    def _generate():
        response = genAI.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "document", "uri": file_uri, "mime_type": file_mime_type},
                {"type": "text", "text": prompt}
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ExtractionResponse.model_json_schema()
            }
        )
        return response.output_text
        
    response_text = await anyio.to_thread.run_sync(_generate)
    
    try:
        data = json.loads(response_text)
        reqs = data.get("requirements", [])
        detected_conflicts = data.get("conflicts", [])
    except Exception as e:
        logger.error(f"Failed to parse JSON response from Gemini: {e}. Raw response: {response_text}")
        raise ValueError(f"Gemini joint extraction returned invalid JSON: {e}")
        
    # Save Requirements & Tasks
    await publish_progress(doc_id, f"Saving {len(reqs)} draft requirements and tasks to the database...", "extract_requirements")
    
    temp_id_to_uuid = {}
    req_created = 0
    task_created = 0
    req_created_ids = []
    
    for req in reqs:
        # Map requirement type
        req_type = RequirementType.FUNCTIONAL
        if req.get("type") == "NON_FUNCTIONAL":
            req_type = RequirementType.NON_FUNCTIONAL
            
        # Map requirement priority
        priority_str = req.get("priority", "P2").upper()
        req_priority = RequirementPriority.P2
        if priority_str in ["P0", "P1", "P2"]:
            req_priority = RequirementPriority[priority_str]
            
        # Create requirement (as SUGGESTION/draft status)
        db_req = await prisma.requirement.create(
            data={
                "projectId": proj_id,
                "title": req["title"],
                "description": req["description"],
                "type": req_type,
                "priority": req_priority,
                "status": RequirementStatus.SUGGESTION
            }
        )
        temp_id_to_uuid[req["temp_id"]] = db_req.id
        req_created_ids.append(db_req.id)
        req_created += 1
        
        # Create associated developer tasks (as PENDING approval status)
        for t in req.get("tasks", []):
            t_priority_str = t.get("priority", "P2").upper()
            t_priority = TaskPriority.P2
            if t_priority_str in ["P0", "P1", "P2"]:
                t_priority = TaskPriority[t_priority_str]
                
            await prisma.task.create(
                data={
                    "projectId": proj_id,
                    "requirementId": db_req.id,
                    "title": t["title"],
                    "description": t["description"],
                    "priority": t_priority,
                    "status": TaskStatus.TODO,
                    "source": TaskSource.SYSTEM,
                    "approvalStatus": TaskApprovalStatus.PENDING
                }
            )
            task_created += 1
            
    # Save Conflicts
    conflicts_created = 0
    if detected_conflicts:
        await publish_progress(doc_id, f"Saving {len(detected_conflicts)} flagged requirement conflicts...", "extract_requirements")
        for conflict in detected_conflicts:
            conflict_type = conflict.get("conflict_type", "EXISTING_VS_NEW").upper()
            
            # Resolve new requirement A UUID
            req_a_id = temp_id_to_uuid.get(conflict.get("requirement_temp_id_a"))
            if not req_a_id:
                continue
                
            req_b_id = None
            description_prefix = ""
            
            # Resolve requirement B UUID depending on conflict type
            if conflict_type == "NEW_VS_NEW":
                req_b_id = temp_id_to_uuid.get(conflict.get("requirement_temp_id_b"))
                description_prefix = "[New vs New] "
            else:
                req_b_id = conflict.get("existing_requirement_id")
                description_prefix = "[Existing vs New] "
                
            if not req_b_id:
                continue
                
            # Map conflict severity
            sev_str = conflict.get("severity", "MEDIUM").upper()
            severity = ConflictSeverity.MEDIUM
            if sev_str in ["HIGH", "MEDIUM", "LOW"]:
                severity = ConflictSeverity[sev_str]
                
            # Avoid redundant conflict rows
            existing_link = await prisma.requirementconflict.find_first(
                where={
                    "OR": [
                        {"requirementId": req_a_id, "conflictingRequirementId": req_b_id},
                        {"requirementId": req_b_id, "conflictingRequirementId": req_a_id}
                    ]
                }
            )
            if not existing_link:
                await prisma.requirementconflict.create(
                    data={
                        "projectId": proj_id,
                        "requirementId": req_a_id,
                        "conflictingRequirementId": req_b_id,
                        "severity": severity,
                        "description": f"{description_prefix}{conflict['description']}",
                        "aiRecommendation": conflict.get("recommendation"),
                        "status": ConflictStatus.ACTIVE
                    }
                )
                
                # Flag both requirements as conflicted
                await prisma.requirement.update(where={"id": req_a_id}, data={"isConflicted": True})
                await prisma.requirement.update(where={"id": req_b_id}, data={"isConflicted": True})
                conflicts_created += 1
                
    await publish_progress(
        document_id=doc_id,
        message=f"Joint processing complete. Saved {req_created} requirements, {task_created} tasks, and flagged {conflicts_created} conflicts.",
        step="extract_requirements"
    )
    logger.info(f"Saved {req_created} requirements, {task_created} tasks, and {conflicts_created} conflicts in DB")
    
    return {
        "extracted_requirement_ids": req_created_ids
    }

# Node 3: Generate suggestions (Gap Analysis Suggestions)
async def generate_suggestions_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    proj_id = state["project_id"]
    file_uri = state.get("gemini_file_uri")
    file_mime_type = state.get("gemini_file_mime_type")
    
    logger.info(f"[Node: generate_suggestions] Extracting product gap suggestions for document {doc_id}")
    await publish_progress(doc_id, "Analyzing document for product gaps and improvements...", "generate_suggestions")
    
    if not file_uri:
        raise ValueError("Gemini file URI is missing. Cannot perform native document understanding.")
        
    prompt = (
        "You are an experienced product manager. Analyze the provided project document "
        "and identify any product gaps, missing requirements, or improvements that should be made "
        "to ensure project success.\n\n"
        "Generate a list of actionable suggestions. For each suggestion, provide a title, description, "
        "reasoning (why it is a gap/improvement), and the suggested category (e.g., 'security', 'usability', 'performance')."
    )
    
    def _generate():
        response = genAI.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "document", "uri": file_uri, "mime_type": file_mime_type},
                {"type": "text", "text": prompt}
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": SuggestionsResponse.model_json_schema()
            }
        )
        return response.output_text
        
    response_text = await anyio.to_thread.run_sync(_generate)
    
    try:
        data = json.loads(response_text)
        suggestions = data.get("suggestions", [])
    except Exception as e:
        logger.error(f"Failed to parse suggestions JSON from Gemini: {e}")
        raise ValueError(f"Gemini suggestions extraction returned invalid JSON: {e}")
        
    if not suggestions:
        await publish_progress(doc_id, "No suggestions or product gaps detected.", "generate_suggestions")
        return {}
        
    await publish_progress(doc_id, f"Saving {len(suggestions)} AI suggestions to the database...", "generate_suggestions")
    
    created_count = 0
    for sug in suggestions:
        await prisma.aisuggestion.create(
            data={
                "projectId": proj_id,
                "type": "gap_analysis",
                "content": Json({
                    "title": sug["title"],
                    "description": sug["description"],
                    "reasoning": sug["reasoning"],
                    "category": sug["category"]
                }),
                "status": SuggestionStatus.PENDING
            }
        )
        created_count += 1
        
    await publish_progress(doc_id, f"Gap analysis complete. Saved {created_count} gap improvement suggestions.", "generate_suggestions")
    logger.info(f"Saved {created_count} AISuggestions to the database.")
    return {}
