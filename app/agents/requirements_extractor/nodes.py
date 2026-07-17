import logging
from typing import List
import json
import os
import anyio
from pydantic import BaseModel, Field
from google import genai
from google.genai.types import GenerateContentConfig

from app.core.database import prisma
from app.core.storage import storage_utility
from app.core.text_extractor import extract_text
from app.core.progress import publish_progress
from prisma.enums import RequirementType, RequirementPriority, RequirementStatus
from app.core.env import settings
from app.agents.requirements_extractor.state import DocumentAnalysisState

logger = logging.getLogger("uvicorn.error")

# Structured Output Pydantic schemas for Gemini
class ExtractedRequirement(BaseModel):
    title: str = Field(description="Short, descriptive title of the requirement")
    description: str = Field(description="Detailed explanation of the requirement")
    type: str = Field(description="Must be either 'FUNCTIONAL' or 'NON_FUNCTIONAL'")
    priority: str = Field(description="Priority level: 'P0', 'P1', or 'P2'")

class ExtractionResponse(BaseModel):
    requirements: List[ExtractedRequirement]

# Node 1: Extract Text
async def extract_text_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: extract_text] Extracting text for document {doc_id}")
    await publish_progress(doc_id, "Starting text extraction...", "extract_text")
    
    # 1. Fetch document metadata
    document = await prisma.document.find_unique(where={"id": doc_id})
    if not document:
        raise ValueError(f"Document with ID {doc_id} not found in database")
        
    # 2. Download raw file content
    await publish_progress(doc_id, f"Downloading file '{document.name}' from storage...", "extract_text")
    file_bytes = await storage_utility.download_file(document.filePath)
    
    # 3. Extract text
    await publish_progress(doc_id, f"Extracting text from content type '{document.fileType}'...", "extract_text")
    text = await extract_text(file_bytes, document.fileType)
    
    if not text.strip():
        raise ValueError("No text could be extracted from the document file")
        
    logger.info(f"Successfully extracted {len(text)} characters of text")
    await publish_progress(doc_id, f"Text extraction complete. Extracted {len(text)} characters.", "extract_text")
    
    return {"extracted_text": text}

# Node 2: Extract Requirements
async def extract_requirements_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    proj_id = state["project_id"]
    text = state["extracted_text"]
    
    logger.info(f"[Node: extract_requirements] Extracting requirements for document {doc_id}")
    await publish_progress(doc_id, "Extracting requirements using Gemini AI...", "extract_requirements")
    
    if not text:
        raise ValueError("Extracted text is empty. Cannot extract requirements.")
        
    # Set up Gemini
    api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured in settings or environment variables.")
    
    client = genai.Client(api_key=api_key)
    
    prompt = (
        "You are an expert systems analyst. Analyze the following document text and extract all functional "
        "and non-functional software requirements mentioned in it. Classify them carefully, select their priority "
        "level, and compile them into the structured schema format.\n\n"
        f"Document Text:\n{text}"
    )
    
    # Run API request in thread pool
    def _generate():
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResponse
            )
        )
        return response.text
        
    response_text = await anyio.to_thread.run_sync(_generate)
    
    try:
        data = json.loads(response_text)
        reqs = data.get("requirements", [])
    except Exception as e:
        logger.error(f"Failed to parse JSON response from Gemini: {e}. Raw response: {response_text}")
        raise ValueError(f"Gemini requirement extraction returned invalid JSON: {e}")
        
    if not reqs:
        await publish_progress(doc_id, "No requirements found in the document.", "extract_requirements")
        logger.info("No requirements extracted.")
        return {}
        
    # Save to database
    await publish_progress(doc_id, f"Saving {len(reqs)} draft requirements to the database...", "extract_requirements")
    
    created_count = 0
    for req in reqs:
        # map type
        req_type = RequirementType.FUNCTIONAL
        if req.get("type") == "NON_FUNCTIONAL":
            req_type = RequirementType.NON_FUNCTIONAL
            
        # map priority
        priority_str = req.get("priority", "P2").upper()
        req_priority = RequirementPriority.P2
        if priority_str in ["P0", "P1", "P2"]:
            req_priority = RequirementPriority[priority_str]
            
        await prisma.requirement.create(
            data={
                "projectId": proj_id,
                "title": req["title"],
                "description": req["description"],
                "type": req_type,
                "priority": req_priority,
                "status": RequirementStatus.SUGGESTION
            }
        )
        created_count += 1
        
    await publish_progress(doc_id, f"Extracted and saved {created_count} draft requirements successfully.", "extract_requirements")
    logger.info(f"Successfully saved {created_count} requirements in DB")
    return {}

# Node 3: Chunk & Embed Document
async def chunk_and_embed_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: chunk_and_embed] Chunking and embedding text for document {doc_id}")
    # Stub
    return {}

# Node 4: Detect Conflicts
async def detect_conflicts_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: detect_conflicts] Detecting conflicts for document {doc_id}")
    # Stub
    return {}

# Node 5: Create Tasks
async def create_tasks_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: create_tasks] Creating tasks for document {doc_id}")
    # Stub
    return {}

# Node 6: Generate Suggestions
async def generate_suggestions_node(state: DocumentAnalysisState) -> dict:
    doc_id = state["document_id"]
    logger.info(f"[Node: generate_suggestions] Generating suggestions for document {doc_id}")
    # Stub
    return {}
