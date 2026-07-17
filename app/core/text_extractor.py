import logging
from io import BytesIO
import docx
from pypdf import PdfReader
import google.generativeai as genai
import anyio
from app.core.env import settings

logger = logging.getLogger("uvicorn.error")

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Synchronous helper to extract text from a DOCX file."""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text.append(cell.text)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error reading DOCX file: {e}")
        raise e

def extract_text_from_pdf_digital(file_bytes: bytes) -> str:
    """Synchronous helper to extract text from a digital PDF."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text.append(extracted)
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error reading PDF digitally: {e}")
        raise e

async def extract_text_via_gemini(file_bytes: bytes, mime_type: str) -> str:
    """Call Gemini to extract text from scanned PDFs or images."""
    try:
        import os
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in settings or environment variables.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            "You are a document transcription system. Extract and transcribe all the textual content "
            "from this document. Present it clearly, preserving structural flow where possible. "
            "Do not summarize or add commentary. Output only the extracted text."
        )
        
        def _generate():
            response = model.generate_content([
                {"mime_type": mime_type, "data": file_bytes},
                prompt
            ])
            return response.text
            
        return await anyio.to_thread.run_sync(_generate)
    except Exception as e:
        logger.error(f"Error performing OCR text extraction via Gemini: {e}")
        raise e

async def extract_text(file_bytes: bytes, content_type: str) -> str:
    """
    Core entry point to extract text based on mime-type.
    Uses native libraries for digital files and falls back to Gemini for OCR/scanned files.
    """
    c_type = content_type.lower()
    
    # DOCX
    if "wordprocessingml.document" in c_type or c_type == "docx" or c_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        logger.info("Extracting text from DOCX document")
        return await anyio.to_thread.run_sync(extract_text_from_docx, file_bytes)
        
    # PDF
    elif "pdf" in c_type or c_type == "application/pdf":
        logger.info("Extracting text from PDF document")
        # Try digital extraction first
        digital_text = await anyio.to_thread.run_sync(extract_text_from_pdf_digital, file_bytes)
        
        # If we got reasonable text, return it
        if len(digital_text.strip()) > 100:
            logger.info("Successfully extracted text digitally from PDF")
            return digital_text
        else:
            logger.info("Digital PDF extraction returned minimal text. Falling back to Gemini Vision OCR")
            return await extract_text_via_gemini(file_bytes, "application/pdf")
            
    # Images
    elif "image" in c_type or c_type in ["png", "jpg", "jpeg", "webp"]:
        logger.info(f"Extracting text from image ({content_type}) via Gemini Vision")
        # Treat image types properly or fallback to general image/png
        mime_type = content_type if "image" in c_type else f"image/{content_type.replace('jpg', 'jpeg')}"
        return await extract_text_via_gemini(file_bytes, mime_type)
        
    # Text fallback
    else:
        logger.info(f"Treating content type {content_type} as plain text")
        try:
            return file_bytes.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to decode text file: {e}")
            raise ValueError(f"Unsupported content type for text extraction: {content_type}")
