from typing import TypedDict, Optional, List

class DocumentAnalysisState(TypedDict):
    project_id: str
    document_id: str
    extracted_text: Optional[str]
    gemini_file_uri: Optional[str]
    gemini_file_mime_type: Optional[str]
    extracted_requirement_ids: Optional[List[str]]
