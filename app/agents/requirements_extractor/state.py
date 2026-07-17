from typing import TypedDict, Optional

class DocumentAnalysisState(TypedDict):
    project_id: str
    document_id: str
    extracted_text: Optional[str]
