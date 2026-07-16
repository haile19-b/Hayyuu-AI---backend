import datetime
from pydantic import BaseModel
from prisma.enums import DocumentStatus
from app.domain.entities.auth.auth_schema import ApiResponse

class DocumentResponse(BaseModel):
    id: str
    projectId: str
    name: str
    filePath: str
    fileType: str
    sizeBytes: int
    status: DocumentStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class DocumentDownloadResponse(BaseModel):
    downloadUrl: str
