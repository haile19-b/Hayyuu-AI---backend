import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field
from prisma.enums import (
    ProjectStatus,
    DocumentStatus,
    RequirementStatus,
    RequirementPriority,
    RequirementType,
    TaskStatus,
    TaskPriority,
    TaskSource,
    TaskApprovalStatus,
    SuggestionStatus,
    ConflictSeverity,
    ConflictStatus
)
from app.domain.entities.auth.auth_schema import ApiResponse

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None

class UpdateProjectRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    status: Optional[ProjectStatus] = None

class ProjectResponse(BaseModel):
    id: str
    userId: str
    name: str
    description: Optional[str] = None
    status: ProjectStatus
    githubInstallationId: Optional[str] = None
    githubRepoOwner: Optional[str] = None
    githubRepoName: Optional[str] = None
    githubRepoUrl: Optional[str] = None
    githubConnectedAt: Optional[datetime.datetime] = None
    createdAt: datetime.datetime
    updatedAt: datetime.datetime
    documentsCount: int = 0
    requirementsCount: int = 0
    tasksCount: int = 0
    conflictsCount: int = 0

    class Config:
        from_attributes = True

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

class RequirementResponse(BaseModel):
    id: str
    projectId: str
    title: str
    description: str
    type: RequirementType
    priority: RequirementPriority
    status: RequirementStatus
    isConflicted: bool
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class TaskResponse(BaseModel):
    id: str
    projectId: str
    requirementId: Optional[str] = None
    title: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    source: TaskSource
    approvalStatus: TaskApprovalStatus
    githubIssueUrl: Optional[str] = None
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class AISuggestionResponse(BaseModel):
    id: str
    projectId: str
    type: str
    content: Any  # Handles dynamic JSON payload
    status: SuggestionStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class CreateRequirementRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    type: RequirementType = RequirementType.FUNCTIONAL
    priority: RequirementPriority = RequirementPriority.P2

class UpdateRequirementRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    type: Optional[RequirementType] = None
    priority: Optional[RequirementPriority] = None
    status: Optional[RequirementStatus] = None
    isConflicted: Optional[bool] = None

class CreateTaskRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str
    requirementId: Optional[str] = None
    priority: TaskPriority = TaskPriority.P2
    status: TaskStatus = TaskStatus.TODO

class UpdateTaskRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    requirementId: Optional[str] = None
    priority: Optional[TaskPriority] = None
    status: Optional[TaskStatus] = None

class ConflictResponse(BaseModel):
    id: str
    projectId: str
    requirementId: str
    conflictingRequirementId: str
    severity: ConflictSeverity
    description: str
    aiRecommendation: Optional[str] = None
    status: ConflictStatus
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class ResolveConflictRequest(BaseModel):
    status: ConflictStatus

# GitHub Integration Schemas
class GitHubInstallUrlResponse(BaseModel):
    installUrl: str
    state: str

class GitHubConnectRequest(BaseModel):
    installationId: str
    state: Optional[str] = None
    selectedRepos: Optional[list[str]] = None

class GitHubSelectRepoRequest(BaseModel):
    owner: str
    repo: str

class GitHubRepoAssignmentItem(BaseModel):
    owner: str
    repo: str

class GitHubAssignReposRequest(BaseModel):
    repositories: list[GitHubRepoAssignmentItem]
    activeRepo: Optional[str] = None

class ProjectRepositoryResponse(BaseModel):
    id: str
    projectId: str
    owner: str
    name: str
    fullName: str
    htmlUrl: str
    defaultBranch: str = "main"
    isActive: bool = False
    createdAt: datetime.datetime
    updatedAt: datetime.datetime

    class Config:
        from_attributes = True

class GitHubRepositoryItem(BaseModel):
    id: int
    name: str
    fullName: str
    owner: str
    description: Optional[str] = None
    htmlUrl: str
    defaultBranch: str = "main"
    private: bool = False
    updatedAt: Optional[str] = None
    isAssignedToThisProject: bool = False
    isClaimedByOtherProject: bool = False
    claimedByProjectName: Optional[str] = None

class GitHubStatusResponse(BaseModel):
    isConnected: bool
    installationId: Optional[str] = None
    repoOwner: Optional[str] = None
    repoName: Optional[str] = None
    repoUrl: Optional[str] = None
    connectedAt: Optional[datetime.datetime] = None
    accessibleRepositories: list[GitHubRepositoryItem] = []
    assignedRepositories: list[ProjectRepositoryResponse] = []
    activeRepository: Optional[ProjectRepositoryResponse] = None
    suggestedInstallationId: Optional[str] = None

class GitHubBranchItem(BaseModel):
    name: str
    commitSha: str
    protected: bool = False
    isDefault: bool = False

class GitHubCommitItem(BaseModel):
    hash: str
    fullHash: str
    message: str
    author: str
    authorAvatarUrl: Optional[str] = None
    date: Optional[str] = None
    htmlUrl: Optional[str] = None

class GitHubPullRequestItem(BaseModel):
    number: int
    title: str
    status: str
    state: str
    author: str
    authorAvatarUrl: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    mergedAt: Optional[str] = None
    closedAt: Optional[str] = None
    htmlUrl: Optional[str] = None
    draft: bool = False
    sourceBranch: Optional[str] = None
    targetBranch: Optional[str] = None

class GitHubIssueItem(BaseModel):
    number: int
    title: str
    state: str
    author: str
    labels: list[str] = []
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    closedAt: Optional[str] = None
    htmlUrl: Optional[str] = None
    commentsCount: int = 0

class GitHubLiveDataResponse(BaseModel):
    repoOwner: str
    repoName: str
    repoUrl: str
    activeBranch: str
    branches: list[GitHubBranchItem] = []
    commits: list[GitHubCommitItem] = []
    pullRequests: list[GitHubPullRequestItem] = []
    issues: list[GitHubIssueItem] = []
    openIssuesCount: int = 0
    openPRsCount: int = 0
    mergedPRsCount: int = 0
    syncedAt: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

