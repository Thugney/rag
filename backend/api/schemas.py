from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    api_version: str
    embedding_provider: str
    vector_store_backend: str
    llm_model: str
    project_count: int
    document_count: int
    indexed_chunk_count: int
    session_count: int


class ProjectSummary(BaseModel):
    project_id: str
    name: str
    description: str
    created_at: str
    session_count: int
    document_count: int
    indexed_chunk_count: int
    pending_document_count: int


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)


class UpdateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)


class DeleteProjectResponse(BaseModel):
    deleted_project_id: str
    deleted_session_count: int
    deleted_message_count: int
    deleted_document_count: int
    reindexed_document_count: int
    reindexed_chunk_count: int


class DeleteDocumentResponse(BaseModel):
    project_id: str
    deleted_document_id: str
    reindexed_document_count: int
    reindexed_chunk_count: int


class DocumentRecord(BaseModel):
    id: str
    project_id: str
    original_filename: str
    stored_filename: str
    upload_time: str
    size_bytes: int
    processed: bool
    indexed_chunk_count: int = 0
    processing_status: str = "uploaded"
    last_error: Optional[str] = None
    last_indexed_at: Optional[str] = None
    active_index_job_id: Optional[str] = None
    parser_names: List[str] = Field(default_factory=list)
    detected_content_types: List[str] = Field(default_factory=list)
    detected_structural_roles: List[str] = Field(default_factory=list)
    parsed_element_count: int = 0


class DocumentUploadResponse(BaseModel):
    documents: List[DocumentRecord]


class IndexRequest(BaseModel):
    project_id: Optional[str] = None
    document_ids: Optional[List[str]] = None


class IndexResponse(BaseModel):
    project_id: str
    indexed_documents: int
    indexed_chunks: int
    skipped_documents: int
    documents: List[DocumentRecord]


class IndexJobRequest(BaseModel):
    project_id: Optional[str] = None
    document_ids: Optional[List[str]] = None


class IndexJobResponse(BaseModel):
    job_id: str
    project_id: str
    status: str
    document_ids: List[str]
    submitted_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    indexed_documents: int = 0
    indexed_chunks: int = 0
    skipped_documents: int = 0
    error: Optional[str] = None


class SessionSummary(BaseModel):
    session_id: str
    project_id: str
    title: str
    start_time: str


class MessageRecord(BaseModel):
    role: str
    content: str
    timestamp: str


class CreateSessionRequest(BaseModel):
    project_id: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    project_id: str


class RetrievedSource(BaseModel):
    filename: str
    score: float
    snippet: str
    metadata: Dict[str, Any]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    use_fusion: Optional[bool] = None
    num_variants: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    session_id: str
    project_id: str
    response: str
    tool_name: Optional[str] = None
    sources: List[RetrievedSource]


class SettingsResponse(BaseModel):
    llm_model: str
    llm_provider: str
    embedding_provider: str
    embedding_model: str
    vector_store_backend: str
    top_k: int
    chunk_size: int
    overlap: int
    enable_fusion: bool
    enable_hybrid_search: bool
    index_batch_size: int
    index_max_workers: int
    upload_max_file_size_mb: int


class LLMValidationResponse(BaseModel):
    provider: str
    configured: bool
    valid: bool
    message: str
