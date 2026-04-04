from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.api.schemas import DeleteDocumentResponse, DocumentRecord, DocumentUploadResponse, IndexJobRequest, IndexJobResponse, IndexRequest, IndexResponse
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/documents", response_model=List[DocumentRecord])
def list_documents(project_id: Optional[str] = Query(default=None)) -> List[DocumentRecord]:
    service = get_rag_application()
    return [DocumentRecord(**document) for document in service.list_documents(project_id)]


@router.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_documents(
    files: List[UploadFile] = File(...),
    project_id: Optional[str] = Query(default=None),
) -> DocumentUploadResponse:
    service = get_rag_application()
    documents = []
    for file in files:
        documents.append(service.save_upload_stream(file.filename or "upload.bin", file.file, project_id))
    return DocumentUploadResponse(documents=[DocumentRecord(**document) for document in documents])


@router.post("/documents/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    service = get_rag_application()
    result = service.index_documents(request.project_id, request.document_ids)
    return IndexResponse(
        project_id=result["project_id"],
        indexed_documents=result["indexed_documents"],
        indexed_chunks=result["indexed_chunks"],
        skipped_documents=result["skipped_documents"],
        documents=[DocumentRecord(**document) for document in result["documents"]],
    )


@router.post("/documents/index-jobs", response_model=IndexJobResponse)
def start_index_job(request: IndexJobRequest) -> IndexJobResponse:
    service = get_rag_application()
    job = service.start_index_job(request.project_id, request.document_ids)
    return IndexJobResponse(**job)


@router.get("/documents/index-jobs/{job_id}", response_model=IndexJobResponse)
def get_index_job(job_id: str) -> IndexJobResponse:
    service = get_rag_application()
    try:
        job = service.get_index_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IndexJobResponse(**job)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(document_id: str, project_id: str = Query(...)) -> DeleteDocumentResponse:
    service = get_rag_application()
    try:
        result = service.delete_document(project_id, document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DeleteDocumentResponse(**result)
