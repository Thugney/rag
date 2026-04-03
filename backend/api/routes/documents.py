from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.api.schemas import DocumentRecord, DocumentUploadResponse, IndexJobRequest, IndexJobResponse, IndexRequest, IndexResponse
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/documents", response_model=List[DocumentRecord])
def list_documents() -> List[DocumentRecord]:
    service = get_rag_application()
    return [DocumentRecord(**document) for document in service.list_documents()]


@router.post("/documents/upload", response_model=DocumentUploadResponse)
def upload_documents(files: List[UploadFile] = File(...)) -> DocumentUploadResponse:
    service = get_rag_application()
    documents = []
    for file in files:
        documents.append(service.save_upload_stream(file.filename or "upload.bin", file.file))
    return DocumentUploadResponse(documents=[DocumentRecord(**document) for document in documents])


@router.post("/documents/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    service = get_rag_application()
    result = service.index_documents(request.document_ids)
    return IndexResponse(
        indexed_documents=result["indexed_documents"],
        indexed_chunks=result["indexed_chunks"],
        skipped_documents=result["skipped_documents"],
        documents=[DocumentRecord(**document) for document in result["documents"]],
    )


@router.post("/documents/index-jobs", response_model=IndexJobResponse)
def start_index_job(request: IndexJobRequest) -> IndexJobResponse:
    service = get_rag_application()
    job = service.start_index_job(request.document_ids)
    return IndexJobResponse(**job)


@router.get("/documents/index-jobs/{job_id}", response_model=IndexJobResponse)
def get_index_job(job_id: str) -> IndexJobResponse:
    service = get_rag_application()
    try:
        job = service.get_index_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return IndexJobResponse(**job)
