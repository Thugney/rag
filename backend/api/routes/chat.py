from fastapi import APIRouter

from backend.api.schemas import ChatRequest, ChatResponse
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.post("/chat/query", response_model=ChatResponse)
def query_chat(request: ChatRequest) -> ChatResponse:
    service = get_rag_application()
    result = service.query(
        query=request.query,
        project_id=request.project_id,
        session_id=request.session_id,
        top_k=request.top_k,
        use_fusion=request.use_fusion,
        num_variants=request.num_variants,
        metadata_filter=request.metadata_filter,
    )
    return ChatResponse(**result)
