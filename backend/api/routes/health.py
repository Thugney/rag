from fastapi import APIRouter

from backend.api.schemas import HealthResponse
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    service = get_rag_application()
    return HealthResponse(**service.get_health())
