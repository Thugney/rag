from fastapi import APIRouter

from backend.api.schemas import LLMValidationResponse, SettingsResponse
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    service = get_rag_application()
    return SettingsResponse(**service.get_settings())


@router.get("/settings/validate-llm", response_model=LLMValidationResponse)
def validate_llm() -> LLMValidationResponse:
    service = get_rag_application()
    return LLMValidationResponse(**service.validate_llm())
