from typing import List

from fastapi import APIRouter

from backend.api.schemas import CreateSessionResponse, MessageRecord, SessionSummary
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    service = get_rag_application()
    return [SessionSummary(**session) for session in service.list_sessions()]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session() -> CreateSessionResponse:
    service = get_rag_application()
    session_id = service.create_session()
    return CreateSessionResponse(session_id=session_id)


@router.get("/sessions/{session_id}/messages", response_model=List[MessageRecord])
def get_session_messages(session_id: str) -> List[MessageRecord]:
    service = get_rag_application()
    return [MessageRecord(**message) for message in service.get_session_messages(session_id)]
