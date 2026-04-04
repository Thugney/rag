from typing import List, Optional

from fastapi import APIRouter, Query

from backend.api.schemas import CreateSessionRequest, CreateSessionResponse, MessageRecord, SessionSummary
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions(project_id: Optional[str] = Query(default=None)) -> List[SessionSummary]:
    service = get_rag_application()
    return [SessionSummary(**session) for session in service.list_sessions(project_id)]


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    service = get_rag_application()
    session_id = service.create_session(request.project_id)
    project_id = service.chat_db.get_session_project_id(session_id)
    return CreateSessionResponse(session_id=session_id, project_id=project_id)


@router.get("/sessions/{session_id}/messages", response_model=List[MessageRecord])
def get_session_messages(session_id: str) -> List[MessageRecord]:
    service = get_rag_application()
    return [MessageRecord(**message) for message in service.get_session_messages(session_id)]
