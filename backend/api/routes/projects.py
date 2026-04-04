from typing import List

from fastapi import APIRouter, HTTPException

from backend.api.schemas import CreateProjectRequest, DeleteProjectResponse, ProjectSummary, UpdateProjectRequest
from backend.services.rag_service import get_rag_application


router = APIRouter()


@router.get("/projects", response_model=List[ProjectSummary])
def list_projects() -> List[ProjectSummary]:
    service = get_rag_application()
    return [ProjectSummary(**project) for project in service.list_projects()]


@router.post("/projects", response_model=ProjectSummary)
def create_project(request: CreateProjectRequest) -> ProjectSummary:
    service = get_rag_application()
    project = service.create_project(request.name, request.description)
    return ProjectSummary(**project)


@router.patch("/projects/{project_id}", response_model=ProjectSummary)
def update_project(project_id: str, request: UpdateProjectRequest) -> ProjectSummary:
    service = get_rag_application()
    try:
        project = service.update_project(project_id, request.name, request.description)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ProjectSummary(**project)


@router.delete("/projects/{project_id}", response_model=DeleteProjectResponse)
def delete_project(project_id: str) -> DeleteProjectResponse:
    service = get_rag_application()
    try:
        result = service.delete_project(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DeleteProjectResponse(**result)
