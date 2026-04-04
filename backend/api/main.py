from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.api.routes import chat, documents, health, projects, sessions, settings

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(REPO_ROOT.parent / ".env", override=False)

app = FastAPI(
    title="RAGagument API",
    version="0.2.0",
    description="Capability-first API foundation for the RAG product migration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(projects.router, prefix="/api", tags=["projects"])
app.include_router(settings.router, prefix="/api", tags=["settings"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(sessions.router, prefix="/api", tags=["sessions"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
