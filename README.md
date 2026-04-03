# RAGagument
[![GitHub](https://img.shields.io/badge/GitHub-Thugney-181717?style=flat&logo=github)](https://github.com/Thugney)
[![Blog](https://img.shields.io/badge/Blog-eriteach.com-0d9488?style=flat&logo=hugo)](https://blog.eriteach.com)
[![YouTube](https://img.shields.io/badge/YouTube-Eriteach-FF0000?style=flat&logo=youtube)](https://www.youtube.com/@robeleriteach)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Eriteach-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/eriteach/)
RAGagument is an API-first RAG workspace for enterprise document question answering.

The repo now contains only the active product direction:

- FastAPI backend for upload, indexing, sessions, settings, and chat
- Next.js frontend for the modern chat-first UX
- local file and FAISS persistence for the current single-node product phase
- DeepSeek for generation and Ollama or HuggingFace for embeddings

## Current Scope

What works now:

- upload PDF, DOCX, Markdown, and text files
- stream uploads to disk
- index documents through background jobs
- retrieve source chunks from FAISS
- chat against indexed documents
- inspect chat history, library state, and source traces in the new frontend

What is not in scope yet:

- authentication and password management
- tenant isolation
- billing and usage metering
- audit-grade compliance controls

## Architecture

- `backend/` contains the FastAPI layer
- `backend/services/rag_service.py` wraps the shared RAG capability layer
- `frontend/` contains the Next.js application
- root Python modules provide chunking, retrieval, generation, embeddings, and persistence

## Prerequisites

- Python 3.11+
- Node.js 20+
- Ollama running locally
- a local `.env` with `DEEPSEEK_API_KEY`

## Local Development

Backend:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\Activate.ps1
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo\frontend
npm run dev
```

Local URLs:

- Frontend: `http://127.0.0.1:3000`
- API health: `http://127.0.0.1:8000/api/health`

## Docker Development

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
docker compose -f docker-compose.dev.yml up --build
```

Docker URLs:

- Frontend: `http://127.0.0.1:3001`
- API health: `http://127.0.0.1:8001/api/health`

The dev compose file mounts the parent project `.env` into the backend container at runtime. That file is not tracked in git.

## API Surface

Current routes are exposed under `/api`:

- `/health`
- `/settings`
- `/settings/validate-llm`
- `/documents`
- `/documents/index`
- `/documents/index-jobs`
- `/sessions`
- `/chat/query`

## Repository Layout

```text
.
|-- .github/
|   +-- workflows/
|-- backend/
|   |-- api/
|   +-- services/
|-- frontend/
|-- chat_history_db.py
|-- chunker.py
|-- config.yaml
|-- config_loader.py
|-- docker-compose.dev.yml
|-- Dockerfile.api
|-- embedder.py
|-- embedding_factory.py
|-- generator.py
|-- Makefile
|-- requirements.api.txt
|-- requirements.txt
|-- retriever.py
|-- tools.py
|-- translations.py
+-- vector_store.py
```

## Validation

The current stack has been validated with:

- backend Python compile checks
- frontend production build
- Docker dev startup
- API health checks
- document upload and background indexing
- DeepSeek-backed chat responses

See [TESTING_GUIDE.md](./TESTING_GUIDE.md) for the smoke-test flow and [TECHNOLOGY_STACK.md](./TECHNOLOGY_STACK.md) for the active stack summary.

## Secrets

- `.env` is ignored
- uploaded documents are ignored
- FAISS indexes are ignored
- local chat databases are ignored

Do not commit API keys, uploaded data, or runtime databases.
