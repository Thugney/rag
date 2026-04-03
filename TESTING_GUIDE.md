# Testing Guide

This guide covers the smoke tests that match the current repo.

## What Should Work

- FastAPI backend startup
- Next.js frontend startup
- Docker dev startup
- streamed document upload
- background indexing jobs
- DeepSeek-backed chat answers with sources

## Prerequisites

- Python 3.11+
- Node.js 20+
- Ollama running locally
- a valid `DEEPSEEK_API_KEY` in a local `.env`

## Local Smoke Test

### 1. Start the backend

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\Activate.ps1
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the frontend

```powershell
Set-Location J:\workspace-full\projects\Rag\repo\frontend
npm run dev
```

### 3. Check health

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health
Invoke-WebRequest http://127.0.0.1:3000
```

Expected result:

- both return `200`

### 4. Validate DeepSeek

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/api/settings/validate-llm
```

Expected result:

- validation succeeds

### 5. Upload, index, and chat

Use the frontend at `http://127.0.0.1:3000`.

Expected result:

- upload succeeds
- indexing job completes
- chat returns an answer with sources

## Docker Smoke Test

### 1. Start the dev stack

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
docker compose -f docker-compose.dev.yml up --build
```

### 2. Check the services

```powershell
docker compose -f docker-compose.dev.yml ps
Invoke-WebRequest http://127.0.0.1:8001/api/health
Invoke-WebRequest http://127.0.0.1:3001
```

Expected result:

- backend container is healthy
- frontend container is running
- both endpoints return `200`

### 3. Repeat the upload, indexing, and chat flow in the UI

Open `http://127.0.0.1:3001`.

Expected result:

- upload works
- indexing completes
- chat responds with sources

## Build Validation

### Frontend build

```powershell
Set-Location J:\workspace-full\projects\Rag\repo\frontend
npm run build
```

### Backend syntax validation

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\Activate.ps1
python -m compileall backend chat_history_db.py chunker.py config_loader.py embedder.py embedding_factory.py generator.py retriever.py tools.py translations.py vector_store.py
```

## Secret Hygiene Check

Before pushing, confirm the repo is not tracking local secrets or runtime data:

```powershell
git ls-files | Select-String -Pattern "\.env$|uploaded_docs|vector_db|chat_history\.db|\.sqlite"
```

Expected result:

- no real `.env`
- no uploaded documents
- no local vector data
- no chat database files
