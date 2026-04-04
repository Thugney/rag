# RAGagument

RAGagument is an API-first RAG workspace for enterprise document question answering.

The product now uses a project-based workflow:

- create a project with a name and description
- upload the documents that belong to that project
- let indexing run in the background
- keep related chats inside that same project

The repo now contains only the active product direction:

- FastAPI backend for upload, indexing, sessions, settings, and chat
- Next.js frontend for the modern chat-first UX
- local file persistence plus Qdrant as the active vector backend for the current single-node product phase
- a history-store seam so SQLite can be replaced later without another wide service refactor
- a vector-store factory seam so FAISS is no longer hard-wired through the app
- OpenAI-compatible LLM providers for generation, with DeepSeek as the default runtime
- Ollama or HuggingFace for embeddings

## Current Scope

What works now:

- create project workspaces for grouped documents and chats
- upload PDF, DOCX, Markdown, and text files
- upload image files for OCR-backed ingestion
- upload `.xlsx`, `.csv`, and `.tsv` files for table-aware ingestion
- stream uploads to disk
- parse TXT, Markdown, and DOCX files into structured elements before chunking
- parse text-layer PDFs into page-aware layout blocks before chunking
- OCR standalone images and scanned-style PDFs through the built-in Windows OCR path
- parse spreadsheets into workbook/sheet/row-aware table blocks without adding a dedicated XLSX dependency
- chunk parsed content with modality-aware policies so headings stay atomic, tables stay row-aware, and OCR/image blocks stay line-aware
- index documents through background jobs
- retrieve source chunks from Qdrant by default
- filter retrieval by structured metadata such as parser, content type, page number, sheet name, and row ranges
- chat against indexed documents inside a selected project
- inspect project chat history, library state, and source traces in the new frontend
- surface parser, content-type, and structural metadata in document/library and source traces
- see parse-mode badges, inline processing failures, and richer provenance tags directly in the library and source drawers
- switch LLM providers through config without rewriting the generator layer
- benchmark retrieval profiles against a declared corpus and query set
- generate and run a multimodal regression suite covering OCR images, scanned PDFs, XLSX, and CSV fixtures

What is not in scope yet:

- authentication and password management
- tenant isolation
- billing and usage metering
- audit-grade compliance controls

Recommended next phase:

- Supabase Auth for email and password, reset flow, and session handling
- Postgres-backed project ownership and access control
- enterprise SSO after the core auth flow is stable
- metadata-rich retrieval, multimodal evaluation, and auth are the next roadmap layers on top of the parser and OCR foundation

## Architecture

- `backend/` contains the FastAPI layer
- `backend/services/rag_service.py` wraps the shared RAG capability layer
- `frontend/` contains the Next.js application
- root Python modules provide parsing, chunking, retrieval, generation, embeddings, and persistence
- `llm_factory.py` centralizes LLM provider configuration for generation and query expansion
- `vector_store.py` now exposes a backend-agnostic vector-store contract plus Qdrant and FAISS implementations
- `chunker.py` now exposes the parser seam that turns documents into structured elements before chunking
- `PdfDocumentParser` uses a visitor-based layout pass for text-layer PDFs and falls back to plain text when richer extraction is unavailable
- `SmartChunker` now applies chunking strategies by block type instead of forcing tables and OCR blocks through prose-only sentence splitting
- `retriever.py` now supports structured metadata filters, and Qdrant applies the translatable parts of those filters during search

Current data model:

- projects
- sessions within a project
- messages within a session
- documents stored and indexed for a project

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

## Retrieval Evaluation

The repo now includes a local retrieval benchmark so we can compare the current baseline before making bigger retrieval changes.

Run the sample dataset:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --dataset evals\sample_retrieval_eval.json
```

Optional grounded-answer smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals\sample_retrieval_eval.json `
  --evaluate-answers `
  --answer-profile hybrid_rerank_query_transform
```

The benchmark builds a temporary backend-specific index from the declared corpus, so it does not mutate the live workspace index under `vector_db`.

Run the multimodal regression suite:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\python.exe scripts\run_multimodal_eval_suite.py --profiles vector_only,hybrid
```

This generates the multimodal fixtures under `evals\fixtures\multimodal`, runs parser smoke checks on them, and then executes the retrieval benchmark on the multimodal dataset.

## API Surface

Current routes are exposed under `/api`:

- `/health`
- `/projects`
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
- project creation
- API health checks
- project-scoped document upload and background indexing
- project-scoped DeepSeek-backed chat responses
- retrieval benchmark execution across baseline profiles

See [TESTING_GUIDE.md](./TESTING_GUIDE.md) for the smoke-test and evaluation flow, [TECHNOLOGY_STACK.md](./TECHNOLOGY_STACK.md) for the active stack summary, and [evals/README.md](./evals/README.md) for the dataset format.

## Secrets

- `.env` is ignored
- uploaded documents are ignored
- FAISS indexes are ignored
- local chat databases are ignored

Do not commit API keys, uploaded data, or runtime databases.
