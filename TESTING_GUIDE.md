# Testing Guide

This guide covers the smoke tests that match the current repo.

## What Should Work

- FastAPI backend startup
- Next.js frontend startup
- Docker dev startup
- project creation
- streamed document upload
- background indexing jobs
- parser-aware TXT, Markdown, and DOCX ingestion
- layout-aware parsing for text-layer PDFs
- OCR-backed ingestion for image files and scanned-style PDFs
- spreadsheet ingestion for `.xlsx`, `.csv`, and `.tsv`
- project-scoped DeepSeek-backed chat answers with sources

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

### 5. Create a project, upload, index, and chat

Use the frontend at `http://127.0.0.1:3000`.

Expected result:

- project creation succeeds
- upload succeeds
- indexing job completes
- chat returns an answer with sources from the selected project

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

### 3. Repeat the project creation, upload, indexing, and chat flow in the UI

Open `http://127.0.0.1:3001`.

Expected result:

- project creation works
- upload works
- indexing completes
- chat responds with sources from the selected project

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

## Parser Seam Smoke Test

The parser layer now sits before chunking, so smoke-test it separately from chat.

Expected result:

- TXT, Markdown, and DOCX files produce structured elements before chunking
- chunk metadata includes parser, content type, structural role, and element identity
- text-layer PDF parsing should produce page-aware heading/body blocks before chunking
- image files and scanned-style PDFs should produce OCR-backed parsed elements with page/image provenance metadata
- spreadsheet files should produce `table` elements with workbook/sheet or delimiter metadata plus row-range provenance
- chunk metadata should include `chunk_strategy`, with headings staying atomic, tables preserving row groupings, and OCR/image blocks preserving line groupings

## Retrieval Evaluation

This is the first non-UI validation step from `ideas.md`.

Run the sample benchmark:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --dataset evals\sample_retrieval_eval.json
```

Expected result:

- the runner compares `vector_only`, `hybrid`, `hybrid_rerank`, and `hybrid_rerank_query_transform`
- it prints `hit@k`, `recall@k`, `mrr@k`, and `ndcg@k`
- it shows the top-ranked documents for each query

Optional answer review:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals\sample_retrieval_eval.json `
  --evaluate-answers `
  --answer-profile hybrid_rerank_query_transform
```

Expected result:

- the selected LLM provider validates successfully
- queries with `expected_answer_contains` get a simple pass/fail grounding check
- a failing answer review does not invalidate retrieval scoring; it tells us where answer generation still needs work

## Multimodal Regression Suite

Run the multimodal regression path:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\python.exe scripts\run_multimodal_eval_suite.py --profiles vector_only,hybrid
```

Expected result:

- fixture generation succeeds for OCR image, scanned PDF, XLSX, and CSV samples
- parser smoke checks pass before retrieval evaluation begins
- the multimodal dataset reports successful retrieval for the OCR, scanned PDF, and spreadsheet cases

## Frontend Multimodal UX Smoke

Open the library and sources drawers after indexing multimodal documents.

Expected result:

- document cards show parse-mode badges such as `Structured PDF`, `OCR image`, or `Spreadsheet`
- failed documents show the parser or indexing error inline
- source traces show provenance badges for modality, parser, chunk strategy, page or sheet context, and OCR source when available

## History Store Smoke Test

Expected result:

- the active SQLite-backed history store can still create projects, sessions, and messages
- `RAGApplication` no longer constructs the SQLite class directly; it consumes the history-store contract
- a future Postgres adapter can implement the same contract without changing the API service surface

## Metadata Filter Smoke Test

Use a temporary corpus or an indexed workspace and confirm retrieval can be narrowed by structured document metadata.

Expected result:

- filters like `content_type=image`, `sheet_name=Risk Register`, or `page_number >= 2` return only matching sources
- row-range filters such as `row_start >= 2` and `row_end <= 5` work for spreadsheet-derived chunks
- returned source metadata still includes parser, content type, page, sheet, and chunk strategy fields

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
