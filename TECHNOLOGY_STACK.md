# Technology Stack

This file describes the active stack in the repo today.

## Application

- Backend: FastAPI
- Frontend: Next.js 15, React 19, TypeScript
- Styling: Tailwind CSS
- Data fetching: TanStack Query

Why:

- the app now has a real API boundary
- the frontend is built for a modern chat workflow instead of a prototype dashboard

## RAG Layer

- Qdrant as the active vector store
- FAISS retained as a supported fallback backend
- a vector-store abstraction layer so the app and eval runner are backend-agnostic
- BM25 and hybrid retrieval logic
- parser-aware ingestion that turns supported files into structured elements before chunking
- visitor-based structured parsing for text-layer PDFs
- built-in Windows OCR via `powershell.exe` and `Windows.Media.Ocr` for image files and scanned-style PDFs
- standard-library spreadsheet parsing for `.xlsx`, `.csv`, and `.tsv`
- modality-aware chunking so headings, tables, and OCR/image blocks do not all collapse into the same prose chunking policy
- structured retrieval filters for parser, modality, page, sheet, and row-range metadata, with Qdrant payload filtering on the active backend
- Ollama as the default local embedding runtime
- HuggingFace as an optional embedding path
- OpenAI-compatible LLM provider abstraction for answer generation and query expansion
- DeepSeek as the default configured generation provider
- a local retrieval evaluation harness for profile comparison
- a multimodal regression suite that generates OCR image, scanned PDF, XLSX, and CSV fixtures before running parser smoke and retrieval evaluation
- a chat-first frontend that now surfaces parse mode, processing failures, and richer source provenance instead of hiding parser state in raw metadata

Why:

- this keeps embeddings local and cheap
- it keeps generation quality high
- it gives the ingestion layer a clean seam for richer PDF, OCR, spreadsheet, and multimodal work
- it removes single-vendor coupling from the generation layer while the core product is still being hardened

## Persistence

- uploaded documents stored on disk under project folders
- chat projects and sessions stored through a history-store contract, with SQLite as the current adapter
- vector indexes stored locally

Why:

- this is still a capability-first product
- local persistence is enough for the current single-workspace phase while the project model is being hardened
- the vector-store seam now carries the live Qdrant backend without rewriting the app and eval layers
- the history-store seam now lets us add a Postgres adapter later without another service-layer rewrite

## Runtime Modes

### Local

- API on `:8000`
- frontend on `:3000`

### Docker dev

- backend on `:8001`
- frontend on `:3001`

## CI

The repo includes a minimal GitHub Actions workflow that matches the current codebase:

- Python syntax validation through `compileall`
- frontend production build through `npm run build`

This is intentionally smaller than the old placeholder CI/CD setup because the goal is correctness, not inflated ops claims.

## Deferred

These are not implemented yet:

- authentication
- password lifecycle
- multi-tenant data isolation
- billing
- usage metering
- enterprise audit controls

Recommended next auth path:

- Supabase Auth for email, password, password reset, and session handling
- Postgres-backed project ownership once auth is introduced
