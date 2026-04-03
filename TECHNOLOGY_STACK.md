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

- FAISS for vector storage
- BM25 and hybrid retrieval logic
- Ollama as the default local embedding runtime
- HuggingFace as an optional embedding path
- DeepSeek for answer generation

Why:

- this keeps embeddings local and cheap
- it keeps generation quality high
- it avoids premature infrastructure complexity while the core product is still being hardened

## Persistence

- uploaded documents stored on disk
- chat sessions stored in SQLite
- vector indexes stored locally

Why:

- this is still a capability-first product
- local persistence is enough for the current single-workspace phase

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
