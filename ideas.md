# RAGagument Strategy Notes

This file turns the earlier rough notes into a concrete product and architecture plan.

Primary source of these ideas:

- https://github.com/HKUDS/RAG-Anything

Supporting visual reference in this repo:

- `image.png`

Execution tracker for the Tier 2 to Tier 3 upgrade:

- `tasks/all_type_rag_tasks.json`

Current graph decision:

- graph retrieval is deferred for now

## Core takeaway

Do not copy RAG-Anything wholesale.

Use it as a source of ideas for the parts our system is actually missing:

- stronger document parsing
- multimodal ingestion
- optional graph-based retrieval later

The image in this repo makes the right sequencing point:

- Tier 1: naive RAG is not enough
- Tier 2: production baseline comes first
- Tier 3: graph + multimodal comes after the baseline is proven
- Tier 4: agentic workflows are optional, not the first priority

That means our real path is:

1. prove and harden the production retrieval baseline
2. replace local-only persistence where SaaS demands it
3. upgrade ingestion and parsing
4. add multimodal and graph capabilities only where they are justified

Right now, the evidence supports multimodal parsing and evaluation work.
It does not yet support starting graph retrieval.

## What RAG-Anything is actually good for

Based on the official repo, RAG-Anything is strongest in these areas:

- multimodal document processing
- richer parsing of PDFs and Office files
- handling images, tables, equations, and mixed-layout documents
- graph-oriented retrieval on top of vector retrieval
- VLM-assisted query workflows

These are real strengths for enterprise knowledge systems with complex documents.

## What our repo already has

Our current repo is not Tier 1 anymore.

We already have some Tier 2 pieces in progress or partly implemented:

- project-based workspaces
- background indexing
- hybrid retrieval logic
- reranking hooks
- query expansion logic
- provider abstraction for embeddings
- new provider abstraction for the LLM layer
- modern API-first product structure

The real weakness is not that the code has zero advanced retrieval.
The real weakness is that we do not yet have a serious evaluation loop proving what actually works best on our documents.

## Main gaps to fix

### 1. Evaluation is still the real blocker

We need a repeatable retrieval and answer-quality benchmark.

Without that:

- we cannot prove hybrid search is helping
- we cannot prove reranking is helping
- we cannot justify a move to graph retrieval
- we cannot know whether multimodal parsing is worth its complexity

### 2. Ingestion is still too text-only

Today we mostly treat documents as extracted text.

That is weak for:

- tables
- charts
- diagrams
- scanned PDFs
- equations
- complex layouts

This is the highest-value place to borrow from RAG-Anything.

### 3. Persistence is still local-first

FAISS and SQLite are fine for capability development, but not for SaaS or multi-instance deployment.

We should treat these as transitional infrastructure.

### 4. Graph retrieval is interesting, not yet mandatory

Graph and multimodal retrieval are frontier capabilities.
They should come after we prove the production baseline, not before.

## Keep vs borrow

### Keep from our system

- FastAPI backend
- Next.js frontend
- project-based chat workspace model
- Docker and current repo structure
- security and CI hygiene
- current embeddings pattern
- current product direction toward SaaS

### Borrow or adapt from RAG-Anything

- parser-layer strategy
- multimodal content handling
- image/table/equation processing ideas
- optional graph retrieval ideas
- optional VLM-assisted query flows

### Do not copy directly yet

- full graph-first retrieval architecture
- full RAG-Anything dependency stack
- multimodal complexity without eval
- agentic orchestration

## Recommended roadmap

## Phase 1: Prove the production baseline

Goal:

- turn retrieval quality into something measurable

Deliverables:

- gold evaluation set for real enterprise documents
- retrieval metrics: recall, MRR, hit rate, nDCG
- answer quality review set
- side-by-side comparison of:
  - vector only
  - hybrid
  - hybrid + reranking
  - hybrid + reranking + query transformation

Decision rule:

- do not move to frontier retrieval until this phase tells us where the current system actually fails

## Phase 2: Finish the production retrieval baseline

Goal:

- harden the best-performing non-graph retrieval path

Deliverables:

- cleaned hybrid retrieval implementation
- reliable reranking path
- query transformation strategy
- semantic chunking review
- retrieval configuration profiles per document type

Important note:

Some of this already exists in code. The work here is to clean it up, validate it, and make it trustworthy.

## Phase 3: Replace local-only persistence

Goal:

- make the system compatible with SaaS and multi-instance deployment

Recommended changes:

- replace FAISS with Qdrant first
- replace SQLite chat storage with PostgreSQL

Why:

- Qdrant gives persistence, filtering, and a real service boundary
- PostgreSQL gives multi-user durability and future auth/ownership support

This is the correct infrastructure step before serious multi-user or enterprise rollout.

## Phase 4: Upgrade the parser and ingestion layer

Goal:

- handle real enterprise documents properly

Recommended direction:

- introduce a parser layer before chunking
- evaluate MinerU, Docling, and OCR-capable pipelines
- preserve structure for:
  - tables
  - figures
  - headings
  - captions
  - equations
  - page and section boundaries

This is the most valuable area to borrow from RAG-Anything first.

## Phase 5: Add selective multimodal capabilities

Goal:

- support documents where text-only ingestion is insufficient

Add only if the document set justifies it:

- image caption or vision interpretation
- table interpretation
- equation extraction
- multimodal retrieval metadata

This should be incremental, not a full rewrite.

## Phase 6: Evaluate graph retrieval

Goal:

- decide whether graph retrieval solves real unanswered problems

Use this phase only if we confirm that users need:

- cross-document reasoning
- entity relationship exploration
- thematic or global questions across many sources

If yes:

- prototype graph retrieval beside the main system
- do not replace the baseline retrieval path immediately

The likely outcome is hybrid coexistence, not graph-only retrieval.

## Phase 7: Agentic features last

Goal:

- add workflow automation only after retrieval and grounding are strong

Examples:

- multi-step research flow
- tool-driven querying
- structured report generation
- policy review workflows

These are product features, not the first architecture priority.

## Concrete decisions for this repo

### Decision 1

We should not integrate `RAG-Anything` directly as the main runtime right now.

### Decision 2

We should borrow its parser and multimodal ideas first.

### Decision 3

We should move to Qdrant and PostgreSQL before attempting graph-based SaaS features.

### Decision 4

We should treat graph retrieval as an experiment after evaluation, not as the default path.

### Decision 5

We should use evaluation as the gate for every major retrieval change.

## Immediate next implementation steps

1. Build a proper evaluation harness for retrieval and grounded answers.
2. Clean and verify the current hybrid + reranking + query transformation path.
3. Add a vector-store abstraction that lets us introduce Qdrant beside FAISS.
4. Add a persistence abstraction for chat history so PostgreSQL can replace SQLite cleanly.
5. Design a parser-layer interface before touching multimodal ingestion.
6. After the parser interface exists, test one richer parser pipeline on real enterprise documents.

## Recommended order of work

If we are disciplined, the order should be:

1. evaluation
2. retrieval cleanup
3. Qdrant
4. PostgreSQL
5. parser layer
6. multimodal ingestion
7. graph retrieval experiment
8. agentic workflows

## Summary

RAG-Anything is not the system we should copy.
It is the research and product reference we should selectively mine.

The right move is:

- baseline first
- infrastructure second
- ingestion third
- frontier retrieval fourth

That gets us to a real enterprise-grade RAG system without jumping into complexity before the foundation is proven.
