# Retrieval Evaluation

This folder holds the first executable slice from `ideas.md`: a repeatable retrieval benchmark.

## Purpose

The goal is to compare the current retrieval modes on the same corpus and query set before making larger architecture changes.

The runner scores:

- `hit@k`
- `recall@k`
- `mrr@k`
- `ndcg@k`

It compares these built-in profiles:

- `vector_only`
- `hybrid`
- `hybrid_rerank`
- `hybrid_rerank_query_transform`

## Run The Sample Dataset

From the repo root:

```powershell
Set-Location J:\workspace-full\projects\Rag\repo
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py --dataset evals\sample_retrieval_eval.json
```

Optional JSON output:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals\sample_retrieval_eval.json `
  --json-out evals\sample-report.json
```

Optional answer review using the selected LLM provider:

```powershell
.\.venv\Scripts\python.exe scripts\run_retrieval_eval.py `
  --dataset evals\sample_retrieval_eval.json `
  --evaluate-answers `
  --answer-profile hybrid_rerank_query_transform
```

That answer review is intentionally simple. It checks whether generated answers contain the expected fragments declared in the dataset. It is a grounding smoke test, not a full human evaluation replacement.

## Dataset Format

The dataset is a single JSON file with two required sections:

```json
{
  "name": "dataset-name",
  "description": "What this benchmark is trying to prove.",
  "corpus": [
    {
      "document_id": "unique-id",
      "path": "../README.md",
      "metadata": {
        "authority_level": "high"
      }
    }
  ],
  "queries": [
    {
      "query_id": "docker-stack",
      "query": "How do I start the Docker dev stack?",
      "relevant_document_ids": ["readme", "testing-guide"],
      "expected_answer_contains": ["docker compose -f docker-compose.dev.yml up --build"]
    }
  ]
}
```

Queries may also define an optional `metadata_filter` object. When present, the evaluation runner passes it directly to the retriever so dataset cases can target specific content types, parser paths, page ranges, sheet names, or row ranges.

## Notes

- `path` values are resolved relative to the dataset file.
- You can use inline text instead of `path` by providing a `text` field.
- `metadata_filter` values are optional and use the same structured filter format as the chat API.
- Metrics are document-level. Multiple chunks from the same document count as one ranked document.
- The runner builds a temporary FAISS index so it does not touch the live workspace index.

## Multimodal Regression

The repo also includes `multimodal_retrieval_eval.json` plus `scripts/run_multimodal_eval_suite.py`.

That suite:

- generates OCR image, scanned PDF, XLSX, and CSV fixtures
- runs parser smoke checks on those fixtures
- executes the retrieval benchmark against the multimodal dataset
