from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chunker import DocumentChunk, SmartChunker  # noqa: E402
from config_loader import Config  # noqa: E402
from embedding_factory import EmbeddingFactory  # noqa: E402
from generator import GeneratorFactory  # noqa: E402
from retriever import AdvancedRetriever  # noqa: E402
from vector_store import RetrievalResult, VectorStore, create_vector_store  # noqa: E402


logger = logging.getLogger("retrieval_eval")


@dataclass(frozen=True)
class RetrievalProfile:
    name: str
    description: str
    use_fusion: bool
    overrides: Dict[str, Any]


DEFAULT_PROFILES: Dict[str, RetrievalProfile] = {
    "vector_only": RetrievalProfile(
        name="vector_only",
        description="Dense vector retrieval only.",
        use_fusion=False,
        overrides={
            "advanced": {
                "enable_hybrid_search": False,
                "query_expansion": {
                    "techniques": {
                        "llm_variants": {"enable": False},
                        "synonym_expansion": {"enable": False},
                        "entity_recognition": {"enable": False},
                        "contextual_broadening": {"enable": False},
                    }
                },
                "reranking": {"enable": False},
            }
        },
    ),
    "hybrid": RetrievalProfile(
        name="hybrid",
        description="Vector retrieval plus BM25 score fusion.",
        use_fusion=False,
        overrides={
            "advanced": {
                "enable_hybrid_search": True,
                "query_expansion": {
                    "techniques": {
                        "llm_variants": {"enable": False},
                        "synonym_expansion": {"enable": False},
                        "entity_recognition": {"enable": False},
                        "contextual_broadening": {"enable": False},
                    }
                },
                "reranking": {"enable": False},
            }
        },
    ),
    "hybrid_rerank": RetrievalProfile(
        name="hybrid_rerank",
        description="Hybrid retrieval with the current reranking pipeline.",
        use_fusion=False,
        overrides={
            "advanced": {
                "enable_hybrid_search": True,
                "query_expansion": {
                    "techniques": {
                        "llm_variants": {"enable": False},
                        "synonym_expansion": {"enable": False},
                        "entity_recognition": {"enable": False},
                        "contextual_broadening": {"enable": False},
                    }
                },
                "reranking": {"enable": True},
            }
        },
    ),
    "hybrid_rerank_query_transform": RetrievalProfile(
        name="hybrid_rerank_query_transform",
        description="Hybrid retrieval, reranking, and query transformation.",
        use_fusion=True,
        overrides={
            "advanced": {
                "enable_hybrid_search": True,
                "reranking": {"enable": True},
            }
        },
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval evaluation profiles against a declared corpus and query set."
    )
    parser.add_argument("--dataset", required=True, help="Path to the JSON evaluation dataset.")
    parser.add_argument(
        "--profiles",
        default="vector_only,hybrid,hybrid_rerank,hybrid_rerank_query_transform",
        help="Comma-separated profile names to execute.",
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated k values used for hit, recall, MRR, and nDCG.",
    )
    parser.add_argument("--json-out", help="Optional path for a JSON report.")
    parser.add_argument(
        "--evaluate-answers",
        action="store_true",
        help="Generate answer-review checks for queries that define expected_answer_contains.",
    )
    parser.add_argument(
        "--answer-profile",
        help="Profile name to use for optional answer review. Defaults to the last selected profile.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for the evaluation runner.",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
    )


def load_dataset(dataset_path: Path) -> Dict[str, Any]:
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw.get("corpus"), list) or not raw["corpus"]:
        raise ValueError("Dataset must define a non-empty 'corpus' list.")
    if not isinstance(raw.get("queries"), list) or not raw["queries"]:
        raise ValueError("Dataset must define a non-empty 'queries' list.")

    corpus_ids = set()
    for index, document in enumerate(raw["corpus"], start=1):
        document_id = str(document.get("document_id", "")).strip()
        if not document_id:
            raise ValueError(f"Corpus entry #{index} is missing 'document_id'.")
        if document_id in corpus_ids:
            raise ValueError(f"Duplicate corpus document_id '{document_id}'.")
        corpus_ids.add(document_id)
        if "path" not in document and "text" not in document:
            raise ValueError(f"Corpus entry '{document_id}' must define either 'path' or 'text'.")

    for index, query in enumerate(raw["queries"], start=1):
        query_id = str(query.get("query_id", "")).strip()
        query_text = str(query.get("query", "")).strip()
        relevant_ids = query.get("relevant_document_ids")
        if not query_id:
            raise ValueError(f"Query entry #{index} is missing 'query_id'.")
        if not query_text:
            raise ValueError(f"Query entry '{query_id}' is missing 'query'.")
        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(f"Query entry '{query_id}' must define non-empty 'relevant_document_ids'.")
        unknown_ids = [doc_id for doc_id in relevant_ids if doc_id not in corpus_ids]
        if unknown_ids:
            raise ValueError(
                f"Query entry '{query_id}' references unknown relevant documents: {', '.join(unknown_ids)}"
            )
        metadata_filter = query.get("metadata_filter")
        if metadata_filter is not None and not isinstance(metadata_filter, dict):
            raise ValueError(f"Query entry '{query_id}' must define metadata_filter as an object when present.")

    return raw


def parse_k_values(raw: str) -> List[int]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("k values must be positive integers.")
        values.append(value)
    if not values:
        raise ValueError("At least one k value is required.")
    return sorted(set(values))


def get_embedder(config: Config) -> EmbeddingFactory:
    provider = str(config.get("embedding.provider", "ollama"))
    if provider == "huggingface":
        return EmbeddingFactory(
            provider="huggingface",
            huggingface_model=str(config.get("embedding.huggingface_model", "all-MiniLM-L6-v2")),
        )

    return EmbeddingFactory(
        provider="ollama",
        ollama_model=str(config.get("embedding.ollama_model", "mxbai-embed-large:latest")),
    )


def get_vector_store_kwargs(config: Config, temporary_root: Path) -> Dict[str, Any]:
    backend = str(config.get("vector_store.backend", "qdrant"))
    kwargs: Dict[str, Any] = {}
    if backend == "qdrant":
        kwargs.update(
            {
                "qdrant_path": str(temporary_root / "qdrant"),
                "qdrant_collection_name": "rag-eval",
                "qdrant_url": None,
                "qdrant_api_key_env": str(config.get("vector_store.qdrant.api_key_env", "QDRANT_API_KEY")),
                "qdrant_prefer_grpc": bool(config.get("vector_store.qdrant.prefer_grpc", False)),
                "qdrant_timeout": int(config.get("vector_store.qdrant.timeout", 30)),
            }
        )
    return kwargs


def deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_profile_config(base_config_path: Path, temporary_root: Path, profile: RetrievalProfile) -> Path:
    base_mapping = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    merged_mapping = deep_merge(deepcopy(base_mapping), profile.overrides)
    target_path = temporary_root / f"{profile.name}.config.yaml"
    target_path.write_text(yaml.safe_dump(merged_mapping, sort_keys=False), encoding="utf-8")
    return target_path


def batched(chunks: Sequence[DocumentChunk], batch_size: int) -> Iterator[List[DocumentChunk]]:
    for start in range(0, len(chunks), batch_size):
        yield list(chunks[start : start + batch_size])


def iter_corpus_batches(
    document: Dict[str, Any],
    dataset_dir: Path,
    chunker: SmartChunker,
    batch_size: int,
) -> Iterator[List[DocumentChunk]]:
    document_id = str(document["document_id"])
    metadata = dict(document.get("metadata", {}))

    if "path" in document:
        document_path = (dataset_dir / document["path"]).resolve()
        if not document_path.exists():
            raise FileNotFoundError(f"Corpus document not found: {document_path}")

        for batch in chunker.iter_document_chunk_batches(str(document_path), batch_size=batch_size):
            prepared_batch = []
            for chunk in batch:
                chunk.metadata.update(
                    {
                        "eval_document_id": document_id,
                        "document_id": document_id,
                        "filename": str(document.get("filename", document_path.name)),
                        "source": str(document_path),
                        **metadata,
                    }
                )
                prepared_batch.append(chunk)
            yield prepared_batch
        return

    filename = str(document.get("filename", f"{document_id}.txt"))
    inline_chunks = chunker.chunk_text(
        str(document["text"]),
        {
            "source": f"inline://{document_id}",
            "filename": filename,
            "eval_document_id": document_id,
            "document_id": document_id,
            **metadata,
        },
    )
    for batch in batched(inline_chunks, batch_size):
        yield batch


def build_corpus_index(
    dataset: Dict[str, Any],
    dataset_dir: Path,
    base_config: Config,
    vector_store: VectorStore,
    embedder: EmbeddingFactory,
) -> Dict[str, Any]:
    chunker = SmartChunker(
        chunk_size=int(base_config.get("system.chunk_size", 256)),
        overlap=int(base_config.get("system.overlap", 64)),
    )
    batch_size = int(base_config.get("indexing.batch_size", 16))
    corpus_stats = {
        "document_count": len(dataset["corpus"]),
        "chunk_count": 0,
        "documents": {},
    }

    for document in dataset["corpus"]:
        document_id = str(document["document_id"])
        document_chunk_count = 0
        for batch in iter_corpus_batches(document, dataset_dir, chunker, batch_size):
            embedded_chunks = embedder.embed_chunks(batch)
            vector_store.add_chunks(embedded_chunks, persist=False)
            document_chunk_count += len(embedded_chunks)
        corpus_stats["documents"][document_id] = {
            "filename": str(
                document.get(
                    "filename",
                    Path(document["path"]).name if "path" in document else f"{document_id}.txt",
                )
            ),
            "chunk_count": document_chunk_count,
        }
        corpus_stats["chunk_count"] += document_chunk_count

    vector_store.persist()
    return corpus_stats


def rank_documents(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    ranked_documents = []
    seen_document_ids = set()
    for result in results:
        metadata = result.chunk.metadata
        document_id = str(metadata.get("eval_document_id") or metadata.get("document_id") or metadata.get("filename"))
        if document_id in seen_document_ids:
            continue

        seen_document_ids.add(document_id)
        ranked_documents.append(
            {
                "rank": len(ranked_documents) + 1,
                "document_id": document_id,
                "filename": str(metadata.get("filename", "unknown")),
                "score": float(result.score),
                "chunk_index": metadata.get("chunk_index"),
                "snippet": result.chunk.content[:220],
            }
        )
    return ranked_documents


def hit_at_k(ranked_document_ids: Sequence[str], relevant_document_ids: set[str], k: int) -> float:
    return 1.0 if any(document_id in relevant_document_ids for document_id in ranked_document_ids[:k]) else 0.0


def recall_at_k(ranked_document_ids: Sequence[str], relevant_document_ids: set[str], k: int) -> float:
    if not relevant_document_ids:
        return 0.0
    matches = len(set(ranked_document_ids[:k]).intersection(relevant_document_ids))
    return matches / len(relevant_document_ids)


def reciprocal_rank_at_k(ranked_document_ids: Sequence[str], relevant_document_ids: set[str], k: int) -> float:
    for rank, document_id in enumerate(ranked_document_ids[:k], start=1):
        if document_id in relevant_document_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_document_ids: Sequence[str], relevant_document_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, document_id in enumerate(ranked_document_ids[:k], start=1):
        if document_id in relevant_document_ids:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant_document_ids), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def summarize_profile_metrics(query_results: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    query_count = max(1, len(query_results))
    for k in k_values:
        summary[f"hit_rate@{k}"] = sum(result["metrics"][f"hit@{k}"] for result in query_results) / query_count
        summary[f"recall@{k}"] = sum(result["metrics"][f"recall@{k}"] for result in query_results) / query_count
        summary[f"mrr@{k}"] = sum(result["metrics"][f"mrr@{k}"] for result in query_results) / query_count
        summary[f"ndcg@{k}"] = sum(result["metrics"][f"ndcg@{k}"] for result in query_results) / query_count
    return summary


def run_profile(
    profile: RetrievalProfile,
    profile_config_path: Path,
    vector_store: VectorStore,
    embedder: EmbeddingFactory,
    dataset: Dict[str, Any],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    max_k = max(k_values)
    config = Config(config_path=str(profile_config_path))
    retriever = AdvancedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        llm_model=str(config.get("llm.model", "deepseek-chat")),
        config_path=str(profile_config_path),
    )
    retriever.invalidate_cache()

    query_results = []
    for query in dataset["queries"]:
        query_metadata_filter = query.get("metadata_filter")
        results = retriever.retrieve(
            query=str(query["query"]),
            top_k=max_k,
            use_fusion=profile.use_fusion,
            num_variants=int(config.get("advanced.query_expansion.max_variants", 3)),
            metadata_filter=query_metadata_filter,
        )
        ranked_documents = rank_documents(results)
        ranked_document_ids = [document["document_id"] for document in ranked_documents]
        relevant_document_ids = {str(document_id) for document_id in query["relevant_document_ids"]}

        metrics = {}
        for k in k_values:
            metrics[f"hit@{k}"] = hit_at_k(ranked_document_ids, relevant_document_ids, k)
            metrics[f"recall@{k}"] = recall_at_k(ranked_document_ids, relevant_document_ids, k)
            metrics[f"mrr@{k}"] = reciprocal_rank_at_k(ranked_document_ids, relevant_document_ids, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_document_ids, relevant_document_ids, k)

        query_results.append(
            {
                "query_id": str(query["query_id"]),
                "query": str(query["query"]),
                "notes": str(query.get("notes", "")),
                "metadata_filter": deepcopy(query_metadata_filter) if query_metadata_filter else None,
                "relevant_document_ids": sorted(relevant_document_ids),
                "retrieved_documents": ranked_documents,
                "missing_relevant_document_ids": sorted(relevant_document_ids.difference(ranked_document_ids)),
                "metrics": metrics,
            }
        )

    return {
        "profile": profile.name,
        "description": profile.description,
        "use_fusion": profile.use_fusion,
        "summary": summarize_profile_metrics(query_results, k_values),
        "queries": query_results,
    }


def run_answer_review(
    profile_result: Dict[str, Any],
    profile_config_path: Path,
    vector_store: VectorStore,
    embedder: EmbeddingFactory,
    dataset: Dict[str, Any],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    config = Config(config_path=str(profile_config_path))
    retriever = AdvancedRetriever(
        vector_store=vector_store,
        embedder=embedder,
        llm_model=str(config.get("llm.model", "deepseek-chat")),
        config_path=str(profile_config_path),
    )
    retriever.invalidate_cache()
    generator = GeneratorFactory.from_config(config)
    validation = generator.validate_connection()
    if not validation["valid"]:
        return {
            "profile": profile_result["profile"],
            "provider_validation": validation,
            "queries": [],
        }

    profile = DEFAULT_PROFILES[profile_result["profile"]]
    max_k = max(k_values)
    review_queries = []
    total_queries = 0
    passing_queries = 0

    for query in dataset["queries"]:
        expected_fragments = [
            str(fragment).strip()
            for fragment in query.get("expected_answer_contains", [])
            if str(fragment).strip()
        ]
        if not expected_fragments:
            continue

        total_queries += 1
        retrieved_results = retriever.retrieve(
            query=str(query["query"]),
            top_k=max_k,
            use_fusion=profile.use_fusion,
            num_variants=int(config.get("advanced.query_expansion.max_variants", 3)),
            metadata_filter=query.get("metadata_filter"),
        )
        answer = "".join(generator.generate_response(str(query["query"]), retrieved_results)).strip()
        normalized_answer = answer.lower()
        missing_fragments = [
            fragment for fragment in expected_fragments if fragment.lower() not in normalized_answer
        ]
        passed = not missing_fragments
        if passed:
            passing_queries += 1

        review_queries.append(
            {
                "query_id": str(query["query_id"]),
                "query": str(query["query"]),
                "expected_answer_contains": expected_fragments,
                "missing_expected_fragments": missing_fragments,
                "passed": passed,
                "answer_preview": answer[:600],
            }
        )

    return {
        "profile": profile_result["profile"],
        "provider_validation": validation,
        "query_count": total_queries,
        "pass_rate": (passing_queries / total_queries) if total_queries else 0.0,
        "queries": review_queries,
    }


def print_report(report: Dict[str, Any], k_values: Sequence[int]) -> None:
    print(f"Dataset: {report['dataset']['name']}")
    print(f"Description: {report['dataset']['description']}")
    print(
        "Corpus: "
        f"{report['corpus']['document_count']} documents, "
        f"{report['corpus']['chunk_count']} chunks, "
        f"{report['dataset']['query_count']} queries"
    )
    print("")

    for profile in report["profiles"]:
        print(f"[{profile['profile']}] {profile['description']}")
        metrics_line = ", ".join(
            [
                f"hit@{k}={profile['summary'][f'hit_rate@{k}']:.3f} "
                f"recall@{k}={profile['summary'][f'recall@{k}']:.3f} "
                f"mrr@{k}={profile['summary'][f'mrr@{k}']:.3f} "
                f"ndcg@{k}={profile['summary'][f'ndcg@{k}']:.3f}"
                for k in k_values
            ]
        )
        print(f"  {metrics_line}")
        for query in profile["queries"]:
            top_documents = ", ".join(
                f"{entry['document_id']}#{entry['rank']}" for entry in query["retrieved_documents"][:3]
            ) or "none"
            print(
                f"  - {query['query_id']}: "
                f"top={top_documents}; "
                f"missing={','.join(query['missing_relevant_document_ids']) or 'none'}"
            )
        print("")

    if report.get("answer_review"):
        answer_review = report["answer_review"]
        print(f"[answer_review] profile={answer_review['profile']}")
        validation = answer_review["provider_validation"]
        print(f"  provider_valid={validation['valid']} message={validation['message']}")
        if answer_review["queries"]:
            print(f"  pass_rate={answer_review['pass_rate']:.3f}")
            for query in answer_review["queries"]:
                print(
                    f"  - {query['query_id']}: "
                    f"passed={query['passed']} "
                    f"missing={','.join(query['missing_expected_fragments']) or 'none'}"
                )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path)
    k_values = parse_k_values(args.k_values)
    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    if not profile_names:
        raise ValueError("At least one profile must be selected.")

    unknown_profiles = [name for name in profile_names if name not in DEFAULT_PROFILES]
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {', '.join(unknown_profiles)}")

    answer_profile_name = args.answer_profile or profile_names[-1]
    if args.evaluate_answers and answer_profile_name not in profile_names:
        raise ValueError("answer-profile must be included in --profiles.")

    base_config = Config(config_path=str(REPO_ROOT / "config.yaml"))
    embedder = get_embedder(base_config)

    with tempfile.TemporaryDirectory(prefix="rag-eval-") as temporary_root_raw:
        temporary_root = Path(temporary_root_raw)
        vector_store = create_vector_store(
            backend=str(base_config.get("vector_store.backend", "faiss")),
            dimension=int(embedder.get_dimension()),
            persist_path=str(temporary_root / "vector_store"),
            **get_vector_store_kwargs(base_config, temporary_root),
        )
        corpus_stats = build_corpus_index(
            dataset=dataset,
            dataset_dir=dataset_path.parent,
            base_config=base_config,
            vector_store=vector_store,
            embedder=embedder,
        )

        profile_results = []
        profile_config_paths: Dict[str, Path] = {}
        for profile_name in profile_names:
            profile = DEFAULT_PROFILES[profile_name]
            profile_config_path = write_profile_config(REPO_ROOT / "config.yaml", temporary_root, profile)
            profile_config_paths[profile_name] = profile_config_path
            profile_results.append(
                run_profile(
                    profile=profile,
                    profile_config_path=profile_config_path,
                    vector_store=vector_store,
                    embedder=embedder,
                    dataset=dataset,
                    k_values=k_values,
                )
            )

        report: Dict[str, Any] = {
            "dataset": {
                "name": str(dataset.get("name", dataset_path.name)),
                "description": str(dataset.get("description", "")),
                "path": str(dataset_path),
                "query_count": len(dataset["queries"]),
            },
            "corpus": corpus_stats,
            "profiles": profile_results,
        }

        if args.evaluate_answers:
            selected_profile = next(result for result in profile_results if result["profile"] == answer_profile_name)
            report["answer_review"] = run_answer_review(
                profile_result=selected_profile,
                profile_config_path=profile_config_paths[answer_profile_name],
                vector_store=vector_store,
                embedder=embedder,
                dataset=dataset,
                k_values=k_values,
            )

        print_report(report, k_values)

        if args.json_out:
            json_path = Path(args.json_out).resolve()
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"JSON report written to {json_path}")

        vector_store.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
