import faiss
import numpy as np
import json
import logging
import os
import atexit
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable
from chunker import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    chunk: DocumentChunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Minimal vector-store contract used by the app and retriever."""

    def add_chunks(self, chunks: List[DocumentChunk], persist: bool = True) -> None:
        ...

    def persist(self) -> None:
        ...

    def reset(self, persist: bool = True) -> None:
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        ...

    def get_all_chunks(self) -> List[DocumentChunk]:
        ...

    def close(self) -> None:
        ...

class FAISSVectorStore:
    """FAISS-based vector storage and retrieval, from your script."""
    def __init__(
        self,
        dimension: int,
        persist_path: str = "./vector_db",
        use_gpu: bool = False,
        nlist: int = 100,
        nprobe: int = 10,
        quantize: bool = False,
        m: int = 8,
        bits: int = 8
    ):
        """Initialize FAISS vector store with configurable options.
        
        Args:
            dimension: Embedding dimension
            persist_path: Path to store index files
            use_gpu: Whether to use GPU acceleration
            nlist: Number of IVF clusters
            nprobe: Number of IVF clusters to search
            quantize: Whether to use product quantization
            m: Number of subquantizers for PQ
            bits: Bits per subquantizer for PQ
        """
        self.dimension = dimension
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(exist_ok=True, parents=True)
        self.index_file = self.persist_path / "index.faiss"
        self.metadata_file = self.persist_path / "metadata.json"
        
        self.use_gpu = use_gpu
        self.nlist = nlist
        self.nprobe = nprobe
        self.quantize = quantize
        self.m = m
        self.bits = bits
        
        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()
        
        self._init_index()
        self.chunks: List[DocumentChunk] = []
        self._load_index()
    
    def add_chunks(self, chunks: List[DocumentChunk], persist: bool = True):
        """Add chunks with embeddings to the vector store."""
        if not chunks or chunks[0].embedding is None:
            logger.warning("No chunks with embeddings provided to add.")
            return
        
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            if len(embeddings) < max(2, self.nlist):
                logger.info("Dataset too small for IVF training. Using a flat index for now.")
                flat_index = faiss.IndexFlatIP(self.dimension)
                flat_index.add(embeddings)
                self.index = flat_index
            else:
                logger.info("Training IVF index before first add.")
                self.index.train(embeddings)
                self.index.add(embeddings)
        else:
            self.index.add(embeddings)

        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(chunks)} chunks. Total chunks in store: {self.index.ntotal}")
        if persist:
            self._save_index()

    def persist(self):
        """Persist the current vector store state to disk."""
        self._save_index()

    def reset(self, persist: bool = True):
        """Reset the index and in-memory chunk list."""
        logger.info("Resetting vector store at %s", self.persist_path)
        self._init_index()
        self.chunks = []
        if persist:
            self._save_index()
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            return []
        
        # Query embedding is assumed to be normalized already
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
            
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1: # FAISS returns -1 for no result
                results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))
        return results
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Return all stored chunks."""
        return self.chunks
    
    def _save_index(self):
        """Save index and metadata to disk."""
        logger.info(f"Saving vector store to {self.persist_path}...")
        if self.use_gpu and hasattr(self, 'gpu_index'):
            index_to_save = faiss.index_gpu_to_cpu(self.gpu_index)
        else:
            index_to_save = self.index
        faiss.write_index(index_to_save, str(self.index_file))
        metadata_to_save = [
            {'content': chunk.content, 'metadata': chunk.metadata} for chunk in self.chunks
        ]
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_to_save, f, indent=2)
    
    def _init_index(self):
        """Initialize the FAISS index based on configuration."""
        quantizer = faiss.IndexFlatIP(self.dimension)
        
        if self.quantize:
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.dimension,
                self.nlist,
                self.m,
                self.bits
            )
        else:
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.dimension,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
        if self.use_gpu:
            self.gpu_index = faiss.index_cpu_to_gpu(
                self.gpu_resources,
                0,  # device number
                self.index
            )
    
    def _load_index(self):
        """Load index and metadata from disk."""
        if self.index_file.exists() and self.metadata_file.exists():
            logger.info(f"Loading vector store from {self.persist_path}...")
            
            # Check if existing index dimension matches expected dimension
            try:
                self.index = faiss.read_index(str(self.index_file))
                if hasattr(self.index, 'd') and self.index.d != self.dimension:
                    logger.warning(f"Dimension mismatch: existing index has dimension {self.index.d}, expected {self.dimension}")
                    logger.warning("Creating new index with correct dimension")
                    self.index = None  # Force creation of new index
                    # Don't return here - let it fall through to create new index
                else:
                    # Handle backward compatibility with old flat indexes
                    if isinstance(self.index, faiss.IndexFlat):
                        logger.info("Converting old flat index to IVF")
                        quantizer = faiss.IndexFlatIP(self.dimension)
                        new_index = faiss.IndexIVFFlat(
                            quantizer,
                            self.dimension,
                            self.nlist,
                            faiss.METRIC_INNER_PRODUCT
                        )
                        # Only train if index has sufficient vectors
                        if self.index.ntotal > 100:  # Minimum for meaningful clustering
                            vectors = self.index.reconstruct_n(0, min(1000, self.index.ntotal))
                            # Adjust cluster count if needed
                            n_clusters = min(100, len(vectors)//2)  # Ensure nx >= k
                            quantizer = faiss.IndexFlatL2(self.dimension)
                            new_index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters)
                            new_index.train(vectors)
                            new_index.add(vectors)
                            self.index = new_index
                        elif self.index.ntotal > 0:
                            # For small datasets, use flat index without clustering
                            new_index = faiss.IndexFlatL2(self.dimension)
                            new_index.add(self.index.reconstruct_n(0, self.index.ntotal))
                            self.index = new_index
                        else:
                            self.index = new_index
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self.index = None
        
        # If index doesn't exist, is None, or had dimension mismatch, create new one
        if self.index is None:
            logger.info("Creating new vector store index")
            self._init_index()
            self.chunks = []
            return
        
        # Load metadata if index exists
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            self.chunks = [DocumentChunk(content=m['content'], metadata=m['metadata']) for m in metadata_list]
            logger.info(f"Loaded {self.index.ntotal} vectors and {len(self.chunks)} chunks.")
        
        # GPU setup
        if self.use_gpu and self.index is not None:
            try:
                self.gpu_index = faiss.index_cpu_to_gpu(
                    self.gpu_resources,
                    0,
                    self.index
                )
            except Exception as e:
                print(f"Error moving index to GPU: {e}")
                self.use_gpu = False
            
    def benchmark(self, queries: np.ndarray, k_values: List[int] = [1, 5, 10]) -> Dict[str, Any]:
        """Run performance benchmarks on the index.
        
        Args:
            queries: Array of query embeddings
            k_values: List of k values to test
            
        Returns:
            Dictionary containing benchmark results
        """
        results = {}
        
        # Warmup
        self.search(queries[0], max(k_values))
        
        # Time searches
        import time
        for k in k_values:
            start = time.time()
            for query in queries:
                self.search(query, k)
            elapsed = time.time() - start
            results[f"latency_k_{k}"] = elapsed / len(queries)
            
        # Memory usage
        if hasattr(self, 'gpu_index'):
            results['memory_gpu'] = self.gpu_resources.getTempMemory()
        results['memory_cpu'] = self.index.getMemoryUsage()
        
        return results


class QdrantVectorStore:
    """Qdrant-backed vector storage and retrieval."""

    def __init__(
        self,
        dimension: int,
        persist_path: str = "./vector_db/qdrant",
        collection_name: str = "ragagument",
        url: str | None = None,
        api_key_env: str = "QDRANT_API_KEY",
        prefer_grpc: bool = False,
        timeout: int | None = 30,
    ):
        self.dimension = dimension
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.url = (url or "").strip() or None
        self.api_key_env = api_key_env
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.chunks: List[DocumentChunk] = []
        self.point_id_to_index: Dict[str, int] = {}

        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qdrant_models

        self.qdrant_models = qdrant_models

        if self.url:
            api_key = os.getenv(self.api_key_env, "").strip() or None
            self.client = QdrantClient(
                url=self.url,
                api_key=api_key,
                prefer_grpc=self.prefer_grpc,
                timeout=self.timeout,
            )
        else:
            self.persist_path.mkdir(exist_ok=True, parents=True)
            self.client = QdrantClient(path=str(self.persist_path))

        self._closed = False
        self._ensure_collection()
        self._load_points()
        atexit.register(self.close)

    def add_chunks(self, chunks: List[DocumentChunk], persist: bool = True):
        """Add chunks with embeddings to the Qdrant collection."""
        if not chunks or chunks[0].embedding is None:
            logger.warning("No chunks with embeddings provided to add.")
            return

        points = []
        for chunk in chunks:
            point_id = self._point_id_for_chunk(chunk)
            payload = {
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            points.append(
                self.qdrant_models.PointStruct(
                    id=point_id,
                    vector=np.asarray(chunk.embedding, dtype=np.float32).tolist(),
                    payload=payload,
                )
            )
            self._register_chunk(point_id, chunk)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        logger.info("Added %s chunks to Qdrant collection %s", len(points), self.collection_name)

    def persist(self):
        """Qdrant persists writes on upsert; no explicit flush is needed."""
        return None

    def reset(self, persist: bool = True):
        """Reset the Qdrant collection."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self._ensure_collection()
        self.chunks = []
        self.point_id_to_index = {}

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        metadata_filter: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """Search for similar chunks."""
        if top_k <= 0:
            return []
        if not self.chunks and not self.client.collection_exists(self.collection_name):
            return []

        query_filter = self._build_query_filter(metadata_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_embedding, dtype=np.float32).tolist(),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: List[RetrievalResult] = []
        for point in response.points:
            chunk = self._chunk_from_payload(point.payload or {})
            if chunk is None:
                continue
            results.append(RetrievalResult(chunk=chunk, score=float(point.score)))
        return results

    def close(self) -> None:
        return None

    def get_all_chunks(self) -> List[DocumentChunk]:
        """Return all chunks cached from Qdrant payloads."""
        return list(self.chunks)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.client.close()
        except Exception:
            logger.debug("Ignoring Qdrant client shutdown error.", exc_info=True)
        finally:
            self._closed = True

    def _ensure_collection(self):
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self.qdrant_models.VectorParams(
                size=self.dimension,
                distance=self.qdrant_models.Distance.COSINE,
            ),
        )

    def _load_points(self):
        self.chunks = []
        self.point_id_to_index = {}

        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in records:
                chunk = self._chunk_from_payload(record.payload or {})
                if chunk is None:
                    continue
                self._register_chunk(str(record.id), chunk)
            if offset is None:
                break

        logger.info("Loaded %s chunks from Qdrant collection %s", len(self.chunks), self.collection_name)

    def _chunk_from_payload(self, payload: Dict[str, Any]) -> DocumentChunk | None:
        content = payload.get("content")
        metadata = payload.get("metadata", {})
        if not isinstance(content, str):
            return None
        if not isinstance(metadata, dict):
            metadata = {}
        return DocumentChunk(content=content, metadata=metadata)

    def _register_chunk(self, point_id: str, chunk: DocumentChunk):
        existing_index = self.point_id_to_index.get(point_id)
        if existing_index is not None:
            self.chunks[existing_index] = chunk
            return

        self.point_id_to_index[point_id] = len(self.chunks)
        self.chunks.append(chunk)

    def _point_id_for_chunk(self, chunk: DocumentChunk) -> str:
        payload = {
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{serialized}"))

    def _build_query_filter(self, metadata_filter: Dict[str, Any] | None):
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            translated = self._translate_filter_condition(f"metadata.{key}", value)
            if translated:
                conditions.extend(translated)

        if not conditions:
            return None

        return self.qdrant_models.Filter(must=conditions)

    def _translate_filter_condition(self, field_path: str, value: Any):
        if isinstance(value, dict):
            conditions = []
            range_kwargs: Dict[str, float] = {}
            for operator, operand in value.items():
                normalized_operator = str(operator).strip().lower()
                if normalized_operator == "eq":
                    match_value = self._match_value_for_qdrant(operand)
                    if match_value is not None:
                        conditions.append(
                            self.qdrant_models.FieldCondition(
                                key=field_path,
                                match=self.qdrant_models.MatchValue(value=match_value),
                            )
                        )
                elif normalized_operator == "in":
                    match_values = self._match_any_values_for_qdrant(operand)
                    if match_values:
                        conditions.append(
                            self.qdrant_models.FieldCondition(
                                key=field_path,
                                match=self.qdrant_models.MatchAny(any=match_values),
                            )
                        )
                elif normalized_operator in {"gt", "gte", "lt", "lte"}:
                    numeric_value = self._numeric_value(operand)
                    if numeric_value is not None:
                        range_kwargs[normalized_operator] = numeric_value
                elif normalized_operator == "between" and isinstance(operand, (list, tuple)) and len(operand) == 2:
                    lower = self._numeric_value(operand[0])
                    upper = self._numeric_value(operand[1])
                    if lower is not None:
                        range_kwargs["gte"] = lower
                    if upper is not None:
                        range_kwargs["lte"] = upper

            if range_kwargs:
                conditions.append(
                    self.qdrant_models.FieldCondition(
                        key=field_path,
                        range=self.qdrant_models.Range(**range_kwargs),
                    )
                )

            return conditions

        if isinstance(value, list):
            match_values = self._match_any_values_for_qdrant(value)
            if match_values:
                return [
                    self.qdrant_models.FieldCondition(
                        key=field_path,
                        match=self.qdrant_models.MatchAny(any=match_values),
                    )
                ]
            return []

        match_value = self._match_value_for_qdrant(value)
        if match_value is None:
            return []

        return [
            self.qdrant_models.FieldCondition(
                key=field_path,
                match=self.qdrant_models.MatchValue(value=match_value),
            )
        ]

    def _match_value_for_qdrant(self, value: Any):
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value
        return None

    def _match_any_values_for_qdrant(self, value: Any):
        if not isinstance(value, list):
            return None
        normalized = [item for item in value if isinstance(item, (str, int)) and not isinstance(item, bool)]
        return normalized or None

    def _numeric_value(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None


def create_vector_store(
    backend: str,
    dimension: int,
    persist_path: str,
    **kwargs: Any,
) -> VectorStore:
    normalized_backend = str(backend).strip().lower()

    if normalized_backend == "faiss":
        return FAISSVectorStore(
            dimension=dimension,
            persist_path=persist_path,
            **kwargs,
        )

    if normalized_backend == "qdrant":
        qdrant_path = kwargs.pop("qdrant_path", persist_path)
        qdrant_collection_name = kwargs.pop("qdrant_collection_name", "ragagument")
        qdrant_url = kwargs.pop("qdrant_url", None)
        qdrant_api_key_env = kwargs.pop("qdrant_api_key_env", "QDRANT_API_KEY")
        qdrant_prefer_grpc = kwargs.pop("qdrant_prefer_grpc", False)
        qdrant_timeout = kwargs.pop("qdrant_timeout", 30)
        return QdrantVectorStore(
            dimension=dimension,
            persist_path=qdrant_path,
            collection_name=qdrant_collection_name,
            url=qdrant_url,
            api_key_env=qdrant_api_key_env,
            prefer_grpc=qdrant_prefer_grpc,
            timeout=qdrant_timeout,
        )

    raise ValueError(f"Unsupported vector store backend: {backend}")
