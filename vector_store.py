import faiss
import numpy as np
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from chunker import DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    chunk: DocumentChunk
    score: float

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
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
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
