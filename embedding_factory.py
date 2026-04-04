import logging
import os
from typing import Any, Optional
import numpy as np
import ollama
from chunker import DocumentChunk

logger = logging.getLogger(__name__)

class EmbeddingFactory:
    """Factory class to create embeddings using different providers"""
    
    def __init__(self, provider: str = "huggingface", 
                 huggingface_model: str = "all-MiniLM-L6-v2",
                 ollama_model: str = "mxbai-embed-large:latest"):
        self.provider = provider
        self.huggingface_model = huggingface_model
        self.ollama_model = ollama_model
        self.dimension: Optional[int] = None
        self.model: Optional[Any] = None
        self.max_input_words = 192 if provider == "ollama" else None
        
        if provider == "huggingface":
            self._init_huggingface()
        elif provider == "ollama":
            self._init_ollama()
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace embeddings require the 'sentence-transformers' package."
            ) from exc

        logger.info(f"Initializing HuggingFace embedding model: {self.huggingface_model}")
        self.model = SentenceTransformer(self.huggingface_model)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"HuggingFace model '{self.huggingface_model}' ready (dimension: {self.dimension})")
    
    def _init_ollama(self):
        """Initialize Ollama embedding model"""
        logger.info(f"Initializing Ollama embedding model: {self.ollama_model}")
        self.client = ollama.Client(host=self._resolve_ollama_host())
        
        # Test the model to get dimension
        try:
            test_response = self.client.embeddings(model=self.ollama_model, prompt="test")
            self.dimension = len(test_response['embedding'])
            logger.info(f"Ollama model '{self.ollama_model}' ready (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model '{self.ollama_model}': {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalized embedding for a single text."""
        try:
            prepared_text = self._prepare_text(text)
            if self.provider == "huggingface":
                embedding = self.model.encode(prepared_text, convert_to_numpy=True)
            elif self.provider == "ollama":
                embedding = self._embed_with_ollama(prepared_text)
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm != 0 else embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding with {self.provider}: {e}")
            raise
    
    def embed_chunks(self, chunks: list) -> list:
        """Generate embeddings for multiple chunks."""
        if not chunks:
            return []
            
        logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.provider}...")
        
        texts_to_embed = [chunk.content for chunk in chunks]
        
        if self.provider == "huggingface":
            # Use batch processing for HuggingFace
            prepared_texts = [self._prepare_text(text) for text in texts_to_embed]
            embeddings = self.model.encode(prepared_texts, convert_to_numpy=True, batch_size=32)
        elif self.provider == "ollama":
            # Ollama doesn't support batch embedding, so do one by one
            embeddings = []
            for text in texts_to_embed:
                prepared_text = self._prepare_text(text)
                embeddings.append(self._embed_with_ollama(prepared_text))
            embeddings = np.array(embeddings)
        
        # Normalize and assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i]
            norm = np.linalg.norm(embedding)
            chunk.embedding = embedding / norm if norm != 0 else embedding
            
        logger.info("Embedding generation complete.")
        return chunks
    
    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def _resolve_ollama_host(self) -> str:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip().strip('"')
        if "://" not in host:
            host = f"http://{host}"
        if "0.0.0.0" in host:
            host = host.replace("0.0.0.0", "127.0.0.1")
        return host

    def _prepare_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        if self.max_input_words is None:
            return cleaned

        words = cleaned.split()
        if len(words) <= self.max_input_words:
            return cleaned

        truncated = " ".join(words[: self.max_input_words])
        logger.warning(
            "Embedding input truncated from %s to %s words for provider %s",
            len(words),
            self.max_input_words,
            self.provider,
        )
        return truncated

    def _embed_with_ollama(self, text: str) -> np.ndarray:
        prepared_text = text
        attempt_words = prepared_text.split()

        while True:
            try:
                response = self.client.embeddings(model=self.ollama_model, prompt=prepared_text)
                return np.array(response['embedding'], dtype=np.float32)
            except Exception as exc:
                message = str(exc).lower()
                if "input length exceeds the context length" not in message:
                    raise

                if len(attempt_words) <= 32:
                    raise

                next_limit = max(32, int(len(attempt_words) * 0.8))
                if next_limit >= len(attempt_words):
                    next_limit = len(attempt_words) - 1

                attempt_words = attempt_words[:next_limit]
                prepared_text = " ".join(attempt_words)
                logger.warning(
                    "Retrying Ollama embedding with %s words after a context-length failure.",
                    len(attempt_words),
                )
