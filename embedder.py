import numpy as np
import ollama
import logging
from typing import List
from chunker import DocumentChunk

logger = logging.getLogger(__name__)

class OllamaEmbedder:
    """Ollama-based embedding service, from your script."""
    def __init__(self, model_name: str = "mxbai-embed-large:latest"):
        self.model_name = model_name
        self.client = ollama.Client()
        self.dimension = None
        self._verify_model()
    
    def _verify_model(self):
        """Verify that the embedding model is available and get its dimension."""
        try:
            test_embedding = self.embed_text("test")
            self.dimension = len(test_embedding)
            logger.info(f"Embedding model '{self.model_name}' ready (dimension: {self.dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.model_name}': {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalized embedding for a single text."""
        try:
            response = self.client.embeddings(model=self.model_name, prompt=text)
            embedding = np.array(response['embedding'], dtype=np.float32)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm != 0 else embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for multiple chunks in batches for efficiency."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Implementation Note: Your original code embedded one by one.
        # This version uses batching for significant performance improvement,
        # which is crucial for a responsive UI.
        
        texts_to_embed = [chunk.content for chunk in chunks]
        if not texts_to_embed:
            return []

        # The ollama client handles batching internally when given a list.
        if not isinstance(texts_to_embed, list):
            texts_to_embed = [texts_to_embed]
            
        embeddings = []
        for text in texts_to_embed:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            embeddings.append(np.array(response["embedding"], dtype=np.float32))
        
        # Normalize and assign embeddings back to chunks
        for i, chunk in enumerate(chunks):
            embedding = embeddings[i]
            norm = np.linalg.norm(embedding)
            chunk.embedding = embedding / norm if norm != 0 else embedding
            
        logger.info("Embedding generation complete.")
        return chunks