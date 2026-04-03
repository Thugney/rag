import os
import logging
from typing import List, Dict
import datetime
from functools import lru_cache
from threading import Lock
from openai import OpenAI  # <-- IMPORT THE OFFICIAL OPENAI CLIENT
from vector_store import FAISSVectorStore, RetrievalResult
from embedder import OllamaEmbedder
import yaml
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class CacheMetrics:
    """Track cache hit/miss statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.lock = Lock()
    
    def record_hit(self):
        with self.lock:
            self.hits += 1
            logger.info(f"Cache hit recorded. Total hits: {self.hits}, misses: {self.misses}, hit rate: {self.hit_rate():.2%}")
    
    def record_miss(self):
        with self.lock:
            self.misses += 1
            logger.info(f"Cache miss recorded. Total hits: {self.hits}, misses: {self.misses}, hit rate: {self.hit_rate():.2%}")
    
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        with self.lock:
            self.hits = 0
            self.misses = 0
            logger.info("Cache metrics reset.")

class AdvancedRetriever:
    """Advanced retrieval with fusion, re-ranking, LRU caching, hybrid search, and advanced re-ranking techniques."""
    def __init__(self, vector_store: FAISSVectorStore, embedder: OllamaEmbedder, llm_model: str, config_path: str = "config.yaml"):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm_model = llm_model
        self.cache_metrics = CacheMetrics()
        self.vector_store_update_count = 0
        self.query_expansion_metrics = {'llm_variants': 0, 'synonym_expansion': 0, 'entity_recognition': 0, 'contextual_broadening': 0}
        self.reranking_metrics = {'cross_encoder': 0, 'mmr': 0, 'position_decay': 0, 'source_authority': 0}
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.cache_max_size = config.get('cache', {}).get('max_size', 1000)
            self.enable_hybrid_search = config.get('advanced', {}).get('enable_hybrid_search', False)
            self.hybrid_weights = config.get('advanced', {}).get('hybrid_weights', {'vector': 0.6, 'bm25': 0.4})
            self.metadata_filters = config.get('advanced', {}).get('metadata_filters', {'enable': False, 'fields': []})
            self.query_expansion_config = config.get('advanced', {}).get('query_expansion', {'enable': True, 'max_variants': 3, 'techniques': {}})
            self.reranking_config = config.get('advanced', {}).get('reranking', {'enable': True, 'cross_encoder': {'enable': True, 'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2', 'batch_size': 32}, 'mmr': {'enable': True, 'lambda_param': 0.5}, 'position_decay': {'enable': True, 'decay_rate': 0.9}, 'source_authority': {'enable': True, 'weights': {'default': 1.0, 'high': 1.5, 'low': 0.5}}, 'latency_monitoring': {'enable': True}})
            logger.info(f"LRU Cache initialized with max size: {self.cache_max_size}")
            logger.info(f"Hybrid search enabled: {self.enable_hybrid_search}, Weights: {self.hybrid_weights}")
            logger.info(f"Metadata filters enabled: {self.metadata_filters['enable']}, Fields: {self.metadata_filters['fields']}")
            logger.info(f"Query expansion enabled: {self.query_expansion_config['enable']}, Max variants: {self.query_expansion_config['max_variants']}")
            logger.info(f"Reranking enabled: {self.reranking_config['enable']}, Cross-Encoder: {self.reranking_config['cross_encoder']['enable']}, MMR: {self.reranking_config['mmr']['enable']}, Position Decay: {self.reranking_config['position_decay']['enable']}, Source Authority: {self.reranking_config['source_authority']['enable']}")
        
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found for the retriever.")
            
        # --- KEY CHANGE: Initialize the OpenAI client and point it to DeepSeek's API endpoint ---
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.bm25_index = None
        self.chunk_texts = []
        self.chunk_ids = []
        self.cross_encoder = None
        self._initialize_bm25_index()
        self._initialize_cross_encoder()

    def retrieve(self, query: str, top_k: int, use_fusion: bool, num_variants: int, metadata_filter: Dict = None) -> List[RetrievalResult]:
        """Main retrieval method that decides between fusion, hybrid, or basic, and applies re-ranking if enabled."""
        if use_fusion:
            initial_results = self._retrieve_with_fusion(query, top_k * 2, num_variants, metadata_filter)
        elif self.enable_hybrid_search:
            initial_results = self._retrieve_hybrid(query, top_k * 2, metadata_filter)
        else:
            initial_results = self._retrieve_basic(query, top_k * 2, metadata_filter)
        
        if self.reranking_config['enable']:
            return self._apply_reranking(query, initial_results, top_k)
        return initial_results[:top_k]
    
    def _retrieve_basic(self, query: str, top_k: int, metadata_filter: Dict = None) -> List[RetrievalResult]:
        embedding = self._get_cached_embedding(query)
        results = self.vector_store.search(embedding, top_k * 2)
        if metadata_filter and self.metadata_filters['enable']:
            results = self._apply_metadata_filter(results, metadata_filter)
        return results[:top_k]

    def _retrieve_hybrid(self, query: str, top_k: int, metadata_filter: Dict = None) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector similarity and BM25 keyword matching."""
        start_time = datetime.datetime.now()
        # Vector similarity search
        embedding = self._get_cached_embedding(query)
        vector_results = self.vector_store.search(embedding, top_k * 2)
        vector_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Vector search completed in {vector_time:.3f} seconds, retrieved {len(vector_results)} results")
        
        # BM25 keyword search
        start_time = datetime.datetime.now()
        bm25_scores = self._get_bm25_scores(query)
        bm25_results = []
        for chunk_id, score in bm25_scores.items():
            for vr in vector_results:
                if hash(vr.chunk.content) == chunk_id:
                    bm25_results.append(RetrievalResult(chunk=vr.chunk, score=score))
                    break
        bm25_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"BM25 search completed in {bm25_time:.3f} seconds, retrieved {len(bm25_results)} results")
        
        # Combine scores
        start_time = datetime.datetime.now()
        combined_results = defaultdict(float)
        for vr in vector_results:
            chunk_key = hash(vr.chunk.content)
            combined_results[chunk_key] += vr.score * self.hybrid_weights['vector']
        for br in bm25_results:
            chunk_key = hash(br.chunk.content)
            combined_results[chunk_key] += br.score * self.hybrid_weights['bm25']
        
        # Convert to list and sort
        final_results = [RetrievalResult(chunk=vr.chunk, score=score) for chunk_key, score in combined_results.items() for vr in vector_results if hash(vr.chunk.content) == chunk_key]
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply metadata filter if enabled
        if metadata_filter and self.metadata_filters['enable']:
            pre_filter_count = len(final_results)
            final_results = self._apply_metadata_filter(final_results, metadata_filter)
            logger.info(f"Metadata filter applied, reduced results from {pre_filter_count} to {len(final_results)}")
        
        fusion_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Score fusion completed in {fusion_time:.3f} seconds, final results: {len(final_results[:top_k])}")
        return final_results[:top_k]

    def _retrieve_with_fusion(self, query: str, top_k: int, num_variants: int, metadata_filter: Dict = None) -> List[RetrievalResult]:
        """RAG Fusion: retrieve with multiple query variants."""
        variants = self._generate_query_variants(query, num_variants)
        logger.info(f"Generated query variants for fusion: {variants}")
        
        all_results: Dict[int, RetrievalResult] = {}
        for variant in variants:
            if self.enable_hybrid_search:
                results = self._retrieve_hybrid(variant, top_k, metadata_filter)
            else:
                embedding = self._get_cached_embedding(variant)
                results = self.vector_store.search(embedding, top_k)
            for res in results:
                chunk_key = hash(res.chunk.content)
                if chunk_key not in all_results:
                    all_results[chunk_key] = res
                else:
                    # Boost score for newer documents (within last 7 days)
                    upload_time_str = res.chunk.metadata.get('upload_time', '')
                    if upload_time_str:
                        try:
                            upload_time = datetime.datetime.fromisoformat(upload_time_str)
                            if (datetime.datetime.now() - upload_time).days < 7:
                                all_results[chunk_key].score += 0.2
                            else:
                                all_results[chunk_key].score += 0.1
                        except ValueError:
                            logger.warning(f"Invalid upload_time format: {upload_time_str}")

        final_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return final_results
    
    def _generate_query_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using multiple techniques with caching."""
        cache_key = f"{query}_{num_variants}"
        cached_result = self._get_cached_variants(cache_key)
        if cached_result is not None:
            self.cache_metrics.record_hit()
            return cached_result
        
        self.cache_metrics.record_miss()
        variants = [query]
        max_variants = min(num_variants, self.query_expansion_config.get('max_variants', 3))
        
        # LLM-generated variants
        if self.query_expansion_config.get('techniques', {}).get('llm_variants', {}).get('enable', True):
            variants.extend(self._generate_llm_variants(query, max_variants - 1))
            self.query_expansion_metrics['llm_variants'] += 1
        
        # Synonym expansion (placeholder for WordNet/NLTK)
        if self.query_expansion_config.get('techniques', {}).get('synonym_expansion', {}).get('enable', True):
            variants.extend(self._generate_synonym_variants(query, max_variants - len(variants)))
            self.query_expansion_metrics['synonym_expansion'] += 1
        
        # Entity recognition (placeholder for spaCy)
        if self.query_expansion_config.get('techniques', {}).get('entity_recognition', {}).get('enable', True):
            variants.extend(self._generate_entity_variants(query, max_variants - len(variants)))
            self.query_expansion_metrics['entity_recognition'] += 1
        
        # Contextual broadening
        if self.query_expansion_config.get('techniques', {}).get('contextual_broadening', {}).get('enable', True):
            variants.extend(self._generate_contextual_variants(query, max_variants - len(variants)))
            self.query_expansion_metrics['contextual_broadening'] += 1
        
        result = list(set(variants))[:max_variants]
        self._cache_variants(cache_key, result)
        if self.query_expansion_config.get('metrics', {}).get('enable', True):
            logger.info(f"Query expansion metrics: {self.query_expansion_metrics}")
        return result

    def _generate_llm_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using the DeepSeek API via the OpenAI client."""
        prompt = f"""You are an expert query writer. Rewrite the following user query in {num_variants} different ways to improve document retrieval from a vector database. Focus on using synonyms, rephrasing, and asking related sub-questions.

Original Query: "{query}"

Return ONLY the rewritten queries, each on a new line. Do not include numbering, bullets, or the original query."""
        
        variants = []
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            generated_queries = response.choices[0].message.content.strip().split('\n')
            variants.extend([q.strip() for q in generated_queries if q.strip()])
        except Exception as e:
            logger.warning(f"Failed to generate query variants with DeepSeek API: {e}")
        return variants[:num_variants]

    def _generate_synonym_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using synonym expansion (placeholder for WordNet/NLTK)."""
        # Placeholder implementation
        return [f"{query} synonym variant {i}" for i in range(num_variants)]

    def _generate_entity_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using entity recognition (placeholder for spaCy)."""
        # Placeholder implementation
        return [f"{query} entity variant {i}" for i in range(num_variants)]

    def _generate_contextual_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using contextual broadening."""
        # Placeholder implementation
        return [f"{query} contextual variant {i}" for i in range(num_variants)]
        
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings for queries and variants."""
        self.cache_metrics.record_miss()
        return self.embedder.embed_text(text)
        
    def _get_cached_variants(self, cache_key: str) -> List[str]:
        """Check if query variants are cached."""
        if hasattr(self._generate_query_variants, 'cache') and cache_key in self._generate_query_variants.cache:
            return self._generate_query_variants.cache[cache_key]
        return None
        
    def _cache_variants(self, cache_key: str, variants: List[str]):
        """Cache query variants."""
        if hasattr(self._generate_query_variants, 'cache'):
            self._generate_query_variants.cache[cache_key] = variants
            
    def invalidate_cache(self):
        """Invalidate cache when vector store is updated."""
        self.vector_store_update_count += 1
        logger.info(f"Cache invalidated due to vector store update. Update count: {self.vector_store_update_count}")
        self._get_cached_embedding.cache_clear()
        if hasattr(self._generate_query_variants, 'cache'):
            self._generate_query_variants.cache.clear()
        self.cache_metrics.reset()
        self._initialize_bm25_index()
        self.cross_encoder = None  # Reset cross-encoder to reload on next use
        self._initialize_cross_encoder()

    def _initialize_bm25_index(self):
        """Initialize BM25 index from vector store chunks."""
        if not self.vector_store.index:
            logger.warning("Vector store not initialized, skipping BM25 index creation.")
            return
        
        self.chunk_texts = []
        self.chunk_ids = []
        for chunk in self.vector_store.get_all_chunks():
            self.chunk_texts.append(chunk.content.split())
            self.chunk_ids.append(hash(chunk.content))
        if self.chunk_texts:
            self.bm25_index = BM25Okapi(self.chunk_texts)
            logger.info(f"BM25 index initialized with {len(self.chunk_texts)} chunks.")
        else:
            logger.warning("No chunks available for BM25 index.")

    def _get_bm25_scores(self, query: str) -> Dict[int, float]:
        """Calculate BM25 scores for the query."""
        if self.bm25_index is None:
            logger.warning("BM25 index not initialized, returning empty scores.")
            return {}
        
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        return {chunk_id: float(score) for chunk_id, score in zip(self.chunk_ids, scores) if score > 0}

    def _apply_metadata_filter(self, results: List[RetrievalResult], metadata_filter: Dict) -> List[RetrievalResult]:
        """Apply metadata filters to the results."""
        if not metadata_filter:
            return results
        
        filtered_results = []
        for res in results:
            matches_filter = True
            for key, value in metadata_filter.items():
                if key in self.metadata_filters['fields']:
                    chunk_value = res.chunk.metadata.get(key)
                    if isinstance(value, list):
                        if chunk_value not in value:
                            matches_filter = False
                            break
                    else:
                        if chunk_value != value:
                            matches_filter = False
                            break
            if matches_filter:
                filtered_results.append(res)
        logger.info(f"Applied metadata filter, reduced results from {len(results)} to {len(filtered_results)}")
        return filtered_results

    def _initialize_cross_encoder(self):
        """Initialize or reinitialize the cross-encoder model for re-ranking with caching."""
        if self.reranking_config['cross_encoder']['enable']:
            start_time = datetime.datetime.now()
            try:
                from sentence_transformers import CrossEncoder

                model_name = self.reranking_config['cross_encoder']['model']
                self.cross_encoder = CrossEncoder(model_name)
                load_time = (datetime.datetime.now() - start_time).total_seconds()
                logger.info(f"Cross-encoder model {model_name} loaded in {load_time:.3f} seconds")
            except ImportError as e:
                logger.warning(f"Cross-encoder re-ranking disabled because sentence-transformers is unavailable: {e}")
                self.cross_encoder = None
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                self.cross_encoder = None
        else:
            logger.info("Cross-encoder re-ranking disabled in configuration")

    def _apply_reranking(self, query: str, results: List[RetrievalResult], top_k: int) -> List[RetrievalResult]:
        """
        Apply advanced re-ranking techniques to the initial retrieval results.
        Techniques include cross-encoder scoring, MMR diversity sampling, position decay for novelty,
        and source authority weighting.
        """
        if not results:
            return results

        start_time_total = datetime.datetime.now()
        reranked_results = results

        # Step 1: Cross-Encoder Scoring
        if self.reranking_config['cross_encoder']['enable'] and self.cross_encoder:
            start_time = datetime.datetime.now()
            batch_size = self.reranking_config['cross_encoder']['batch_size']
            scores = []
            for i in range(0, len(results), batch_size):
                batch_results = results[i:i + batch_size]
                batch_pairs = [(query, res.chunk.content) for res in batch_results]
                batch_scores = self.cross_encoder.predict(batch_pairs)
                scores.extend(batch_scores)
            
            # Update scores with cross-encoder results
            for res, score in zip(results, scores):
                res.score = float(score)
            
            reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
            cross_encoder_time = (datetime.datetime.now() - start_time).total_seconds()
            self.reranking_metrics['cross_encoder'] += 1
            if self.reranking_config['latency_monitoring']['enable']:
                logger.info(f"Cross-encoder re-ranking completed in {cross_encoder_time:.3f} seconds for {len(results)} results")

        # Step 2: MMR (Maximal Marginal Relevance) for Diversity
        if self.reranking_config['mmr']['enable']:
            start_time = datetime.datetime.now()
            lambda_param = self.reranking_config['mmr']['lambda_param']
            selected_results = []
            remaining_results = reranked_results.copy()
            
            # Get embeddings for diversity calculation
            result_embeddings = [self._get_cached_embedding(res.chunk.content) for res in reranked_results]
            query_embedding = self._get_cached_embedding(query)
            
            while remaining_results and len(selected_results) < top_k:
                best_score = -float('inf')
                best_result = None
                best_idx = -1
                
                for i, res in enumerate(remaining_results):
                    # Ensure relevance_score is a scalar
                    relevance_score = float(res.score) * float(lambda_param)
                    diversity_score = 0.0
                    if selected_results:
                        # Compute max_similarity ensuring scalar output
                        similarities = []
                        res_idx = reranked_results.index(res)
                        res_emb = self._normalize(result_embeddings[res_idx])
                        for sel in selected_results:
                            sel_idx = reranked_results.index(sel)
                            sel_emb = self._normalize(result_embeddings[sel_idx])
                            dot_product1 = np.dot(res_emb, sel_emb)
                            dot_product2 = np.dot(sel_emb, query_embedding)
                            # Ensure dot products are scalars
                            sim_val = float(dot_product1) * float(dot_product2) if isinstance(dot_product1, (np.ndarray, np.generic)) or isinstance(dot_product2, (np.ndarray, np.generic)) else dot_product1 * dot_product2
                            similarities.append(sim_val)
                        max_similarity = float(max(similarities)) if similarities else 0.0
                        diversity_score = float(1.0 - lambda_param) * float(1.0 - max_similarity)
                    # Double-check that score is a scalar
                    total_score = relevance_score + diversity_score
                    if isinstance(total_score, (np.ndarray, np.generic)):
                        score = float(total_score.item() if total_score.size == 1 else total_score[0])
                    else:
                        score = float(total_score)
                    
                    if score > best_score:
                        best_score = score
                        best_result = res
                        best_idx = i
                
                if best_result:
                    selected_results.append(best_result)
                    remaining_results.pop(best_idx)
            
            reranked_results = selected_results
            mmr_time = (datetime.datetime.now() - start_time).total_seconds()
            self.reranking_metrics['mmr'] += 1
            if self.reranking_config['latency_monitoring']['enable']:
                logger.info(f"MMR diversity re-ranking completed in {mmr_time:.3f} seconds for {len(reranked_results)} results")

        # Step 3: Position Decay for Novelty Detection
        if self.reranking_config['position_decay']['enable']:
            start_time = datetime.datetime.now()
            decay_rate = self.reranking_config['position_decay']['decay_rate']
            for i, res in enumerate(reranked_results):
                res.score *= (decay_rate ** i)
            reranked_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)
            position_decay_time = (datetime.datetime.now() - start_time).total_seconds()
            self.reranking_metrics['position_decay'] += 1
            if self.reranking_config['latency_monitoring']['enable']:
                logger.info(f"Position decay re-ranking completed in {position_decay_time:.3f} seconds for {len(reranked_results)} results")

        # Step 4: Source Authority Weighting
        if self.reranking_config['source_authority']['enable']:
            start_time = datetime.datetime.now()
            weights = self.reranking_config['source_authority']['weights']
            for res in reranked_results:
                authority_level = res.chunk.metadata.get('authority_level', 'default')
                weight = weights.get(authority_level, weights['default'])
                res.score *= weight
            reranked_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)
            source_authority_time = (datetime.datetime.now() - start_time).total_seconds()
            self.reranking_metrics['source_authority'] += 1
            if self.reranking_config['latency_monitoring']['enable']:
                logger.info(f"Source authority weighting completed in {source_authority_time:.3f} seconds for {len(reranked_results)} results")

        total_reranking_time = (datetime.datetime.now() - start_time_total).total_seconds()
        if self.reranking_config['latency_monitoring']['enable']:
            logger.info(f"Total re-ranking process completed in {total_reranking_time:.3f} seconds, final results: {len(reranked_results[:top_k])}")
            logger.info(f"Reranking metrics: {self.reranking_metrics}")

        return reranked_results[:top_k]

    def _normalize(self, vector: List[float]) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        return np.array(vector) / norm if norm > 0 else np.array(vector)
