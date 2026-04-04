import hashlib
import logging
import re
from typing import Any, Dict, List
import datetime
from functools import lru_cache
from threading import Lock
from vector_store import RetrievalResult, VectorStore
from embedder import OllamaEmbedder
from llm_factory import LLMFactory
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
    def __init__(self, vector_store: VectorStore, embedder: OllamaEmbedder, llm_model: str, config_path: str = "config.yaml"):
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
        
        self.llm_settings = LLMFactory.from_mapping(config.get('llm', {}))
        self.llm_client = None
        self.bm25_index = None
        self.chunk_texts = []
        self.chunk_keys = []
        self.chunk_lookup: Dict[str, Any] = {}
        self.query_variant_cache: Dict[str, List[str]] = {}
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
        prepared_filter = self._prepare_metadata_filter(metadata_filter)
        candidate_count = self._candidate_pool_size(top_k, prepared_filter)
        results = self.vector_store.search(embedding, candidate_count, metadata_filter=prepared_filter)
        if prepared_filter:
            results = self._apply_metadata_filter(results, prepared_filter)
        return results[:top_k]

    def _retrieve_hybrid(self, query: str, top_k: int, metadata_filter: Dict = None) -> List[RetrievalResult]:
        """Hybrid retrieval combining vector similarity and BM25 keyword matching."""
        start_time = datetime.datetime.now()
        # Vector similarity search
        embedding = self._get_cached_embedding(query)
        prepared_filter = self._prepare_metadata_filter(metadata_filter)
        candidate_count = self._candidate_pool_size(top_k, prepared_filter)
        vector_results = self.vector_store.search(embedding, candidate_count, metadata_filter=prepared_filter)
        vector_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"Vector search completed in {vector_time:.3f} seconds, retrieved {len(vector_results)} results")
        
        # BM25 keyword search
        start_time = datetime.datetime.now()
        bm25_scores = self._get_bm25_scores(query)
        bm25_results = [
            RetrievalResult(chunk=self.chunk_lookup[chunk_key], score=score)
            for chunk_key, score in bm25_scores.items()
            if chunk_key in self.chunk_lookup
        ]
        bm25_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(f"BM25 search completed in {bm25_time:.3f} seconds, retrieved {len(bm25_results)} results")
        
        # Combine scores
        start_time = datetime.datetime.now()
        if prepared_filter:
            vector_results = self._apply_metadata_filter(vector_results, prepared_filter)
            bm25_results = self._apply_metadata_filter(bm25_results, prepared_filter)

        normalized_vector_scores = self._normalize_scores(
            {self._chunk_key(vr.chunk): vr.score for vr in vector_results}
        )
        normalized_bm25_scores = self._normalize_scores(
            {self._chunk_key(br.chunk): br.score for br in bm25_results}
        )

        combined_results = defaultdict(float)
        for vr in vector_results:
            chunk_key = self._chunk_key(vr.chunk)
            combined_results[chunk_key] += normalized_vector_scores.get(chunk_key, 0.0) * self.hybrid_weights['vector']
        for br in bm25_results:
            chunk_key = self._chunk_key(br.chunk)
            combined_results[chunk_key] += normalized_bm25_scores.get(chunk_key, 0.0) * self.hybrid_weights['bm25']
        
        # Convert to list and sort
        final_results = [
            RetrievalResult(chunk=self.chunk_lookup[chunk_key], score=score)
            for chunk_key, score in combined_results.items()
            if chunk_key in self.chunk_lookup
        ]
        final_results.sort(key=lambda x: x.score, reverse=True)
        
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
                prepared_filter = self._prepare_metadata_filter(metadata_filter)
                candidate_count = self._candidate_pool_size(top_k, prepared_filter)
                results = self.vector_store.search(embedding, candidate_count, metadata_filter=prepared_filter)
                if prepared_filter:
                    results = self._apply_metadata_filter(results, prepared_filter)
            for res in results:
                chunk_key = self._chunk_key(res.chunk)
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
        if not self.query_expansion_config.get('enable', True) or num_variants <= 1:
            return [query]

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
        
        # Synonym expansion intentionally disabled until a real implementation exists.
        if self.query_expansion_config.get('techniques', {}).get('synonym_expansion', {}).get('enable', True):
            synonym_variants = self._generate_synonym_variants(query, max_variants - len(variants))
            variants.extend(synonym_variants)
            if synonym_variants:
                self.query_expansion_metrics['synonym_expansion'] += 1
        
        # Entity expansion intentionally disabled until a real implementation exists.
        if self.query_expansion_config.get('techniques', {}).get('entity_recognition', {}).get('enable', True):
            entity_variants = self._generate_entity_variants(query, max_variants - len(variants))
            variants.extend(entity_variants)
            if entity_variants:
                self.query_expansion_metrics['entity_recognition'] += 1
        
        # Contextual broadening intentionally disabled until a real implementation exists.
        if self.query_expansion_config.get('techniques', {}).get('contextual_broadening', {}).get('enable', True):
            contextual_variants = self._generate_contextual_variants(query, max_variants - len(variants))
            variants.extend(contextual_variants)
            if contextual_variants:
                self.query_expansion_metrics['contextual_broadening'] += 1
        
        result = self._dedupe_preserving_order(variants)[:max_variants]
        self._cache_variants(cache_key, result)
        if self.query_expansion_config.get('metrics', {}).get('enable', True):
            logger.info(f"Query expansion metrics: {self.query_expansion_metrics}")
        return result

    def _generate_llm_variants(self, query: str, num_variants: int) -> List[str]:
        """Generate query variants using the configured OpenAI-compatible provider."""
        prompt = f"""You are an expert query writer. Rewrite the following user query in {num_variants} different ways to improve document retrieval from a vector database. Focus on using synonyms, rephrasing, and asking related sub-questions.

Original Query: "{query}"

Return ONLY the rewritten queries, each on a new line. Do not include numbering, bullets, or the original query."""
        
        variants = []
        if num_variants <= 0:
            return variants

        try:
            client = self._get_llm_client()
            if client is None:
                return variants

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            generated_queries = response.choices[0].message.content.strip().split('\n')
            variants.extend([q.strip() for q in generated_queries if q.strip()])
        except Exception as e:
            logger.warning("Failed to generate query variants with %s: %s", self.llm_settings.display_name, e)
        return variants[:num_variants]

    def _get_llm_client(self):
        if self.llm_client is not None:
            return self.llm_client

        validation = LLMFactory.validate_connection(self.llm_settings)
        if not validation["valid"]:
            logger.warning(
                "LLM-backed query expansion disabled because provider validation failed: %s",
                validation["message"],
            )
            return None

        self.llm_client = LLMFactory.create_client(self.llm_settings)
        return self.llm_client

    def _generate_synonym_variants(self, query: str, num_variants: int) -> List[str]:
        """Placeholder kept explicit so retrieval is not polluted by fake variants."""
        return []

    def _generate_entity_variants(self, query: str, num_variants: int) -> List[str]:
        """Placeholder kept explicit so retrieval is not polluted by fake variants."""
        return []

    def _generate_contextual_variants(self, query: str, num_variants: int) -> List[str]:
        """Placeholder kept explicit so retrieval is not polluted by fake variants."""
        return []
        
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Cache embeddings for queries and variants."""
        self.cache_metrics.record_miss()
        return self.embedder.embed_text(text)
        
    def _get_cached_variants(self, cache_key: str) -> List[str]:
        """Check if query variants are cached."""
        return self.query_variant_cache.get(cache_key)
        
    def _cache_variants(self, cache_key: str, variants: List[str]):
        """Cache query variants."""
        self.query_variant_cache[cache_key] = variants
            
    def invalidate_cache(self):
        """Invalidate cache when vector store is updated."""
        self.vector_store_update_count += 1
        logger.info(f"Cache invalidated due to vector store update. Update count: {self.vector_store_update_count}")
        self._get_cached_embedding.cache_clear()
        self.query_variant_cache.clear()
        self.cache_metrics.reset()
        self._initialize_bm25_index()
        self.cross_encoder = None  # Reset cross-encoder to reload on next use
        self._initialize_cross_encoder()

    def _initialize_bm25_index(self):
        """Initialize BM25 index from vector store chunks."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            logger.warning("No chunks available for BM25 index.")
            self.bm25_index = None
            self.chunk_texts = []
            self.chunk_keys = []
            self.chunk_lookup = {}
            return
        
        self.chunk_texts = []
        self.chunk_keys = []
        self.chunk_lookup = {}
        for chunk in all_chunks:
            chunk_key = self._chunk_key(chunk)
            self.chunk_texts.append(self._tokenize_text(chunk.content))
            self.chunk_keys.append(chunk_key)
            self.chunk_lookup[chunk_key] = chunk
        if self.chunk_texts:
            self.bm25_index = BM25Okapi(self.chunk_texts)
            logger.info(f"BM25 index initialized with {len(self.chunk_texts)} chunks.")

    def _get_bm25_scores(self, query: str) -> Dict[int, float]:
        """Calculate BM25 scores for the query."""
        if self.bm25_index is None:
            logger.warning("BM25 index not initialized, returning empty scores.")
            return {}
        
        tokenized_query = self._tokenize_text(query)
        scores = self.bm25_index.get_scores(tokenized_query)
        return {chunk_key: float(score) for chunk_key, score in zip(self.chunk_keys, scores) if score > 0}

    def _candidate_pool_size(self, requested_top_k: int, metadata_filter: Dict = None) -> int:
        total_chunks = len(self.vector_store.get_all_chunks())
        if total_chunks == 0:
            return 0

        if metadata_filter and self.metadata_filters.get('enable'):
            if total_chunks <= 5000:
                return total_chunks
            return min(total_chunks, max(requested_top_k * 25, 500))

        return min(total_chunks, max(requested_top_k * 2, 20))

    def _chunk_key(self, chunk) -> str:
        metadata = chunk.metadata or {}
        content_digest = hashlib.sha1(chunk.content.encode('utf-8', errors='ignore')).hexdigest()[:12]
        return "|".join(
            [
                str(metadata.get('document_id', metadata.get('eval_document_id', 'unknown-doc'))),
                str(metadata.get('source', metadata.get('filename', 'unknown-source'))),
                str(metadata.get('chunk_index', 'na')),
                content_digest,
            ]
        )

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}

        values = list(scores.values())
        max_score = max(values)
        min_score = min(values)
        if max_score == min_score:
            return {key: 1.0 for key in scores}

        spread = max_score - min_score
        return {key: (value - min_score) / spread for key, value in scores.items()}

    def _tokenize_text(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text.lower())

    def _dedupe_preserving_order(self, values: List[str]) -> List[str]:
        seen = set()
        deduped = []
        for value in values:
            normalized = value.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(normalized)
        return deduped

    def _apply_metadata_filter(self, results: List[RetrievalResult], metadata_filter: Dict) -> List[RetrievalResult]:
        """Apply metadata filters to the results."""
        prepared_filter = self._prepare_metadata_filter(metadata_filter)
        if not prepared_filter:
            return results
        
        filtered_results = []
        for res in results:
            if self._matches_metadata_filter(res.chunk.metadata, prepared_filter):
                filtered_results.append(res)
        logger.info(f"Applied metadata filter, reduced results from {len(results)} to {len(filtered_results)}")
        return filtered_results

    def _prepare_metadata_filter(self, metadata_filter: Dict | None) -> Dict | None:
        if not metadata_filter or not self.metadata_filters.get('enable'):
            return None

        allowed_fields = set(self.metadata_filters.get('fields', []))
        prepared = {
            key: value
            for key, value in metadata_filter.items()
            if key in allowed_fields
        }
        return prepared or None

    def _matches_metadata_filter(self, chunk_metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
        for key, condition in metadata_filter.items():
            if not self._matches_filter_condition(chunk_metadata.get(key), condition):
                return False
        return True

    def _matches_filter_condition(self, chunk_value: Any, condition: Any) -> bool:
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if not self._matches_filter_operator(chunk_value, str(operator).strip().lower(), expected):
                    return False
            return True

        if isinstance(condition, list):
            return self._matches_in_condition(chunk_value, condition)

        return self._values_equal(chunk_value, condition)

    def _matches_filter_operator(self, chunk_value: Any, operator: str, expected: Any) -> bool:
        if operator == 'eq':
            return self._values_equal(chunk_value, expected)
        if operator == 'neq':
            return not self._values_equal(chunk_value, expected)
        if operator == 'in':
            return self._matches_in_condition(chunk_value, expected if isinstance(expected, list) else [expected])
        if operator == 'not_in':
            return not self._matches_in_condition(chunk_value, expected if isinstance(expected, list) else [expected])
        if operator == 'contains':
            return self._contains_value(chunk_value, expected)
        if operator == 'contains_any':
            values = expected if isinstance(expected, list) else [expected]
            return any(self._contains_value(chunk_value, value) for value in values)
        if operator == 'exists':
            return (chunk_value is not None) if bool(expected) else (chunk_value is None)
        if operator in {'gt', 'gte', 'lt', 'lte'}:
            return self._compare_numeric(chunk_value, expected, operator)
        if operator == 'between' and isinstance(expected, (list, tuple)) and len(expected) == 2:
            lower, upper = expected
            return self._compare_numeric(chunk_value, lower, 'gte') and self._compare_numeric(chunk_value, upper, 'lte')
        return False

    def _matches_in_condition(self, chunk_value: Any, expected_values: List[Any]) -> bool:
        if isinstance(chunk_value, (list, tuple, set)):
            return any(self._values_equal(item, expected) for item in chunk_value for expected in expected_values)
        return any(self._values_equal(chunk_value, expected) for expected in expected_values)

    def _contains_value(self, chunk_value: Any, expected: Any) -> bool:
        if isinstance(chunk_value, str) and isinstance(expected, str):
            return expected.lower() in chunk_value.lower()
        if isinstance(chunk_value, dict):
            return str(expected) in chunk_value
        if isinstance(chunk_value, (list, tuple, set)):
            return any(self._values_equal(item, expected) for item in chunk_value)
        return False

    def _compare_numeric(self, chunk_value: Any, expected: Any, operator: str) -> bool:
        chunk_number = self._coerce_numeric(chunk_value)
        expected_number = self._coerce_numeric(expected)
        if chunk_number is None or expected_number is None:
            return False
        if operator == 'gt':
            return chunk_number > expected_number
        if operator == 'gte':
            return chunk_number >= expected_number
        if operator == 'lt':
            return chunk_number < expected_number
        if operator == 'lte':
            return chunk_number <= expected_number
        return False

    def _coerce_numeric(self, value: Any) -> float | None:
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

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, (list, tuple, set)):
            return any(self._values_equal(item, right) for item in left)
        if isinstance(right, (list, tuple, set)):
            return any(self._values_equal(left, item) for item in right)
        return left == right

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
