import yaml
import logging

logger = logging.getLogger(__name__)

class Config:
    """System configuration, loaded from your script."""
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        
    def _load_config(self, path: str) -> dict:
        default_config = {
            'system': {
                'chunk_size': 256, 'overlap': 64, 'top_k': 5
            },
            'uploads': {
                'max_file_size_mb': 250
            },
            'indexing': {
                'batch_size': 16,
                'max_workers': 2
            },
            'embedding': {
                'model': 'nomic-embed-text', 'dimension': 768
            },
            'llm': {
                'provider': 'deepseek',
                'model': 'deepseek-v3',
                'temperature': 0.2,
                'max_tokens': 4096,
                'api_key_env': 'DEEPSEEK_API_KEY',
                'base_url': 'https://api.deepseek.com/v1',
            },
            'vector_store': {
                'backend': 'qdrant',
                'persist_path': './vector_db',
                'qdrant': {
                    'path': './vector_db/qdrant',
                    'collection_name': 'ragagument',
                    'url': '',
                    'api_key_env': 'QDRANT_API_KEY',
                    'prefer_grpc': False,
                    'timeout': 30,
                },
            },
            'advanced': {
                'enable_fusion': True, 'enable_rewriting': True, 'enable_tools': True, 'fusion_queries': 3
            }
        }
        
        try:
            with open(path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Deep merge user config into defaults
                for key, value in user_config.items():
                    if isinstance(value, dict) and isinstance(default_config.get(key), dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                return default_config
        except FileNotFoundError:
            logger.warning(f"Config file {path} not found, using defaults.")
            return default_config
    
    def get(self, key_path: str, default=None):
        """Get nested config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default
