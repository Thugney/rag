import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from openai import OpenAI


logger = logging.getLogger(__name__)


_PROVIDER_ALIASES = {
    "openai-compatible": "openai_compatible",
}

_PROVIDER_DEFAULTS = {
    "deepseek": {
        "display_name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
    },
    "openai": {
        "display_name": "OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "openai_compatible": {
        "display_name": "OpenAI-compatible provider",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
}


@dataclass(frozen=True)
class LLMProviderSettings:
    provider: str
    display_name: str
    model_name: str
    temperature: float
    max_tokens: int
    api_key_env: str
    base_url: Optional[str]

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


class LLMFactory:
    @staticmethod
    def from_config(config: Any) -> LLMProviderSettings:
        llm_config = {
            "provider": config.get("llm.provider", "deepseek"),
            "model": config.get("llm.model", "deepseek-chat"),
            "temperature": config.get("llm.temperature", 0.2),
            "max_tokens": config.get("llm.max_tokens", 4096),
            "api_key_env": config.get("llm.api_key_env"),
            "base_url": config.get("llm.base_url"),
        }
        return LLMFactory.from_mapping(llm_config)

    @staticmethod
    def from_mapping(mapping: Mapping[str, Any]) -> LLMProviderSettings:
        raw_provider = str(mapping.get("provider", "deepseek")).strip().lower()
        provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
        provider_defaults = _PROVIDER_DEFAULTS.get(provider)
        if provider_defaults is None:
            raise ValueError(f"Unknown llm.provider: {raw_provider}")

        api_key_env = str(mapping.get("api_key_env") or provider_defaults["api_key_env"]).strip()
        configured_base_url = str(mapping.get("base_url") or "").strip()
        base_url = configured_base_url or provider_defaults["base_url"]

        return LLMProviderSettings(
            provider=provider,
            display_name=str(provider_defaults["display_name"]),
            model_name=str(mapping.get("model", "deepseek-chat")),
            temperature=float(mapping.get("temperature", 0.2)),
            max_tokens=int(mapping.get("max_tokens", 4096)),
            api_key_env=api_key_env,
            base_url=base_url,
        )

    @staticmethod
    def create_client(settings: LLMProviderSettings) -> OpenAI:
        if not settings.configured:
            raise ValueError(f"{settings.api_key_env} is not configured for {settings.display_name}.")

        client_args = {"api_key": settings.api_key}
        if settings.base_url:
            client_args["base_url"] = settings.base_url
        return OpenAI(**client_args)

    @staticmethod
    def validate_connection(settings: LLMProviderSettings) -> dict[str, Any]:
        if not settings.configured:
            return {
                "provider": settings.provider,
                "configured": False,
                "valid": False,
                "message": f"{settings.api_key_env} is not configured.",
            }

        try:
            client = LLMFactory.create_client(settings)
            client.models.list()
            return {
                "provider": settings.provider,
                "configured": True,
                "valid": True,
                "message": f"{settings.display_name} credentials validated successfully.",
            }
        except Exception as exc:
            message = str(exc)
            if "authentication" in message.lower() or "invalid" in message.lower():
                message = f"{settings.display_name} authentication failed. Check {settings.api_key_env}."
            logger.warning("LLM validation failed for provider %s: %s", settings.provider, exc)
            return {
                "provider": settings.provider,
                "configured": True,
                "valid": False,
                "message": message,
            }
