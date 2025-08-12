from typing import Any, Dict, Optional, Type, TypeVar

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelClientFactory:
    @staticmethod
    def _init_model_info(client_config: Dict[str, Any]) -> Optional[ModelInfo]:
        if "model_info" in client_config:
            model_info = client_config["model_info"]
            model_info["family"] = ModelFamily.UNKNOWN
            return ModelInfo(**model_info)
        return None

    @staticmethod
    def _apply_provider_specific_config(client_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply provider-specific configurations based on base_url."""
        config = client_config.copy()
        base_url = config.get("base_url", "")
        
        # yunwu.ai specific configurations
        if "yunwu.ai" in base_url:
            # Ensure stream_options for token usage tracking
            if "stream_options" not in config:
                config["stream_options"] = {"include_usage": True}
        
        return config

    @classmethod
    def create_model_client(
        cls,
        client_config: Dict[str, Any],
        response_format: Optional[Type[T]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> OpenAIChatCompletionClient:
        # Apply provider-specific configurations
        client_config = cls._apply_provider_specific_config(client_config)
        
        model_info = cls._init_model_info(client_config)
        client_kwargs = {
            "model": client_config["model"],
            "base_url": client_config["base_url"],
            "api_key": client_config["api_key"],
            "model_info": model_info,
            "max_tokens": client_config.get("max_tokens", 16000),
        }

        if response_format:
            client_kwargs["response_format"] = response_format

        if parallel_tool_calls is not None:
            client_kwargs["parallel_tool_calls"] = parallel_tool_calls

        # Add the remaining client kwargs from the client_config
        for key, value in client_config.items():
            if key not in client_kwargs:
                client_kwargs[key] = value

        return OpenAIChatCompletionClient(**client_kwargs)
