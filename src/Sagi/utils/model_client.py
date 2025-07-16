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

    @classmethod
    def create_model_client(
        cls,
        client_config: Dict[str, Any],
        response_format: Optional[Type[T]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> OpenAIChatCompletionClient:
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
