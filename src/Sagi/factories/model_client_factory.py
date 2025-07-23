from typing import Any, Dict, Optional, Type, TypeVar, Union

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
ModelClient = Union[OpenAIChatCompletionClient, AnthropicChatCompletionClient]


class ModelClientFactory:
    """
    Factory for creating OpenAI and Anthropic Chat Completion Clients with simplified validation.

    This factory implements the creation logic for Model Clients with basic validation
    and configuration processing, removing over-validation as per technical report requirements.
    """

    @staticmethod
    def _init_model_info(
        client_config: Dict[str, Any], provider: str = "openai"
    ) -> Optional[ModelInfo]:
        """
        Initialize ModelInfo from client configuration.

        Args:
            client_config: Dictionary containing client configuration
            provider: The model provider ("openai" or "anthropic")

        Returns:
            Optional[ModelInfo]: ModelInfo object if configuration contains model_info, None otherwise
        """
        if "model_info" in client_config:
            model_info = client_config["model_info"].copy()
            # Set appropriate model family based on provider
            if provider == "anthropic":
                model_info["family"] = ModelFamily.ANTHROPIC
            else:
                model_info["family"] = ModelFamily.OPENAI
            return ModelInfo(**model_info)
        return None

    @staticmethod
    def _determine_provider(client_config: Dict[str, Any]) -> str:
        """
        Determine the provider based on client configuration.

        Args:
            client_config: Dictionary containing client configuration

        Returns:
            str: Provider name ("openai" or "anthropic")
        """
        # Check for explicit provider specification
        if "provider" in client_config:
            return client_config["provider"].lower()

        # Infer from base_url or model name
        base_url = client_config.get("base_url", "").lower()
        model = client_config.get("model", "").lower()

        if "anthropic" in base_url or "claude" in model:
            return "anthropic"
        return "openai"

    @classmethod
    def create_model_client(
        cls,
        client_config: Dict[str, Any],
        response_format: Optional[Type[T]] = None,
        parallel_tool_calls: Optional[bool] = None,
    ) -> ModelClient:
        """
        Create an OpenAI or Anthropic Chat Completion Client with simplified validation.

        Args:
            client_config: Dictionary containing client configuration
            response_format: Optional response format for structured output
            parallel_tool_calls: Whether to enable parallel tool calls

        Returns:
            ModelClient: Configured client instance (OpenAI or Anthropic)

        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Determine provider
        provider = cls._determine_provider(client_config)

        # Basic configuration validation (removing over-validation)
        required_fields = ["model", "max_tokens"]
        if provider == "openai":
            required_fields.extend(["base_url", "api_key"])
        elif provider == "anthropic":
            # Anthropic can use either api_key or auth_token
            if "api_key" not in client_config and "auth_token" not in client_config:
                raise ValueError("Missing required field: api_key or auth_token")

        for field in required_fields:
            if field not in client_config or not client_config[field]:
                raise ValueError(f"Missing required field: {field}")

        # Create client configuration
        model_info = cls._init_model_info(client_config, provider)
        client_kwargs = {
            "model": client_config["model"],
            "model_info": model_info,
            "max_tokens": client_config["max_tokens"],
        }

        # Add authentication key (different field names for different providers)
        if provider == "openai":
            client_kwargs["api_key"] = client_config["api_key"]
        elif provider == "anthropic":
            # Anthropic supports both api_key and auth_token
            if "api_key" in client_config:
                client_kwargs["api_key"] = client_config["api_key"]
            elif "auth_token" in client_config:
                client_kwargs["api_key"] = client_config["auth_token"]

        # Add provider-specific configuration
        if provider == "openai":
            client_kwargs["base_url"] = client_config["base_url"]

            # Handle optional parameters for OpenAI
            if response_format:
                client_kwargs["response_format"] = response_format

            if parallel_tool_calls is not None:
                client_kwargs["parallel_tool_calls"] = parallel_tool_calls

            if "default_headers" in client_config:
                client_kwargs["default_headers"] = client_config["default_headers"]

            return OpenAIChatCompletionClient(**client_kwargs)

        elif provider == "anthropic":
            # Handle optional parameters for Anthropic
            if response_format:
                client_kwargs["response_format"] = response_format

            if "default_headers" in client_config:
                client_kwargs["default_headers"] = client_config["default_headers"]

            return AnthropicChatCompletionClient(**client_kwargs)

        else:
            raise ValueError(f"Unsupported provider: {provider}")
