from typing import Any, Dict, Optional, Type, TypeVar

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ModelClientFactory:
    """
    Factory for creating OpenAI Chat Completion Clients with simplified validation.
    
    This factory implements the creation logic for Model Clients with basic validation
    and configuration processing, removing over-validation as per technical report requirements.
    """
    
    @staticmethod
    def _init_model_info(client_config: Dict[str, Any]) -> Optional[ModelInfo]:
        """
        Initialize ModelInfo from client configuration.
        
        Args:
            client_config: Dictionary containing client configuration
            
        Returns:
            Optional[ModelInfo]: ModelInfo object if configuration contains model_info, None otherwise
        """
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
        """
        Create an OpenAI Chat Completion Client with simplified validation.
        
        Args:
            client_config: Dictionary containing client configuration
            response_format: Optional response format for structured output
            parallel_tool_calls: Whether to enable parallel tool calls
            
        Returns:
            OpenAIChatCompletionClient: Configured client instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Basic configuration validation (removing over-validation)
        required_fields = ["model", "base_url", "api_key", "max_tokens"]
        for field in required_fields:
            if field not in client_config or not client_config[field]:
                raise ValueError(f"Missing required field: {field}")
        
        # Create client configuration
        model_info = cls._init_model_info(client_config)
        client_kwargs = {
            "model": client_config["model"],
            "base_url": client_config["base_url"],
            "api_key": client_config["api_key"],
            "model_info": model_info,
            "max_tokens": client_config["max_tokens"],
        }

        # Handle optional parameters
        if response_format:
            client_kwargs["response_format"] = response_format

        if parallel_tool_calls is not None:
            client_kwargs["parallel_tool_calls"] = parallel_tool_calls

        if "default_headers" in client_config:
            client_kwargs["default_headers"] = client_config["default_headers"]

        return OpenAIChatCompletionClient(**client_kwargs)