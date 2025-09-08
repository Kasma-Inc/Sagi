from typing import Any, Dict


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get model information based on model name.

    Args:
        model_name: The name of the model

    Returns:
        Dictionary containing model information including provider, max_tokens, etc.
    """
    model_info = {
        # OpenAI models
        # https://platform.openai.com/docs/models/gpt-4o
        "gpt-4o": {
            "provider": "openai",
            "max_tokens": 16384,
            "context_window": 128000,
            "input": ["text", "image"],
            "output": ["text"],
            "stream": True,
        },
        # https://platform.openai.com/docs/models/gpt-4o-mini
        "gpt-4o-mini": {
            "provider": "openai",
            "max_tokens": 16384,
            "context_window": 128000,
            "input": ["text", "image"],
            "stream": True,
        },
        # https://platform.openai.com/docs/models/gpt-4.1
        "gpt-4.1": {
            "provider": "openai",
            "max_tokens": 32768,
            "context_window": 1047576,
            "input": ["text", "image"],
            "output": ["text"],
            "stream": True,
        },
        # Anthropic models
        # https://docs.anthropic.com/en/docs/about-claude/models/overview#model-comparison-table
        "claude-opus-4": {
            "provider": "anthropic",
            "max_tokens": 32000,
            "context_window": 200000,
            "input": ["text", "image"],
            "output": ["text"],
            "stream": True,
        },
        "claude-sonnet-4": {
            "provider": "anthropic",
            "max_tokens": 64000,
            "context_window": 200000,
            "input": ["text", "image"],
            "output": ["text"],
            "stream": True,
        },
        # DeepSeek models
        "deepseek-r1": {
            "provider": "deepseek",
            "max_tokens": 32768,
            "context_window": 128000,
            "type": "multimodal",
            "stream": True,
        },
        # Grok models
        "grok-4": {
            "provider": "xai",
            "max_tokens": 4096,
            "context_window": 256000,
            "input": ["text"],
            "output": ["text"],
            "stream": True,
        },
        # Qwen models
        "qwen-turbo": {
            "provider": "qwen",
            "max_tokens": 32768,
            "context_window": 1000000,
            "input": ["text", "image"],
            "output": ["text"],
            "stream": True,
        },
        "qwen-plus": {
            "provider": "qwen",
            "max_tokens": 32768,
            "context_window": 100000,
            "type": "multimodal",
            "stream": True,
        },
        "qwen-max": {
            "provider": "qwen",
            "max_tokens": 32768,
            "context_window": 100000,
            "type": "multimodal",
            "stream": True,
        },
    }

    result = model_info.get(model_name)
    if result is None:
        raise ValueError(f"Model {model_name} not found")
    return result


def get_model_name_by_api_provider(api_provider: str, model_name: str) -> str:
    """
    Get the model name by API provider and model name.

    Args:
        api_provider: The API provider
        model_name: The name of the model

    Returns:
        The model name
    """
    aiml_model_name = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4.1": "openai/gpt-4.1-2025-04-14",
        "claude-opus-4": "anthropic/claude-opus-4",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "deepseek-r1": "deepseek/deepseek-r1",
        "grok-4": "x-ai/grok-4-07-09",
        "qwen-turbo": "qwen-turbo",
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max",
    }
    yunwu_model_name = {
        "gpt-4o": "chatgpt-4o-latest",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4.1": "gpt-4.1-2025-04-14",
        "claude-opus-4": "claude-opus-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "deepseek-r1": "deepseek-r1",
        "grok-4": "x-ai/grok-4-07-09",
        "qwen-turbo": "qwen-turbo",
        "qwen-plus": "qwen-plus",
        "qwen-max": "qwen-max",
    }
    match api_provider:
        case "yunwu":
            model_name_api = yunwu_model_name.get(model_name)
            if model_name_api is None:
                raise ValueError(f"Model {model_name} not found in yunwu model name")
        case "aiml":
            model_name_api = aiml_model_name.get(model_name)
            if model_name_api is None:
                raise ValueError(f"Model {model_name} not found in aiml model name")
        case _:
            raise ValueError(f"API provider {api_provider} not supported")
    return model_name_api


def get_model_provider(model_name: str) -> str:
    """
    Get the provider for a given model name.

    Args:
        model_name: The name of the model

    Returns:
        The provider name (e.g., "openai", "anthropic", "deepseek")
    """
    return get_model_info(model_name).get("provider", "unknown")


def get_model_max_tokens(model_name: str) -> int:
    """
    Get the maximum output tokens for a given model.

    Args:
        model_name: The name of the model

    Returns:
        The maximum number of output tokens
    """
    return get_model_info(model_name).get("max_tokens", 4096)


def get_model_context_window(model_name: str) -> int:
    """
    Get the context window size for a given model.

    Args:
        model_name: The name of the model

    Returns:
        The context window size in tokens
    """
    return get_model_info(model_name).get("context_window", 8192)
