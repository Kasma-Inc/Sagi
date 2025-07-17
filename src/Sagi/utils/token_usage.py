from typing import Any, Dict, List, Optional, Union

import tiktoken
from autogen_core.memory import MemoryContent
from autogen_core.models import LLMMessage
from pydantic import BaseModel

from Sagi.utils.model_info import get_model_info


def count_tokens_openai(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens for OpenAI models using tiktoken.

    Args:
        text: The text to count tokens for
        model: The OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to cl100k_base encoding if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def count_tokens_anthropic(
    message: dict[str, Any],
    model: str = "claude-3-sonnet-20240229",
    api_config: Optional[dict[str, Any]] = None,
) -> int:
    """
    Count tokens for Anthropic Claude models.

    Args:
        message: The message to count tokens for
        model: The Anthropic model name
        api_config: The API config for Anthropic

    Returns:
        Estimated number of tokens (approximation: 1 token ≈ 4 characters)
    """
    import anthropic

    if api_config is None:
        api_config = {}
    client = anthropic.Anthropic(**api_config)
    response = client.messages.count_tokens(
        model=model,
        messages=[message],
    )
    return response.input_tokens


def count_tokens_deepseek(text: str, model: str = "deepseek-chat") -> int:
    """
    Count tokens for DeepSeek models.

    Args:
        text: The text to count tokens for
        model: The DeepSeek model name

    Returns:
        Estimated number of tokens (using tiktoken cl100k_base as approximation)
    """
    # DeepSeek models are similar to GPT models, so we can use tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_tokens_messages(
    messages: Union[List[Dict[str, Any]], List[LLMMessage], List[MemoryContent]],
    model: str,
    api_config: Optional[dict[str, Any]] = None,
) -> int:
    """
    Count tokens for a list of chat messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name
        api_config: The API config for the model provider

    Returns:
        Total number of tokens including message formatting overhead
    """
    total_tokens = 0

    for message in messages:
        if isinstance(message, BaseModel):
            message = message.model_dump(mode="json")

        content = message.get("content", "")
        provider = get_model_info(model).get("provider")
        assert provider is not None, "Provider is not set"

        # Count tokens for content
        if provider.lower() == "openai":
            content_tokens = count_tokens_openai(content, model)
        elif provider.lower() == "anthropic":
            content_tokens = count_tokens_anthropic(message, model, api_config)
        elif provider.lower() == "deepseek":
            content_tokens = count_tokens_deepseek(content, model)
        else:
            print(f"Unsupported provider: {provider} of model: {model}")

        total_tokens += content_tokens

        # Add overhead for message formatting (role, separators, etc.)
        # This is an approximation based on OpenAI's token counting
        if provider.lower() == "openai":
            total_tokens += 4  # Overhead per message for OpenAI
        else:
            total_tokens += 2  # Conservative estimate for other providers

    # Add conversation-level overhead
    if provider.lower() == "openai":
        total_tokens += 2  # Conversation overhead for OpenAI

    return total_tokens
