from typing import Any, Dict, List, Union

import tiktoken
from autogen_core.memory import MemoryContent
from autogen_core.models import LLMMessage
from configs.functions import get_llm_provider
from pydantic import BaseModel
from transformers import AutoTokenizer


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
    message: List[Dict[str, Any]],
) -> int:
    """
    Count tokens for Anthropic Claude models.

    Args:
        message: The message to count tokens for

    Returns:
        Number of tokens
    """
    # TODO: Implement token counting for Anthropic models using their API
    # import anthropic

    # client = anthropic.Anthropic(api_key="", base_url="")
    # response = client.messages.count_tokens(
    #     model=model,
    #     messages=message
    # )
    # # ask the client as normal message
    # return response.input_tokens

    # Currently, we don't have a direct API to count tokens for Anthropic models.
    # Instead, we can estimate based on character count.

    # Based on empirical testing, Claude tokens are roughly:
    # - 1 token ≈ 3.5-4 characters for English text
    # - Similar to GPT models but slightly different tokenization
    total_chars = 0

    # Handle both single message and list of messages
    if isinstance(message, dict):
        messages = [message]
    else:
        messages = message

    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        # Count characters in content and role
        total_chars += len(str(content)) + len(str(role))
        # Add overhead for message structure (XML-like tags, etc.)
        total_chars += 20  # Estimated overhead per message

    # Anthropic tokenization estimation:
    # - English text: ~3.8 characters per token
    # - Add some buffer for special tokens and formatting
    estimated_tokens = int(total_chars / 3.8) + 10

    return estimated_tokens


def count_tokens_deepseek(text: str) -> int:
    """
    Count tokens for DeepSeek models.

    Args:
        text: The text to count tokens for

    Returns:
        Estimated number of tokens (using tiktoken cl100k_base as approximation)
    """
    # DeepSeek models are similar to GPT models, so we can use tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_tokens_local(content: str, model: str, provider: str) -> int:
    if provider[6:] == "qwen3-8b":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        return len(tokenizer.encode(content))
    else:
        raise ValueError(
            f"Unsupported local provider: {provider[6:]} of model: {model}"
        )


def count_tokens_messages(
    messages: Union[List[Dict[str, Any]], List[LLMMessage], List[MemoryContent]],
    model: str,
) -> int:
    """
    Count tokens for a list of chat messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: The model name

    Returns:
        Total number of tokens including message formatting overhead
    """
    total_tokens = 0
    provider = get_llm_provider(model)
    assert provider is not None, "Provider is not set"

    if provider.lower() == "anthropic":
        # For Anthropic, we use their API to count tokens
        return count_tokens_anthropic(messages)

    for message in messages:
        if isinstance(message, BaseModel):
            message = message.model_dump(mode="json")

        content = message.get("content", "")

        # Count tokens for content
        if provider.lower() == "openai":
            content_tokens = count_tokens_openai(content, model)
        elif provider.lower() == "deepseek":
            content_tokens = count_tokens_deepseek(content)
        elif provider.lower().startswith("local"):
            content_tokens = count_tokens_local(content, model, provider.lower())
        elif provider.lower() == "google":
            content_tokens = count_tokens_openai(content, model)
        else:
            raise ValueError(f"Unsupported provider: {provider} of model: {model}")

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
