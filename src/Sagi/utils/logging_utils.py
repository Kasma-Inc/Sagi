import json
import logging
from datetime import datetime

from autogen_agentchat import TRACE_LOGGER_NAME
from autogen_core.logging import LLMCallEvent, LLMStreamEndEvent, LLMStreamStartEvent


class LLMFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, "msg") and isinstance(
            record.msg, (LLMStreamStartEvent, LLMStreamEndEvent, LLMCallEvent)
        ):
            return True
        return False


class ReadableFormatter(logging.Formatter):
    def format(self, record):
        # Get the base format
        base_msg = super().format(record)

        try:
            # Extract JSON part from the message
            if isinstance(record.msg, LLMStreamStartEvent):
                formatted = self._format_llm_start(record.msg.kwargs)
            elif isinstance(record.msg, LLMStreamEndEvent):
                formatted = self._format_llm_end(record.msg.kwargs)
            elif isinstance(record.msg, LLMCallEvent):
                formatted = self._format_llm_call(record.msg.kwargs)
            else:
                formatted = base_msg
            return formatted
        except Exception as e:
            print(e)
            exit()

        return base_msg

    def _format_llm_start(self, data):
        agent_id = data.get("agent_id")
        if not agent_id:
            agent_id = "Unknown"
        lines = [
            "\n" + "=" * 80,
            f"🤖 LLM Stream Call",
            f"Agent: {agent_id.split('-')[0]}",
            "-" * 80,
        ]

        messages = data.get("messages", [])
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            lines.append(f"\n📨 Message {i+1} ({role}):")
            lines.append(self._indent_text(content, 4))

        return "\n".join(lines)

    def _format_llm_end(self, data):
        lines = []

        response = data.get("response", {})

        # Show usage stats
        usage = response.get("usage", {})
        if usage:
            lines.append(f"\n📊 Token Usage:")
            lines.append(f"    Prompt: {usage.get('prompt_tokens', 0)}")
            lines.append(f"    Completion: {usage.get('completion_tokens', 0)}")
            lines.append(
                f"    Total: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
            )

        # Show content
        content = response.get("content")
        if content:
            lines.append(f"\n💬 Response:")
            if isinstance(content, dict):
                # Format nested JSON content
                lines.append(json.dumps(content, indent=4, ensure_ascii=False))
            else:
                # Truncate if too long
                lines.append(self._indent_text(str(content), 4))

        lines.append("=" * 80)
        return "\n".join(lines)

    def _format_llm_call(self, data):
        agent_id = data.get("agent_id")
        if not agent_id:
            agent_id = "Unknown"
        lines = [
            "\n" + "=" * 80,
            f"🤖 LLM Call",
            f"Agent: {agent_id.split('-')[0]}",
            "-" * 80,
        ]

        # Show messages
        messages = data.get("messages", [])
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            lines.append(f"\n📨 Message {i+1} ({role}):")
            lines.append(self._indent_text(content, 4))

        response = data.get("response", {})
        if response:
            # Show token usage
            usage = response.get("usage", {})
            if usage:
                lines.append(f"\n📊 Token Usage:")
                lines.append(f"    Prompt: {usage.get('prompt_tokens', 0)}")
                lines.append(f"    Completion: {usage.get('completion_tokens', 0)}")
                lines.append(
                    f"    Total: {usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)}"
                )

            # Show response summary
            choices = response.get("choices", [])
            for choice in choices:
                finish_reason = choice.get("finish_reason", "unknown")
                message = choice.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_call_id = tool_call.get("id", "unknown")
                        function = tool_call.get("function", {})
                        tool_call_name = function.get("name", "unknown")
                        tool_call_args = function.get("arguments", {})
                        lines.append(f"\n📤 Tool Call: {tool_call_id}:")
                        lines.append(self._indent_text(f"name: {tool_call_name}", 4))
                        lines.append(
                            self._indent_text(f"arguments: {tool_call_args}", 4)
                        )

                lines.append(self._indent_text(content, 4))

        lines.append("=" * 80)
        return "\n".join(lines)

    def _indent_text(self, text, spaces):
        if text is None:
            return ""
        indent = " " * spaces
        return "\n".join(indent + line for line in text.split("\n"))


def setup_logging():
    # Create file handler with custom filter
    file_handler = logging.FileHandler(
        f"logging/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log", mode="a"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        ReadableFormatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
    )
    file_handler.addFilter(LLMFilter())

    # Create logger and add handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # For trace logging with filter
    trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
    stream_handler = logging.StreamHandler()
    stream_handler.addFilter(LLMFilter())
    trace_logger.addHandler(stream_handler)
    trace_logger.setLevel(logging.INFO)
