from typing import AsyncGenerator, List, Optional, Sequence

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.agents._code_executor_agent import RetryDecision
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    CodeExecutionEvent,
    CodeGenerationEvent,
    ModelClientStreamingChunkEvent,
    TextMessage,
    ThoughtEvent,
)
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock, CodeResult
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    SystemMessage,
    UserMessage,
)

from Sagi.tools.stream_code_executor.stream_code_executor import (
    CodeFileMessage,
    StreamCodeExecutor,
    CodeBlockErrorHistory,
    CodeStepHistory,
)
from Sagi.tools.stream_code_executor.stream_docker_command_line_code_executor import (
    StreamDockerCommandLineCodeExecutor,
)

import re

class StreamCodeExecutorAgent(CodeExecutorAgent):
    def __init__(
        self,
        name: str,
        stream_code_executor: StreamCodeExecutor,
        *,
        model_client: ChatCompletionClient | None = None,
        model_context: ChatCompletionContext | None = None,
        model_client_stream: bool = False,
        max_retries_on_error: int = 0,
        description: str | None = None,
        system_message: str | None = CodeExecutorAgent.DEFAULT_SYSTEM_MESSAGE,
        sources: Sequence[str] | None = None,
    ) -> None:
        super().__init__(
            name,
            stream_code_executor,
            model_client=model_client,
            model_context=model_context,
            model_client_stream=model_client_stream,
            max_retries_on_error=max_retries_on_error,
            description=description,
            system_message=system_message,
            sources=sources,
        )
        self._code_executor: StreamCodeExecutor = stream_code_executor
        self.chat_id: Optional[str] = None
        self._step_history: List[CodeStepHistory] = []  # To keep track of summary of each code execution steps

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        """
        Process the incoming messages with the assistant agent and yield events/responses as they happen.
        """

        # Start the code executor if it is a StreamDockerCommandLineCodeExecutor
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            await self._code_executor.start()

        # Gather all relevant state here
        agent_name = self.name
        model_context = self._model_context
        system_messages = self._system_messages
        model_client = self._model_client
        model_client_stream = self._model_client_stream
        max_retries_on_error = self._max_retries_on_error

        if model_client is None:  # default behaviour for backward compatibility
            # execute generated code if present
            code_blocks: List[CodeBlock] = await self.extract_code_blocks_from_messages(
                messages
            )
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=self.NO_CODE_BLOCKS_FOUND_MESSAGE,
                        source=agent_name,
                    )
                )
                return

            async for result in self.execute_code_block(
                code_blocks, cancellation_token
            ):
                if isinstance(result, CodeFileMessage):
                    yield result
                elif isinstance(result, CodeResult):
                    yield Response(
                        chat_message=TextMessage(
                            content=result.output,
                            source=self.name,
                        )
                    )
            return

        inner_messages: List[BaseAgentEvent | BaseChatMessage] = []

        await model_context.clear()  # Clear the model context to start fresh

        # Add the summary of the previous steps to the model context
        for history in self._step_history:
            await model_context.add_message(
                SystemMessage(
                    content=self.history_to_model_text(history),
                    source=agent_name,
                )
            )


        current_history_path: List[CodeBlockErrorHistory] = [] # current error history path
        message_texts = [f" - {msg.to_model_text()}" for msg in messages] # messages instruction
        current_history_path.append( # Append the first stage of error history (root node)
            CodeBlockErrorHistory(
                code_blocks=None,  # No code blocks at the first stage
                shell_commands=None,  # No command blocks at the first stage
                error="This is the first stage. The code has not been generated yet. Please follow the instructions above to generate the code.",
                previous_state=-1,
            )
        )

        # Add the instruction messages to the model context
        await model_context.add_message(
            UserMessage(
                content=f"Current Instructions with the previous messages history:\n{'\n'.join(message_texts)}",
                source=agent_name,
            )
        )

        # Initialize code_blocks here, so that the final code_blocks after the loop can be used by refering to this variable
        code_blocks : CodeBlock = None

        # Initialize the result of the code execution, so that the final result can be used by refering to this variable
        result_output : str = None

        for nth_try in range(
            max_retries_on_error + 1
        ):  # Do one default generation, execution and inference loop
            # Step 1: Add new user/handoff messages to the model context
            
            original_messages = await model_context.save_state()

            assert len(current_history_path)>0, "Expected current_history_path to be initialized."

            await model_context.add_message(
                UserMessage(
                    content=self.error_history_prompt(current_history_path),
                    source=agent_name,
                )
            )

            # Step 2: Run inference with the model context
            model_result = None
            async for inference_output in self._call_llm(
                model_client=model_client,
                model_client_stream=model_client_stream,
                system_messages=system_messages,
                model_context=model_context,
                agent_name=agent_name,
                cancellation_token=cancellation_token,
            ):
                if isinstance(inference_output, CreateResult):
                    model_result = inference_output
                else:
                    # Streaming chunk event
                    yield inference_output

            assert model_result is not None, "No model result was produced."

            # Step 3: [NEW] If the model produced a hidden "thought," yield it as an event
            if model_result.thought:
                thought_event = ThoughtEvent(
                    content=model_result.thought, source=agent_name
                )
                yield thought_event
                inner_messages.append(thought_event)

            # Step 4: Add the assistant message to the model context (including thought if present)
            await model_context.add_message(
                AssistantMessage(
                    content=model_result.content,
                    source=agent_name,
                    thought=getattr(model_result, "thought", None),
                )
            )

            # Step 5: Extract the code blocks and previous state from inferred text
            assert isinstance(
                model_result.content, str
            ), "Expected inferred model_result.content to be of type str."

            # Find the PREVIOUS_STATE in the model's response
            previous_state_match = re.search(r"PREVIOUS_STATE:\s*(-?\d+)", model_result.content)

            if not previous_state_match:
                print("Model's response did not contain 'PREVIOUS_STATE'. Assume it to be the last stage.")
                previous_state = len(current_history_path) - 1  # Assume the last stage if not found
            else:
                try:
                    previous_state = int(previous_state_match.group(1))
                except ValueError:
                    raise ValueError("Expected 'PREVIOUS_STATE' to be a valid integer in the model's response.")

            if (previous_state == -1):
                # No solution can be found, so we exit the loop
                yield Response(
                    chat_message=TextMessage(
                        content=f"No solution can be found. The model's response was: {model_result.content}",
                        source=agent_name,
                    )
                )
                return
            
            assert (0 <= previous_state < len(current_history_path)), "Expected 'PREVIOUS_STATE' to be within the range of error history."

            env_issue = "ENVIRONMENT_ISSUE" in model_result.content

            shell_commands: List[CodeBlock] = None

            if env_issue:
                # The issue arises from the environment itself, so the code_blocks can be refer from the ancestor error history state(node)
                # We don't store the code blocks in the error history to save the model_context_size and token used,
                # because the code blocks are not changed to fix the environment issue.
                now = previous_state
                while now != -1 and current_history_path[now].code_blocks is None:
                    now = current_history_path[now].previous_state

                assert now != -1, "Expected to find a ancestor state with code blocks."
                code_blocks = current_history_path[now].code_blocks

                # Shell to fix the environment issue
                command_blocks = self._extract_markdown_code_blocks(str(model_result.content))
                shell_commands = [block for block in command_blocks if block.language == "sh"]

            else:
                # The issue arises from the code itself, so we need to extract the code blocks from the model's response
                code_blocks = self._extract_markdown_code_blocks(str(model_result.content))


            # Step 6: Exit the loop if no code blocks found
            if not code_blocks:
                yield Response(
                    chat_message=TextMessage(
                        content=f"No code blocks found. The model's response was: {model_result.content}",
                        source=agent_name,
                    )
                )
                return

            # combine the code blocks with the shell commands if they are present
            combined_blocks = []
            if env_issue and shell_commands:
                combined_blocks.extend(shell_commands)
            if code_blocks:
                combined_blocks.extend(code_blocks)

            # Step 7: Yield a CodeGenerationEvent
            inferred_text_message: CodeGenerationEvent = CodeGenerationEvent(
                retry_attempt=nth_try,
                content=model_result.content,
                code_blocks=combined_blocks,
                source=agent_name,
            )

            yield inferred_text_message
            inner_messages.append(inferred_text_message)

            # Step 8: Execute the extracted code blocks
            for codeblock in combined_blocks:
                async for result in self.execute_code_block(
                    [codeblock], cancellation_token
                ):
                    # Return CodeFileMessage or CodeResult
                    # CodeFileMessage stores the command and the code file content
                    # CodeResult stores the exit code, stdout and stderr
                    # First yield the CodeFileMessage and then the CodeResult
                    if isinstance(result, CodeFileMessage):
                        result.source = self.name
                        yield result
                        inner_messages.append(result)
                    elif isinstance(result, CodeResult):
                        # Step 9: Update model context with the code execution result
                        result_output = result.output

                        await model_context.add_message(
                            SystemMessage(
                                content=f"The command {result.code_file} was executed with the following output: {result.description}",
                                source=agent_name,
                            )
                        )

                        # Step 10: Yield a CodeExecutionEvent
                        code_execution_event = CodeExecutionEvent(
                            retry_attempt=nth_try, result=result, source=self.name
                        )
                        yield code_execution_event
                        inner_messages.append(code_execution_event)
                    
                if result.exit_code != 0:
                    # If the code execution failed, we break the loop and store the error -> no need to continue the execution
                    break

            # If execution was successful or last retry, then exit
            if result.exit_code == 0 or nth_try == max_retries_on_error:
                if nth_try == max_retries_on_error and result.exit_code != 0:
                    # If we reached the maximum number of retries and the exit code is still non-zero, we break the loop
                    yield Response(
                        chat_message=TextMessage(
                            content=f"Reached maximum retries ({max_retries_on_error}) with exit code {result.exit_code}. The model's response was: {model_result.content}",
                            source=agent_name,
                        )
                    )

                    model_context.add_message(
                        SystemMessage(
                            content=f"Reached maximum retries ({max_retries_on_error}) with exit code {result.exit_code}. The model's response was: {model_result.content}",
                            source=agent_name,
                        )
                    )
                    
                break

            # Reset context only when we know that the code will be executed again (in order to add the new history tree)
            # If the code can be run, we have to keep the model_context because this context will be used for summarization and reflection
            await model_context.clear()
            await model_context.load_state(original_messages)

            # Step 11: If exit code is non-zero, then we add the error to the error history
            #          and pop the current_history_path until the previous state
    
            # Pop the current history path until the previous state
            while len(current_history_path) > previous_state + 1:
                current_history_path.pop()

            # Create a new error node with the current code blocks/command blocks and error message
            new_error_node = CodeBlockErrorHistory(
                shell_commands=shell_commands if env_issue else None, # If the error is due to syntax/logical issue, we don't need to store the command blocks
                code_blocks=code_blocks if not env_issue else None, # If the error is due to environment issue, we don't need to store the code blocks
                error=f"The command {result.code_file} was executed with the following output: {result.description}",
                previous_state=previous_state,
            )
            
            current_history_path[previous_state].children_error_nodes.append(new_error_node) # connect the new error node to the previous state
            current_history_path.append(new_error_node)  # Add the new error node to the current history path

        self._step_history.append(
            CodeStepHistory(
                instruction="\n".join(message_texts),
                code_blocks=code_blocks,
                result=result_output if result_output is not None else "No code execution in this step.",
            )
        )

        # Stop the code executor if it is a StreamDockerCommandLineCodeExecutor
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            await self._code_executor.stop()

        # Always reflect on the execution result
        async for (
            reflection_response
        ) in StreamCodeExecutorAgent._reflect_on_code_block_results_flow(
            system_messages=system_messages,
            model_client=model_client,
            model_client_stream=model_client_stream,
            model_context=model_context,
            agent_name=agent_name,
            inner_messages=inner_messages,
        ):
            yield reflection_response  # Last reflection_response is of type Response so it will finish the routine

    async def execute_code_block( # TO DO: have to fix this function, sometimes it executes the code in the near time (conflict)
        self, code_blocks: List[CodeBlock], cancellation_token: CancellationToken
    ) -> AsyncGenerator[CodeFileMessage | CodeResult, None]:
        # Execute the code blocks.

        # Ensure the code executor ir runnung
        if isinstance(self._code_executor, StreamDockerCommandLineCodeExecutor):
            assert getattr(self._code_executor, '_running', False), "Expected StreamDockerCommandLineCodeExecutor to be running."

        async for result in self._code_executor.execute_code_blocks_stream(
            self.chat_id, code_blocks, cancellation_token=cancellation_token
        ):
            if isinstance(result, CodeResult):
                if result.output.strip() == "":
                    # No output
                    result.description = f"The script ran but produced no output to console. The POSIX exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed."
                elif result.exit_code != 0:
                    # Error
                    result.description = f"The script ran, then exited with an error (POSIX exit code: {result.exit_code})\nIts output was:\n{result.output}"
                else:
                    result.description = result.output
            yield result


    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        await self._model_context.clear()


    def history_to_model_text(self, history: CodeStepHistory) -> str:
        return (
            f"Previous Instruction: {history.instruction}\n"
            f"Result Code Blocks: {history.code_blocks}\n"
            f"Result Execution: {history.result}\n"
        )

    def error_history_prompt(self, current_history_path: List[CodeBlockErrorHistory]) -> str:
        return (
            "Review this error history tree showing previous code execution attempts\n"
            "Each node(history path item) is a CodeBlockErrorHistory objects, formatted as :\n"
            " - code_blocks: code at that stage (None at first stage)\n"
            " - error: error message\n"
            " - children_error_nodes: recursive list of CodeBlockErrorHistory objects (tree structure)\n"
            "   representing previously failed solution attempts at that stage\n"
            " - previous_state: index of parent state (0-based)\n\n"
            "First try fixing the last stage (leaf node in the current path). If too difficult, go back to earlier stages but avoid methods already in children_error_nodes.\n\n"
            "Current error path:\n" +
            "\n".join(
                f" Stage {i}:\n"
                f" Code: {current_history_path[i].code_blocks}\n"
                f" Error: {current_history_path[i].error}\n"
                f" children_error_nodes: {current_history_path[i].children_error_nodes}\n"
                f" shell_commands: {current_history_path[i].shell_commands}"
                for i in range(len(current_history_path))
            ) +
            "\n\nPLEASE Write PREVIOUS_STATE: <number> to indicate which stage to continue from (unless you are trying to reflect the process, you don't need to specify previous state).\n\n"
            "For environment issues (including missing packages):\n"
            "1. Write ENVIRONMENT_ISSUE on a separate line\n"
            "2. Follow it with a proper code block like this:\n"
            "```sh\n"
            "pip install package_name\n"
            "```\n"
            "3. PREVIOUS_STATE: <number> to indicate which stage to continue from, for the environment issue, the previous state must NOT be 0, because there will be no code blocks\n\n"
            "For non-environment issues:\n"
            "1. Write the code to fix the issue in a code block\n"
            "2. PREVIOUS_STATE: <number> to indicate which stage to continue from, for the non environment issue, the previous state can be any number\n\n"
            "Now generate code to fix the issue. If no solution exists, write PREVIOUS_STATE: -1\n"
            "NOTE: you don't need to create the all new codeblocks by yourself, you can refer to the previous messages or error path (if available), and make some changes from them."
        )


    @classmethod
    async def _reflect_on_code_block_results_flow(
        cls,
        system_messages: List[SystemMessage],
        model_client: ChatCompletionClient,
        model_client_stream: bool,
        model_context: ChatCompletionContext,
        agent_name: str,
        inner_messages: List[BaseAgentEvent | BaseChatMessage],
    ) -> AsyncGenerator[Response | ModelClientStreamingChunkEvent | ThoughtEvent, None]:
        """
        If reflect_on_code_block_results=True, we do another inference based on tool results
        and yield the final text response (or streaming chunks).
        """
        reflection_prompt = (
            "Reflect briefly ( <= 4 sentences ) on the code execution results. "
            "If the process was error, please state clearly that the code was errored. "
            "If the process was successful, please state clearly that the code was successful.\n"
            "IMPORTANT: do NOT repeat the code blocks!!!\n"
            "Remember: you are reflection agent, you do NOT have to provide PREVIOUS_STATE or ENVIRONMENT_ISSUE, you can just reflect on the process.\n"
        )

        all_messages = system_messages + await model_context.get_messages() + [UserMessage(content=reflection_prompt, source=agent_name)]
        llm_messages = cls._get_compatible_context(model_client=model_client, messages=all_messages)

        reflection_result: Optional[CreateResult] = None

        if model_client_stream:
            async for chunk in model_client.create_stream(llm_messages):
                if isinstance(chunk, CreateResult):
                    reflection_result = chunk
                elif isinstance(chunk, str):
                    yield ModelClientStreamingChunkEvent(content=chunk, source=agent_name)
                else:
                    raise RuntimeError(f"Invalid chunk type: {type(chunk)}")
        else:
            reflection_result = await model_client.create(llm_messages)

        if not reflection_result or not isinstance(reflection_result.content, str):
            raise RuntimeError("Reflect on tool use produced no valid text response.")

        # --- NEW: If the reflection produced a thought, yield it ---
        if reflection_result.thought:
            thought_event = ThoughtEvent(content=reflection_result.thought, source=agent_name)
            yield thought_event
            inner_messages.append(thought_event)

        # Add to context (including thought if present)
        await model_context.add_message(
            AssistantMessage(
                content=reflection_result.content,
                source=agent_name,
                thought=getattr(reflection_result, "thought", None),
            )
        )

        yield Response(
            chat_message=TextMessage(
                content=reflection_result.content,
                source=agent_name,
                models_usage=reflection_result.usage,
            ),
            inner_messages=inner_messages,
        )