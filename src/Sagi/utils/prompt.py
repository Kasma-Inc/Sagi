def get_step_triage_prompt(*, task: str, current_plan: str, names: list[str]) -> str:
    """Generates a prompt template for triaging the step execution to the right team member.

    Args:
        task: Description of the main task.
        current_plan: Currently executing sub-task.
        names: List of available team members to select from.
    """
    template = """
        Recall we are working on the following request:

        {task}

        We are executing the following sub-task based on the plan:
        {current_plan}

        To make progress on the request, please answer the following questions, including necessary reasoning:

            - Who should speak next? (select from: {names})
            - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)

        Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

            {{
                "next_speaker": {{
                    "reason": string,
                    "answer": string (select from: {names})
                }},
                "instruction_or_question": {{
                    "reason": string,
                    "answer": string
                }}
            }}
    """

    return template.format(
        task=task,
        current_plan=current_plan,
        names=", ".join(names),
    )


def get_reflection_step_completion_prompt(
    *, current_plan: str, conversation_context: str
) -> str:
    """Generates a prompt template for evaluating plan step completion.

    Args:
        current_plan: The plan step being evaluated for completion.
        conversation_context: The formatted conversation history to analyze.
    """
    template = """
        Review the conversation history and determine if the following plan step has been completed:

        PLAN STEP: {current_plan}

        CONVERSATION CONTEXT:
        {conversation_context}

        REMEMBER: The POSIX exit code 0 indicates success of code execution.

        Analyze the messages to check if:
        1. The required actions for this step have been successfully executed
        2. The expected outputs or results from this step are present
        3. There are no pending questions or unresolved issues related to this step

        Return a JSON with the following structure:
        {{
            "is_complete": true/false,
            "reason": "Detailed explanation of why the step is considered complete or incomplete"
        }}
    """

    return template.format(
        current_plan=current_plan,
        conversation_context=conversation_context,
    )


def get_appended_plan_prompt(
    *, current_task: str, contexts_history: str, team_composition: str
) -> str:
    """Generates a prompt template for appended planning.

    Args:
        current_task: The current task
        contexts_history: The formatted context history
        team_composition: The team composition
    """
    template = """
        CURRENT TASK: {current_task}

        PREVIOUS CONTEXT SUMMARY: {contexts_history}

        TEAM COMPOSITION: {team_composition}

        
        You are a professional planning assistant. 
        Based on the team composition, user query, and the previous context, create a detailed plan for the next steps.

        Each plan step should contain the following elements:
        1. name: A short title for this step
        2. description: Detailed explanation of the step objective and financial content
        3. data_collection_task: Specific instructions for gathering financial data needed for this step (optional)
        4. code_executor_task: Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK.(optional)
    """
    return template.format(
        current_task=current_task,
        contexts_history=contexts_history,
        team_composition=team_composition,
    )


def get_final_answer_prompt(*, task: str) -> str:
    """Generates a prompt template for final answer.

    Args:
        task: The current task
    """
    template = """
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"""
    return template.format(task=task)


def get_data_collector_prompt(
    *, shared_section: str, step_section: str, step_content: str
) -> str:
    """Generates a prompt for the DataCollector agent role.

    Args:
        shared_section: The formatted summaries of previous results.
        step_section: The formatted current sub-task section.
        step_content: The plain content of the current sub-task.
    """
    template = """
        Completed summaries:
        {shared_section}

        Current sub-task:
        {step_section}

        Your task:
        - Using the above summaries, search for and extract only the key information relevant to "{step_content}".
        - Return only those key points; do not include unrelated data.
    """
    return template.format(
        shared_section=shared_section,
        step_section=step_section,
        step_content=step_content,
    )


def get_code_executor_prompt(*, shared_section: str, step_section: str) -> str:
    """Generates a prompt for the CodeExecutor agent role.

    Args:
        shared_section: The formatted summaries of previous results.
        step_section: The formatted current sub-task section.
    """
    template = """
        Completed summaries:
        {shared_section}

        Current sub-task:
        {step_section}

        Your task:
        - Write Python code to accomplish "{step_section}".
        - Return the full, runnable code block in a ```python``` fence; do not add any extra explanation.
    """
    return template.format(
        shared_section=shared_section,
        step_section=step_section,
    )


def get_default_execution_prompt(*, shared_section: str, step_section: str) -> str:
    """Generates the default execution prompt for other agent roles.

    Args:
        shared_section: The formatted summaries of previous results.
        step_section: The formatted current sub-task section.
    """
    template = """
        {shared_section}

        {step_section}

        Your task:
        - Execute using only the above summaries and sub-task content.
    """
    return template.format(
        shared_section=shared_section,
        step_section=step_section,
    )


def get_previous_results_section(*, relevant_summaries: list[str]) -> str:
    """Generates a section listing all previous result summaries.

    Args:
        relevant_summaries: A list of strings summarizing past results.
    """
    if relevant_summaries:
        items = "\n".join(f"- {s}" for s in relevant_summaries)
    else:
        items = "- (none)"
    template = """
        Previous Results:
        {items}
    """
    return template.format(items=items)


def get_current_subtask_section(*, step) -> str:
    """Generates the section for the current sub-task details.

    Args:
        step: An object with attributes `group_id`, `step_id`, and `content`.
    """
    template = """
        Current Sub-Task (group {group_id}, step {step_id}):
        {content}
    """
    return template.format(
        group_id=step.group_id,
        step_id=step.step_id,
        content=step.content,
    )


def get_instruction_prompt(
    *, refined_context: str, instruction_or_question: str
) -> str:
    """Combines refined context with the instruction or question into a single prompt.

    Args:
        refined_context: The up-to-date context string.
        instruction_or_question: The instruction or question to pose.
    """
    template = """
        {refined_context}

        === Instruction ===
        {instruction_or_question}
    """
    return template.format(
        refined_context=refined_context,
        instruction_or_question=instruction_or_question,
    )


def get_relevance_filter_prompt(*, numbered: str, task: str) -> str:
    """Creates a prompt asking which numbered summaries are relevant to the given task.

    Args:
        numbered: A numbered list of summary strings.
        task: Description of the current task.
    """
    template = """
        Below are previous result summaries, each numbered:
        {numbered}

        Current Task:
        {task}

        Question: Which of the above summaries are directly relevant to executing this task?
        Answer with a JSON array of numbers, e.g. [1,4].
    """
    return template.format(
        numbered=numbered,
        task=task,
    )
