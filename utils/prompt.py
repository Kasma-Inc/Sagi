# manage prompts
def _get_tools_prompt(tools_description: str, date) -> str:
    return f"""
        YOU SHOULD USE AT LEAST THREE DIFFERENT tools.
        YOU SHOULD STRICTLY FOLLOW the required JSON format.

        You are a versatile assistant with strategic tool-chaining capabilities, designed to handle a wide range of tasks by combining and sequencing tools effectively.

        **Available Tools:**
        {tools_description}

        **Response Protocol:**
        - Output ONLY a JSON array of tool calls in sequence (multiple calls allowed).
        - Each tool call should include the tool name and its required arguments.
        - Focus solely on determining the appropriate tools and their sequence for the task.

        **Search Optimization Rules:**
        1. Keyword Extraction: 
           - Analyze user query to identify 2-3 core keywords/concepts
           - Generate multiple search queries using synonyms, related terms, and contextual variations
        2. Search Constraints:
           - NEVER use the raw user query directly as search input
           - Split complex queries into parallel search calls
        3. Multiple calls:
           - ALLOW use SEARCH tools MULTIPLE times with DIFFERENT arguments
        4. NOW is {date}, PLEASE DO NOT search things based on your time

        **IMPORTANT:** 
        - You must ONLY respond with the STRICT JSON FORMAT below if you need to use tools.
        - NO JSON markdown label. NO OTHER CONTENT if you use tools.
        - A tool with the same arguments cannot be used again. You can use the same tool with different arguments.
        - USE AS MANY TOOLS AS YOU CAN.

        **STRICT JSON FORMAT:**
        [
            {{
                "tool": "tool-name",
                "arguments": {{
                    "argument-name": "value"
                }}
            }},
            {{
                "tool": "tool-name",
                "arguments": {{
                    "argument-name": "value"
                }}
            }},
            ... (more tools if needed)
        ]

        **NOTE:**
        If NO tool is needed, REPLY DIRECTLY with user input, STRICTLY follow the format below:
        [
            {{
                "tool": "direct_response",
                "arguments": {{
                    "content": "Your answer here"
                }}
            }}
        ]
    """


def _generation_prompt(tools_history: list) -> str:
    prompt = (
        "The following is additional information collected from various tools to assist in answering the user's query. Please use this information to formulate your response. "
        "If multiple tools provide relevant data, integrate this information appropriately. "
        "Do not include facts or details not provided in the tool responses, but you may perform direct logical deductions or reasoning based on the given information.\n\n"
    )

    prompt += "Tool Responses:\n\n"

    for i, tool_record in enumerate(tools_history):
        prompt += f"Tool: {tool_record['tool']}\n"
        prompt += f"Arguments: {tool_record['arguments']}\n"
        prompt += f"Response: {tool_record['response']}\n"
        if i < len(tools_history) - 1:
            prompt += "---\n"

    prompt += "\nBased on the information above, provide a comprehensive and relevant response to the user's query"
    prompt += "\nIf the guideline tool (ESG, press-release, chair statement, s.t.)exist, please STRICLY follow the guideline to generate answer."

    return prompt


def _retry_strategy_prompt(
    used_tools: list, reason: str, missing_info: str, system_message: str
) -> str:
    return (
        f"For the user question,\n"
        "I have already tried the following tools:\n"
        f"{used_tools}" + "\n"
        f"However, the previous attempts were not satisfactory.\n"
        f"The reason is {reason}\n"
        f"The missing info are {missing_info}\n"
        f"{system_message}"
    )


def _evaluation_prompt(user_input: str, final_response: str) -> str:
    return (
        "You are an evaluator. Assess the answer based on:\n"
        "1. **Relevance**: Does the answer directly address the user's question without unnecessary digressions?\n"
        "2. **Accuracy**: Is the information logically consistent and coherent within the context provided? (Do not verify real-time factual correctness; assume presented information is accurate unless it contains obvious contradictions.)\n"
        "3. **Completeness**: Are all key points and necessary context included to fully answer the question?\n\n"
        "Return ONLY JSON and STRICTLY with the following structure (no extra text outside JSON):\n"
        "{\n"
        '  "satisfied": true/false,\n'
        '  "reason": "explain your judgment",\n'
        '  "missing_info": ["list specific missing details or context (if any)"]\n'
        "}\n\n"
        "Example response:\n"
        "{\n"
        '  "satisfied": true,\n'
        '  "reason": "The answer is relevant, informative and comprehensive",\n'
        '  "missing_info": []\n'
        "}\n\n"
        f"User's question: {user_input}\n"
        f"Current answer: {final_response}"
    )


def _initial_planning_user_prompt(user_input: str) -> str:
    return f"""
        Create a reasonable plan with clear steps to accomplish the task: {user_input}
        Using Language SAME as the above request use
    """


def _initial_planning_system_prompt() -> str:
    return f"""
        You are a planning assistant. Create a concise, actionable plan with clear steps.
        Focus on key milestones rather than detailed sub-steps.
        Optimize for clarity and efficiency.
    """


def step_triage_prompt(*, task: str, current_plan: str, names: list[str]) -> str:
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


def reflection_step_completion_prompt(
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


def appended_plan_prompt(
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
