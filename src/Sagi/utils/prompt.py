SLIDE_CATEGORY_INFO = """
Opening slide: The first slide that introduces the presentation, typically including the title, presenter's name, and other introductory information.
Ending slide: The final slide that concludes the presentation, usually featuring a summary, conclusions, or a call to action, along with contact details or acknowledgments.
Normal texts slide: Content slides that present information primarily through text, such as bullet points or paragraphs.
Normal texts with images slide: Content slides that combine text with images or diagrams to enhance understanding and visual appeal.
"""


def get_step_triage_prompt(
    *, task: str, current_plan: str, names: list[str], team_description: str
) -> str:
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

        The team members are:
        {team_description}
          
        To make progress on the request, please answer the following questions, including necessary reasoning:

            - Who should speak next? You MUST select from {names}
            - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)

        Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:
        ```
            {{
                "next_speaker": {{
                    "instruction": string,
                    "answer": string (select from: {names})
                }},
            }}
        ```
        
        **IMPORTANT NOTE**: If you need to create files or add content to existing files, please select the 'CodeExecutor' to implement this using Python.
    """

    return template.format(
        task=task,
        current_plan=current_plan,
        names=", ".join(names),
        team_description=team_description,
    )


def get_step_triage_prompt_cn(
    *, task: str, current_plan: str, names: list[str], team_description: str
) -> str:
    """
    生成一个提示模板，用于将步骤执行分配给合适的团队成员。

        参数：
            task: 主要任务的描述。
            current_plan: 当前正在执行的子任务。
            indexed_names: 可供选择的团队成员名单。
    """
    template = """
        回顾我们正在处理的以下请求：

        {task}

        根据计划，我们正在执行以下子任务：
        {current_plan}

        团队成员包括：
        {team_description}
        
        
        为了推进请求的处理，请回答以下问题，并提供必要的理由：

            - 下一个应该发言的人是谁？你必须从 {names} 中选择。
            - 你想要给这个团队成员的指令或问题是什么？（像直接对他们说话一样措辞，并包括他们可能需要的任何特定信息）

        请按照以下模式以纯 JSON 格式输出答案。JSON 对象必须可以直接解析。不要输出任何非 JSON 的内容，也不要偏离这个模式：
        ```
            {{
                "next_speaker": {{
                    "instruction": 字符串的指令或问题,
                    "answer": 选择的发言人的名字 (从 {names} 中选择)
                }},
            }}
        ```

        **重要说明**：如果要创建文件或者向已有的文件中添加内容，请选择 'CodeExecutor' 来使用 Python 实现。
    """

    return template.format(
        task=task,
        current_plan=current_plan,
        names=", ".join(names),
        team_description=team_description,
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


def get_reflection_step_completion_prompt_cn(
    *, current_plan: str, conversation_context: str
) -> str:
    """
    生成一个提示模板，用于评估计划步骤是否完成。

        参数：
            current_plan: 正在评估完成情况的计划步骤。
            conversation_context: 用于分析的格式化对话历史。
    """
    template = """
        回顾对话历史，判断以下计划步骤是否已完成：

        计划步骤：{current_plan}

        对话上下文：
        {conversation_context}

        记住：POSIX退出代码0表示代码执行成功。

        分析消息以检查：
        1. 这一步骤所需的操作是否已成功执行
        2. 这一步骤的预期输出或结果是否已存在
        3. 是否没有与这一步骤相关的未解决的问题或未回答的问题

        返回一个具有以下结构的JSON：
        {{
            "is_complete": true/false,
            "reason": "详细解释为什么认为该步骤已完成或未完成"
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
        2. description: Detailed explanation of the step objective
        3. data_collection_task: Specific instructions for gathering data needed for this step (optional)
        4. code_executor_task: Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK.(optional)
    """
    return template.format(
        current_task=current_task,
        contexts_history=contexts_history,
        team_composition=team_composition,
    )


def get_appended_plan_prompt_cn(
    *, current_task: str, contexts_history: str, team_composition: str
) -> str:
    """
    生成一个追加计划的提示模板。

        参数：
            current_task: 当前任务
            contexts_history: 格式化的上下文历史
            team_composition: 团队构成
    """
    template = """
        当前任务：{current_task}

        前序上下文总结：{contexts_history}

        团队构成：{team_composition}

        你是一位专业的计划助理。
        根据团队构成、用户查询请求以及前序上下文，为下一步骤制定详细计划。

        每个计划步骤应包含以下元素：
        1. name: 此步骤的简短标题
        2. description: 步骤目标的详细说明
        3. data_collection_task: 收集此步骤所需数据的具体说明（可选）
        4. code_executor_task: 对代码执行器应执行的操作的描述，只需详细说明即可，不需要实际的代码块。（可选）
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


def get_final_answer_prompt_cn(*, task: str) -> str:
    """
    生成一个提示模板，用于最终答案。

        参数：
            task: 当前任务
    """
    template = """
我们正在处理以下任务：
{task}

我们已经完成了任务。

上述消息包含了完成任务过程中发生的对话。

根据收集到的信息，提供对原始请求的最终答案。
答案应该像直接对用户说话一样措辞。

请用中文回答。
"""
    return template.format(task=task)


def get_high_level_ppt_plan_prompt(*, task: str, file_content: str) -> str:
    """Generates a prompt template for high-level PPT plan.

    Args:
        task: The current task
        file_content: The file content
    """
    template = """
You are an expert PowerPoint presentation planning assistant.

Your task is to generate a clear, high-level PowerPoint presentation outline to guide slide-by-slide creation.

Instructions:

1. Review the provided content ({file_content}) and user query ({query}) carefully for key topics and objectives.
2. Structure the outline using best practices for presentation design, ensuring logical flow and coherence.
3. Represent each slide as a bullet point; each point must include:
- The main purpose or idea of the slide (e.g., Introduction, Overview of Challenges, Proposed Solutions, Conclusion).
- The assigned category for the slide, as defined in the following information: {slide_category_info}.
4. Do not include detailed content or slide text; focus exclusively on slide purpose, category, and sequence.
5. Organize the plan with a clear progression from introduction, main content, to conclusion/next steps, as appropriate for the subject.
6. Include a minimum of 6 slides to ensure comprehensive coverage.
7. Present the outline as a numbered list for clarity.

Return only the slide-by-slide outline as the output.
    """
    return template.format(
        file_content=file_content, query=task, slide_category_info=SLIDE_CATEGORY_INFO
    )


def get_template_selection_prompt(*, slide_content: str, template_options: str) -> str:
    """Generates a prompt template for template selection.

    Args:
        slide_content: The content for the slide
        template_options: Available template options to choose from
    """
    template = """
You are a presentation design expert. Your task is to analyze a slide's content and select the most appropriate template from the provided options.

## CRITICAL INSTRUCTION:
**Return ONLY the template_id number. Do not include explanations, extra text, or modify the format.**

## Task Steps:
1. Read and understand the current slide's content.
2. Review all provided template options.
3. Select the template that aligns best with the slide's category, content type, and structure.
4. Respond with the template_id number.

## Slide Content:
{slide_content}

## Available Templates:
{template_options}
"""

    return template.format(
        slide_content=slide_content, template_options=template_options
    )


def get_expand_plan_prompt(*, task: str, slide_content: str) -> str:
    """Generates a prompt template for expand plan.

    Args:
        task: The current task
        slide_content: The content for the slide
    """
    template = """
    You are a plan expander. Your task is to expand the plan for the following slide content:

    ## Slide Content:
    {slide_content}

    ## User Query:
    {task}

    ## Expl
    ## Return the expanded plan in the following format:
    {{
        "name": "A short title for this group task",
        "description": "Detailed explanation of the group task objective",
        "data_collection_task": "Specific instructions for gathering data needed for this group task",
        "code_executor_task": "Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK."
    }}
    """
    return template.format(task=task, slide_content=slide_content)


def get_new_group_description_prompt(
    *,
    task: str,
    groups_in_plan: list[str],
    previous_group_summary: str,
    group_description: str,
) -> str:
    """Generates a prompt template for new group description.

    Args:
        task: The current task
        groups_in_plan: The groups in the plan
        previous_group_summary: The summary of the previous group
        group_description: The description of the current group
    """
    template = """
    The context of the step is as follows:
    
    Recall that you are working on the following request:

    ## User Query:
    {task}

    There is a confirmed plan for solving the request, which contains the following groups: \n
    {groups_in_plan_str}

    You are currently focusing on the following group: \n
    {group_description}
    """.format(
        group_description=group_description,
        groups_in_plan_str="\n".join(groups_in_plan),
        task=task,
    )
    if len(previous_group_summary) > 0:
        template += f"""
    So far, you have completed the following groups: \n
    {previous_group_summary}
    """
    return template


def get_code_executor_prompt() -> str:
    """system prompt for code executor"""
    template = """
You are a Code Execution Agent. Given any user request, generate only the most accurate and robust Python or Bash code—in the correct code block—to fulfill the task.  
**REMEMBER: If the request involves creating or appending to a file, you MUST use Python code.**  
Think step by step, and only return the code block in markdown format.
    """
    return template


def get_code_executor_prompt_cn() -> str:
    """系统提示词，用于代码执行器"""
    template = """
    您是一个代码执行代理。面对任何用户请求，请在正确的代码块中生成最准确和健壮的 Python 或 Bash 代码来完成任务。
    **请记住：如果请求涉及创建或追加文件，你必须使用 Python 代码，并且使用中文作为文件名。**
    请逐步思考，只返回代码块，使用 markdown 格式。
    """
    return template


def get_domain_specific_agent_prompt() -> str:
    """system prompt for domain specific agent"""
    return "You are a prompt expert that selects the most appropriate prompt template for different domains."


def get_domain_specific_agent_prompt_cn() -> str:
    """系统提示词，用于领域专用代理"""
    return "您是一个提示词专家，为不同领域挑选最合适的提示词模板。"


def get_general_agent_prompt() -> str:
    """system prompt for general agent"""
    return "You are a general AI assistant that provides answer for simple questions."


def get_general_agent_prompt_cn() -> str:
    """系统提示词，用于通用代理"""
    return "您是一个通用AI助手，为简单问题提供答案。回答使用中文。"


def get_rag_agent_prompt() -> str:
    """system prompt for rag agent"""
    return "You are a information retrieval agent that provides relevant information from the internal database."


def get_rag_agent_prompt_cn() -> str:
    """系统提示词，用于信息检索代理"""
    return "您是一个信息检索代理，从内部数据库中提供相关信息。"
