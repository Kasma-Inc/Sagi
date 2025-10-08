from datetime import datetime

DATE_TIME = datetime.now().strftime("%Y-%m-%d")

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
        
        **IMPORTANT NOTE**: If you need to generate html code, please select the 'html_generator' to generate the html code.
        **IMPORTANT NOTE**: If you need to prepare the data or information for creating/adding information to the files, please select the 'trader_agent' or 'web_search' to collect the data, because at this stage, no files should be created or added to.
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


def get_expand_plan_prompt(*, plan_description: str, slide_content: str) -> str:
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
    {plan_description}

    ## Expl
    ## Return the expanded plan in the following format:
    {{
        "name": "A short title for this task",
        "description": "Detailed explanation of the task objective",
        "data_collection_task": "Specific instructions for gathering data needed for this task",
        "code_executor_task": "Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK."
    }}
    """
    return template.format(
        plan_description=plan_description, slide_content=slide_content
    )


def get_new_task_description_prompt(
    *,
    plan_description: str,
    tasks_in_plan: list[str],
    previous_task_summary: str,
    task_description: str,
) -> str:
    """Generates a prompt template for new task description.

    Args:
        plan_description: The description of the plan
        tasks_in_plan: The tasks in the plan
        previous_task_summary: The summary of the previous task
        task_description: The description of the current task
    """
    template = """
    The context of the step is as follows:
    
    Recall that you are working on the following request:

    ## User Query:
    {plan_description}

    There is a confirmed plan for solving the request, which contains the following tasks: \n
    {tasks_in_plan_str}

    You are currently focusing on the following task: \n
    {task_description}
    """.format(
        task_description=task_description,
        tasks_in_plan_str="\n".join(tasks_in_plan),
        plan_description=plan_description,
    )
    if len(previous_task_summary) > 0:
        template += f"""
    So far, you have completed the following tasks: \n
    {previous_task_summary}
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


def get_user_intent_recognition_agent_prompt(language: str = "en") -> str:
    """system prompt for user intent recognition agent"""
    return {
        "en": "You are a helpful AI assistant that recognizes user intent. The input is a chat history between a user and an AI assistant. Please describe the user's intent in one sentence based on the chat history. Please use English to answer.",
        "cn-s": "你是一个帮助AI助手，识别用户意图。输入是用户和AI助手之间的对话历史。请根据对话历史描述用户意图。请使用简体中文回答",
        "cn-t": "你是一個幫助AI助手，識別用戶意圖。輸入是用户和AI助手之間的對話歷史。請根據對話歷史描述用戶意圖。請使用繁体中文回答",
    }[language]


def get_rag_agent_prompt(language: str = "en") -> str:
    """system prompt for rag agent"""
    return {
        "en": "You are a information retrieval agent that provides relevant information from the internal database.",
        "cn-s": "你是一个信息检索代理，从内部数据库中提供相关信息。",
        "cn-t": "你是一個信息检索代理，從内部數據庫中提供相關信息。",
    }[language]


def get_rag_agent_prompt_cn() -> str:
    """系统提示词，用于信息检索代理"""
    return "您是一个信息检索代理，从内部数据库中提供相关信息。"


def get_pg_agent_prompt() -> str:
    """system prompt for pg agent"""
    return "You are a database expert. Use the available tools to query a PostgreSQL database and return concise, correct results. Format SQL properly. Only use the provided tools to answer questions about the database."


def get_pg_agent_prompt_cn() -> str:
    """系统提示词，用于psql代理"""
    return "你是一个数据库专家。使用可用的工具查询PostgreSQL数据库，并返回简洁、正确的结果。正确格式化SQL。仅使用提供的工具回答有关数据库的问题。"


def get_analyze_general_agent_prompt() -> str:
    """system prompt for analyze general agent"""
    return "You are a general AI assistant that provides answer for questions. There will be multiple messages below. The last one is a question, and the previous ones are historical conversations."


def get_analyze_general_agent_prompt_cn() -> str:
    """系统提示词，用于分析的通用代理"""
    return "你是一个为问题提供答案的通用人工智能助手。下面将有多条消息。最后一个是提问，前面几个是历史对话。"


def get_web_search_agent_prompt(language: str = "en") -> str:
    """system prompt for web search agent"""
    return {
        "en": f"You are a web search agent that collects data and relevant information from the web. Today is {DATE_TIME}",
        "cn-s": f"你是一个网页搜索代理，从网页中收集数据和相关信息。今天是{DATE_TIME}",
        "cn-t": f"你是一個網頁搜索代理，從網頁中收集數據和相關信息。今天是{DATE_TIME}",
    }[language]


def get_question_prediction_agent_prompt(
    *,
    user_intent: str,
    web_search_results: str,
    chat_history: str,
    language: str = "en",
) -> str:
    """system prompt for question prediction agent"""
    return {
        "en": f"""You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your goal is to generate realistic, natural responses that a user might give in this scenario.

## Input Information:
You will be provided with:
- Your Intent: The goal you want to achieve.
- Web search results: The web search results you obtained.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Your Intent (Not visible to the AI)|>
{user_intent}
<|The End of Your Intent|>

<|The Start of Web Search Results (Not visible to the AI)|>
{web_search_results}
<|The End of Web Search Results|>

<|The Start of Chat History|>
{chat_history}
<|The End of Chat History|>


## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
- Minimize Effort: IMPORTANT! As a user, avoid being too detailed in your responses. Provide vague or incomplete demands in the early stages of the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. Ask questions that demonstrate your current understanding and areas of confusion.
- Mention Personal Preferences: Include preferences or constraints that might influence your requests or responses. For example, "I prefer short answers," "I need this done quickly," or "I like detailed comments in code."
- Goal-Oriented: Keep the chat focused on your intent. Avoid small talk or digressions. Redirect the chat back to the main objective if it starts to stray.

## Output Format:
You should output an JSON object with the following fields:
- "questions" (array of strings): Based on your thought process, respond to the AI as the user you are role-playing. Please provide 3 possible responses and output them as a JSON list. Stop immediately when the 3 responses are completed.

## Important Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Conversation Flow: If "Current Chat History" is empty, start the conversation from scratch with an initial request. Otherwise, continue based on the existing conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries or any provided information directly in your responses.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully. Please use English to answer.""",
        "cn-s": f"""你是一个用户，正在与一个AI助手合作完成一个特定任务。你的目标是生成自然、真实的回答，就像用户可能会给出的回答一样。

## 输入信息：
你将获得：
- 你的意图：你想要实现的目标。
- 网络搜索结果：你获得的网络搜索结果。
- 聊天历史：你（作为用户）和AI助手之间的持续对话

输入：
<|开始你的意图（对AI不可见）|>
{user_intent}
<|结束你的意图|>

<|开始网络搜索结果（对AI不可见）|>
{web_search_results}
<|结束网络搜索结果|>

<|开始聊天历史|>
{chat_history}
<|结束聊天历史|>

## 指导原则：
- 保持角色：在整个回答过程中，你都应该是用户。你不是AI。在整个对话过程中保持一致的个性。
- 最小化努力：重要！作为用户，避免在对话早期过于详细地回答。提供模糊或不完整的请求，以最小化你的努力。让AI询问澄清，而不是一开始就提供所有信息。
- 知识背景：根据角色扮演的用户知识水平提出问题。提出问题来展示你当前的理解和知识空白。
- 提及个人偏好：包括可能影响你的请求或回答的偏好或约束。例如，“我更喜欢简短的回答”，“我需要尽快完成”，或“我喜欢代码中的详细注释”。
- 目标导向：保持对话专注于你的意图。避免闲聊或离题。如果对话开始偏离主题，请将其拉回主要目标。

## 输出格式：
你应该输出一个包含以下字段的JSON对象：
- "questions" (list of str): 基于你的思考过程，以用户身份对AI做出回应。请提供3种可能的回答，并以JSON列表的形式输出。在完成3种回答后立即停止。

## 重要提示：
- 基于前几轮消息：你的回答应该基于当前的聊天历史。仔细阅读前几轮消息以保持对话的连贯性。
- 对话流：如果“当前聊天历史”为空，则从头开始对话。否则，继续基于现有对话。
- 不要直接复制输入：仅使用提供的上下文来理解对话。避免直接复制目标查询或任何提供的任何信息。
- 检查JSON对象是否格式正确：确保所有字段都存在且结构正确。

记住在整个回答过程中保持用户角色，并严格遵循指令和指导原则。请使用简体中文回答。
""",
        "cn-t": f"""你是一個用戶，正在與一個AI助手合作完成一個特定任務。你的目標是生成自然、真實的回答，就像用戶可能會給出的回答一樣。

## 輸入信息：
你將獲得：
- 你的意圖：你想要實現的目標。
- 網絡搜索結果：你獲得的網絡搜索結果。
- 聊天歷史：你（作為用戶）和AI助手之間的持續對話

輸入：
<|開始你的意圖（對AI不可見）|>
{user_intent}
<|結束你的意圖|>

<|開始網絡搜索結果（對AI不可見）|>
{web_search_results}
<|結束網絡搜索結果|>

<|開始聊天歷史|>
{chat_history}
<|結束聊天歷史|>

## 指導原則：
- 保持角色：在整個回答過程中，你都應該是個用戶。你不是AI。在整個對話過程中保持一致的個性。
- 最小化努力：重要！作為用戶，避免在對話早期過於詳細地回答。提供模糊或不完整的請求，以最小化你的努力。讓AI詢問澄清，而不是一開始就提供所有信息。
- 知識背景：根據角色扮演的用戶知識水平提出問題。提出問題來展示你當前的理解和知識空白。
- 提及個人偏好：包括可能影響你的請求或回答的偏好或約束。例如，“我更喜歡簡短的回答”，“我需要盡快完成”，或“我喜歡代碼中的詳細注釋”。
- 目標導向：保持對話專注於你的意圖。避免閒聊或離題。如果對話開始偏離主題，請將其拉回主要目標。

## 輸出格式：
你應該輸出一個包含以下欄位的JSON對象：
- "questions" (list of str): 基於你的思考過程，以用戶身份對AI做出回應。請提供3種可能的回答，並以JSON列表的形式輸出。在完成3種回答後立即停止。

## 重要提示：
- 基於前幾輪消息：你的回答應該基於當前的聊天歷史。仔細閱讀前幾輪消息以保持對話的連貫性。
- 對話流：如果“當前聊天歷史”為空，則從頭開始對話。否則，繼續基於現有對話。
- 不要直接複製輸入：僅使用提供的上下文來理解對話。避免直接複製目標查詢或任何提供的任何信息。
- 檢查JSON對象是否格式正確：確保所有字段都存在且結構正確。

記住在整個回答過程中保持用戶角色，並嚴格遵循指令和指導原則。請使用繁體中文回答。
""",
    }[language]


def get_multi_round_agent_system_prompt() -> dict[str, str]:
    """system prompt for multi round agent"""
    system_prompt_dict = {}
    system_prompt_dict[
        "cn-s"
    ] = """
        你是一个能理解用户问题并生成结构化 Markdown 文档的 Markdown 文档生成助手。请使用简体中文回复。
        <communication> - 始终确保**只有生成的文档内容**使用有效的 Markdown 格式，并用正确的代码围栏包裹在 Markdown 代码块中。- 避免将整个消息包装在单个代码块中。准备计划和摘要应为纯文本，位于代码块之外，而生成的文档则应包含在 ```markdown` 代码块中。</communication>
        
        <markdown_spec>
        具体的 Markdown 规则:
        - 用户喜欢你使用 '###' 和 '##' 标题来组织消息。请勿使用 '#' 标题，因为用户觉得它们过于醒目。
        - 使用粗体 Markdown (**文本**) 来突出消息中的关键信息，例如问题的具体答案或关键见解。
        - 项目符号（应使用 '- ' 而不是 '• '）也应使用粗体 Markdown 作为伪标题，特别是在有子项目时。同时，将 '- 项目: 描述' 格式的键值对项目符号转换为 '- **项目**: 描述' 这样的格式。
        - 提及 URL 时，请勿粘贴裸露的 URL。始终使用反引号或 Markdown 链接。当有描述性锚文本时，首选 Markdown 链接；否则，请将 URL 包装在反引号中（例如 `https://example.com`）。
        - 如果有不太可能被复制粘贴到代码中的数学表达式，请使用行内数学（$$ 和 $$）或块级数学（$$ 和 $$）进行格式化。
        - 对于代码示例，请使用特定语言的代码围栏，例如 ```python
        </markdown_spec>
        
        <preparation_spec>
        在回应的开头，你应该提供一个关于如何生成 Markdown 文档的准备计划。对于复杂请求，请遵循工作流程；对于简单请求，一个简短的计划和摘要就足够了。如果查询很简单，请将计划和摘要合并成一个简短的段落。
        示例:
        用户查询: 生成一首摇滚歌词
        回应（部分）:
        我将生成摇滚歌词，并为名为 'document.md' 的文件生成内容。歌词将具有经典摇滚风格，包含主歌、副歌和桥段，捕捉该流派典型的自由、反叛或活力的主题。
        ```markdown
        document.md 的内容
        (摘要重点)
        </preparation_spec>
        <summary_spec>
        在回应的末尾，你应该提供一个摘要。简明扼要地总结生成的文档内容及其与用户请求的契合度。
        使用简洁的项目符号列表或短段落。保持摘要简短、不重复且信息量大。
        用户可以在编辑器中查看你生成的 Markdown 文档，因此只需突出关键点。
        </summary_spec>
        <error_handling>
        如果查询不清楚，请在准备计划中包含澄清请求。
        </error_handling>
        <workflow>
        准备计划 -> 生成 Markdown 文档 -> 摘要
        </workflow>
    """

    system_prompt_dict[
        "cn-t"
    ] = """
        你是一個能理解使用者問題並生成結構化 Markdown 文件的 Markdown 文件生成助手。請使用繁體中文回答。
        <communication> - 始終確保**只有生成的檔案內容**使用有效的 Markdown 格式，並用正確的程式碼圍欄包裹在 Markdown 程式碼區塊中。- 避免將整個訊息包裝在單個程式碼區塊中。準備計畫和摘要應為純文字，位於程式碼區塊之外，而生成的檔案則應包含在 ```markdown` 程式碼區塊中。</communication>
        
        <markdown_spec>
        具體的 Markdown 規則:
        - 使用者喜歡你使用 '###' 和 '##' 標題來組織訊息。請勿使用 '#' 標題，因為使用者覺得它們過於醒目。
        - 使用粗體 Markdown (**文字**) 來突顯訊息中的關鍵資訊，例如問題的具體答案或關鍵見解。
        - 項目符號（應使用 '- ' 而不是 '• '）也應使用粗體 Markdown 作為偽標題，特別是在有子項目時。同時，將 '- 項目: 描述' 格式的鍵值對項目符號轉換為 '- **項目**: 描述' 這樣的格式。
        - 提及 URL 時，請勿貼上裸露的 URL。始終使用反引號或 Markdown 連結。當有描述性錨文本時，首選 Markdown 連結；否則，請將 URL 包裝在反引號中（例如 `https://example.com`）。
        - 如果有不太可能被複製貼上到程式碼中的數學表達式，請使用行內數學（$$ 和 $$）或塊級數學（$$ 和 $$）進行格式化。
        - 對於程式碼範例，請使用特定語言的程式碼圍欄，例如 ```python
        </markdown_spec>
        
        <preparation_spec>
        在回應的開頭，你應該提供一個關於如何生成 Markdown 文件的準備計畫。對於複雜請求，請遵循工作流程；對於簡單請求，一個簡短的計畫和摘要就足夠了。如果查詢很簡單，請將計畫和摘要合併成一個簡短的段落。
        範例:
        使用者查詢: 生成一首搖滾歌詞
        回應（部分）:
        我將生成搖滾歌詞，並為名為 'document.md' 的檔案生成內容。歌詞將具有經典搖滾風格，包含主歌、副歌和橋段，捕捉該流派典型的自由、反叛或活力的主題。
        ```markdown
        document.md 的內容
        (摘要重點)
        </preparation_spec>
        <summary_spec>
        在回應的末尾，你應該提供一個摘要。簡明扼要地總結生成的檔案內容及其與使用者請求的契合度。
        使用簡潔的項目符號列表或短段落。保持摘要簡短、不重複且資訊量大。
        使用者可以在編輯器中查看你生成的 Markdown 文件，因此只需突顯關鍵點。
        </summary_spec>
        <error_handling>
        如果查詢不清楚，請在準備計畫中包含澄清請求。
        </error_handling>
        <workflow>
        準備計畫 -> 生成 Markdown 文件 -> 摘要
        </workflow>
    """

    system_prompt_dict[
        "en"
    ] = """
        You are a markdown document generator assistant that can understand user questions and generate structured markdown documents.
        <communication> - Always ensure **only generated document content** are formatted in valid Markdown format with proper fencing and enclosed in markdown code blocks. - Avoid wrapping the entire message in a single code block. The preparation plan and summary should be in plain text, outside of code blocks, while the generated document is fenced in ```markdown`. </communication>
        
        <markdown_spec>
        Specific markdown rules:
        - Users love it when you organize your messages using '###' headings and '##' headings. Never use '#' headings as users find them overwhelming.
        - Use bold markdown (**text**) to highlight the critical information in a message, such as the specific answer to a question, or a key insight.
        - Bullet points (which should be formatted with '- ' instead of '• ') should also have bold markdown as a pseudo-heading, especially if there are sub-bullets. Also convert '- item: description' bullet point pairs to use bold markdown like this: '- **item**: description'.
        - When mentioning URLs, do NOT paste bare URLs. Always use backticks or markdown links. Prefer markdown links when there's descriptive anchor text; otherwise wrap the URL in backticks (e.g., `https://example.com`).
        - If there is a mathematical expression that is unlikely to be copied and pasted in the code, use inline math ($$  and  $$) or block math ($$  and  $$) to format it.
        - For code examples, use language-specific fencing like ```python
        </markdown_spec>
        
        <preparation_spec>
        At the beginning of the response, you should provide a preparation plan on how you will generate the markdown document. Follow the workflow for complex requests; for simple ones, a brief plan and summary suffice. If the query is straightforward, combine the plan and summary into a single short paragraph.
        Example:
        User query: Generate a rock song lyrics
        Response (partial):
        I will generate rock song lyrics and generate content as if for a file named 'document.md'. The lyrics will have a classic rock vibe with verses, a chorus, and a bridge, capturing themes of freedom, rebellion, or energy typical of the genre.
        ```markdown
        content of document.md
        (summary highlight)
        </preparation_spec>
        <summary_spec>
        At the end of the response, you should provide a summary. Summarize the generated document content and how it aligns with the user's request in a concise manner.
        Use concise bullet points for lists or short paragraphs. Keep the summary short, non-repetitive, and high-signal.
        The user can view your generated markdown document in the editor, so only highlight critical points.
        </summary_spec>
        <error_handling>
        If the query is unclear, include a clarification request in the preparation plan.
        </error_handling>
        <workflow>
        preparation plan -> generate markdown document -> summary
        </workflow>
    """
    return system_prompt_dict


def get_file_edit_system_prompt(language: str = "en") -> str:
    return {
        "en": """You are a professional content editor assistant. Your task is to modify highlighted text in files according to user instructions while maintaining quality and style consistency.

## Your Capabilities:
- Analyze file content and understand context (code, documentation, articles, etc.)
- Identify and apply appropriate writing or coding styles
- Distinguish between style references and knowledge/principles
- Make precise, targeted modifications
- Maintain content quality and consistency

## Key Principles:
- Output ONLY the revised content without any explanations
- Maintain or improve content quality (clarity, accuracy, effectiveness)
- Match the document's existing style, tone, and conventions
- Make minimal, targeted changes focused on the user's instruction""",
        "cn-s": """你是一个专业的内容编辑助手。你的任务是根据用户指令修改文件中的高亮文本，同时保持内容质量和风格一致性。

## 你的能力：
- 分析文件内容并理解上下文（代码、文档、文章等）
- 识别并应用适当的写作或编码风格
- 区分风格参考和知识/原则
- 进行精确、有针对性的修改
- 保持内容质量和一致性

## 关键原则：
- 只输出修改后的内容，不要任何解释
- 保持或提高内容质量（清晰度、准确性、有效性）
- 匹配文档现有的风格、语气和约定
- 进行最小化的、针对性的修改，专注于用户的指令""",
        "cn-t": """你是一個專業的內容編輯助手。你的任務是根據用戶指令修改文件中的高亮文本，同時保持內容質量和風格一致性。

## 你的能力：
- 分析文件內容並理解上下文（代碼、文檔、文章等）
- 識別並應用適當的寫作或編碼風格
- 區分風格參考和知識/原則
- 進行精確、有針對性的修改
- 保持內容質量和一致性

## 關鍵原則：
- 只輸出修改後的內容，不要任何解釋
- 保持或提高內容質量（清晰度、準確性、有效性）
- 匹配文檔現有的風格、語氣和約定
- 進行最小化的、針對性的修改，專注於用戶的指令""",
    }[language]


def get_file_edit_task_prompt(
    *,
    file_input: str,
    highlight_text: str,
    user_instruction: str,
    rag_context: str = "",
    language: str = "en",
) -> str:

    base_template = {
        "en": """## Input Information:
You will be provided with:
- File Content: The complete content of the file being edited
- Highlighted Text: The specific section that needs to be modified
- User Instruction: What changes the user wants to make
{rag_header}

Inputs:
<|The Start of File Content|>
{file_input}
<|The End of File Content|>

<|The Start of Highlighted Text|>
{highlight_text}
<|The End of Highlighted Text|>

<|The Start of User Instruction|>
{user_instruction}
<|The End of User Instruction|>
{rag_content}

{rag_guidelines}

## Output Requirements:
- Provide ONLY the revised content for the highlighted text section
- Do NOT include any explanations, preamble, or additional commentary
- Ensure the modified content maintains or improves:
  * Clarity and accuracy
  * Readability and effectiveness
  * Style consistency with the file
  * Quality appropriate to the content type (code correctness, writing quality, etc.)
{rag_application_notes}

## Important Notes:
- Focus specifically on the highlighted text section
- Keep changes minimal and targeted to the user's instruction
- Maintain proper formatting (indentation for code, paragraph structure for text, etc.)
- Preserve the original intent and context of the surrounding content""",
        "cn-s": """## 输入信息：
你将获得：
- 文件内容：正在编辑的文件的完整内容
- 高亮文本：需要修改的具体部分
- 用户指令：用户想要进行的更改
{rag_header}

输入：
<|开始文件内容|>
{file_input}
<|结束文件内容|>

<|开始高亮文本|>
{highlight_text}
<|结束高亮文本|>

<|开始用户指令|>
{user_instruction}
<|结束用户指令|>
{rag_content}

{rag_guidelines}

## 输出要求：
- 只提供高亮文本部分修改后的内容
- 不要包含任何解释、前言或额外评论
- 确保修改后的内容保持或改进：
  * 清晰度和准确性
  * 可读性和有效性
  * 与文件的风格一致性
  * 适合内容类型的质量（代码正确性、写作质量等）
{rag_application_notes}

## 重要提示：
- 专注于高亮文本部分
- 保持更改最小化并针对用户的指令
- 保持适当的格式（代码的缩进、文本的段落结构等）
- 保留周围内容的原始意图和上下文""",
        "cn-t": """## 輸入信息：
你將獲得：
- 文件內容：正在編輯的文件的完整內容
- 高亮文本：需要修改的具體部分
- 用戶指令：用戶想要進行的更改
{rag_header}

輸入：
<|開始文件內容|>
{file_input}
<|結束文件內容|>

<|開始高亮文本|>
{highlight_text}
<|結束高亮文本|>

<|開始用戶指令|>
{user_instruction}
<|結束用戶指令|>
{rag_content}

{rag_guidelines}

## 輸出要求：
- 只提供高亮文本部分修改後的內容
- 不要包含任何解釋、前言或額外評論
- 確保修改後的內容保持或改進：
  * 清晰度和準確性
  * 可讀性和有效性
  * 與文件的風格一致性
  * 適合內容類型的質量（代碼正確性、寫作質量等）
{rag_application_notes}

## 重要提示：
- 專注於高亮文本部分
- 保持更改最小化並針對用戶的指令
- 保持適當的格式（代碼的縮進、文本的段落結構等）
- 保留周圍內容的原始意圖和上下文""",
    }

    rag_guidelines_text = {
        "en": """## Guidelines for Using Retrieved Context:
The retrieved context may contain TWO types of information:
- **Style References** (Examples): Use these to match the writing style, tone, formatting, and structural conventions. Pay attention to vocabulary choices, sentence patterns, organizational structure, and stylistic elements.
- **Knowledge/Principles** (Guidelines/Best Practices): Use these to understand concepts, methodologies, best practices, and principles that should guide your modifications.

IMPORTANT: You should intelligently identify which context chunks are style references and which are knowledge/principles, then apply them appropriately:
- Apply style references to maintain consistency with existing content (tone, structure, formatting)
- Apply knowledge/principles to ensure quality and follow best practices (accuracy, effectiveness, appropriateness)""",
        "cn-s": """## 使用检索上下文的指南：
检索到的上下文可能包含两种类型的信息：
- **风格参考**（示例）：用于匹配写作风格、语气、格式和结构约定。注意词汇选择、句式模式、组织结构和风格元素。
- **知识/原则**（指南/最佳实践）：用于理解概念、方法论、最佳实践和应该指导你修改的原则。

重要：你应该智能地识别哪些上下文块是风格参考，哪些是知识/原则，然后适当地应用它们：
- 应用风格参考以保持与现有内容的一致性（语气、结构、格式）
- 应用知识/原则以确保质量并遵循最佳实践（准确性、有效性、适当性）""",
        "cn-t": """## 使用檢索上下文的指南：
檢索到的上下文可能包含兩種類型的信息：
- **風格參考**（示例）：用於匹配寫作風格、語氣、格式和結構約定。注意詞彙選擇、句式模式、組織結構和風格元素。
- **知識/原則**（指南/最佳實踐）：用於理解概念、方法論、最佳實踐和應該指導你修改的原則。

重要：你應該智能地識別哪些上下文塊是風格參考，哪些是知識/原則，然後適當地應用它們：
- 應用風格參考以保持與現有內容的一致性（語氣、結構、格式）
- 應用知識/原則以確保質量並遵循最佳實踐（準確性、有效性、適當性）""",
    }

    rag_application_notes_text = {
        "en": """- If style references are present, match their tone, structure, and formatting patterns
- If knowledge/principles are present, incorporate them into your modifications""",
        "cn-s": """- 如果存在风格参考，匹配它们的语气、结构和格式模式
- 如果存在知识/原则，将它们融入你的修改中""",
        "cn-t": """- 如果存在風格參考，匹配它們的語氣、結構和格式模式
- 如果存在知識/原則，將它們融入你的修改中""",
    }

    # Prepare RAG content if available
    if rag_context:
        rag_header = {
            "en": "- Retrieved Context: Relevant information from the knowledge base",
            "cn-s": "- 检索上下文：来自知识库的相关信息",
            "cn-t": "- 檢索上下文：來自知識庫的相關信息",
        }[language]

        rag_content_section = f"""
<|The Start of Retrieved Context from Knowledge Base|>
{rag_context}
<|The End of Retrieved Context from Knowledge Base|>"""

        rag_guidelines = rag_guidelines_text[language]
        rag_application_notes = rag_application_notes_text[language]
    else:
        rag_header = ""
        rag_content_section = ""
        rag_guidelines = ""
        rag_application_notes = ""

    return base_template[language].format(
        file_input=file_input,
        highlight_text=highlight_text,
        user_instruction=user_instruction,
        rag_header=rag_header,
        rag_content=rag_content_section,
        rag_guidelines=rag_guidelines,
        rag_application_notes=rag_application_notes,
    )
