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

            - Who should speak next? (select from: {names})
            - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and include any specific information they may need)

        Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

            {{
                "next_speaker": {{
                    "instruction": string,
                    "answer": string (select from: {names})
                }},
            }}
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
