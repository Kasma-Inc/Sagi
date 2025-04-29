# Planning Workflow

The Planning Workflow is designed to support complex problem-solving through multi-step reasoning, reflection, and tool integration. This workflow enables agents to leverage various capabilities including file retrieval, web search, code execution, and more to accomplish tasks automatically.

## Components
In general, there are two stages for each task: plan generation stage and plan execution stage. Meanwhile, we manage the plan status and the plan history by a plan manager. 

- **Plan Generation**
    - **Domain-specific-plan-template-selector**:
        An agent that selects the appropriate plan template from the predefined templates if the task is recognized as included in the predefined categories. This agent is useful when want to impose domain-specific knowledge to the plan generation by prompt-based method.
    - **Fact-analyzer**:
        An agent that analyzes the task to categorize facts into:
        - Facts explicitly stated in the task
        - Facts already known by the agent
        - Facts that need to be searched/retrieved from the local knowledge base or internet
        - Facts that need to be derived from existing information
    - **Planner**:
        An agent that can generate a plan for the task which contains multiple steps.
    - **Human-in-the-loop**:
        A human can review the generated plan and provide feedback or confirm the plan.
- **Plan Execution**
    - **Triage Agent**:
        An agent that can triage the task into different categories and select the appropriate agent to execute the task and give the instruction to the agent.
    - **Reflection Agent**:
        An agent that can reflect the plan and the execution status to adjust the plan.
    - **Tool Agent**:
        - **File Retrieval**:
            An agent that can retrieve the file from the local file system.
        - **Code Executor**:
            An agent that can generate the code and execute the code.
        - **Web Search**:
            An agent that can search the web.
- **Plan Manager**:
    The Plan Manager is a crucial component that maintains the state and history of all plans within a chat session, and retrieval the plan history for the multi-round conversation:

    - **Multi-Plan Management**: Tracks multiple plans corresponding to different user requests in a multi-round conversation within a chat session.
    - **Plan State Tracking**: Maintains the current state of each plan (pending, in_progress, completed, failed).
    - **Step Management**: For each plan, tracks multiple steps, including the step description, the step status, the messages associated with the step, and the step result.

The workflow architecture is illustrated in the diagram below:

![workflow_diagram](../../assets/planning_workflow.png)



## References

This implementation is built on several excellent projects:

- [AutoGen](https://github.com/microsoft/autogen/tree/main) 
- [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/servers) 
- [MagneticOne](https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/magentic-one.html)
- [Manus](https://manus.im/)