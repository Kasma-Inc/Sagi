# Domain-Specific MCP Server

Python server implementing Model Context Protocol (MCP) for domain-specific prompt templates that enhance planning stage.

## Features

- Provides specialized planning prompt templates for different domains
- Easily extensible for additional domains

## API

### Tools

- **get_general_prompts**
  - Get general-purpose prompt templates (default)
  - Returns:
    - `facts_prompt`: Template for gathering facts and information
    - `plan_prompt`: Template for creating a structured plan

- **get_ppt_prompts**
  - Get PowerPoint (PPT) specific prompt templates for creating brilliant presentations
  - Returns:
    - `facts_prompt`: Template specialized for gathering facts and information to generate PPT
    - `plan_prompt`: Template for creating a specialized plan to generate PPT

## Configuration

```json
{
  "mcpServers": {
    "domain_specific": {
      "command": "uv",
      "args": [
        "--directory",
        "mcp_server/domain_specific_mcp",
        "run",
        "python",
        "src/domain_specific_mcp/server.py"
      ]
    }
  }
}
```

## Extending

To add new domain-specific templates:

1. Define new prompt templates in [`src/domain_specific_mcp/prompt_template.py`](src/domain_specific_mcp/prompt_template.py )
2. Add them to the [`PROMPT_TEMPLATES`](src/domain_specific_mcp/server.py ) dictionary in [`src/domain_specific_mcp/server.py`](src/domain_specific_mcp/server.py )
3. Create a new tool function that returns the specific template
4. Test with the script [`src/domain_specific_mcp/test.py`](src/domain_specific_mcp/test.py ) (details mentioned below)

> **Important Note**: When adding new prompt templates, avoid using single quotes (`'`) or double quotes (`"`) in the prompt text. These characters can cause errors during regex extraction. If quotes are necessary, consider using alternative characters or escaping them properly.

## Test
The [`test.py`](src/domain_specific_mcp/test.py) script demonstrates how to test the domain-specific MCP server, which integrate domain-specific templates with an agent to generate structured responses for specific use cases.

### Customize Test Queries

Modify the user query by editing `line 75` the [`test.py`](src/domain_specific_mcp/test.py) script:

```python
agent_response = await domain_specific_agent.on_messages(
    [TextMessage(content="hello, world", source="tester")],
    CancellationToken(),
)
```

Simply replace `hello, world` with your desired query to test different domain-specific templates.

### Run the test
Simply run the script:
```bash
python mcp_server/domain_specific_mcp/src/domain_specific_mcp/test.py
```

### View Test Result
Test outputs are logged to `test.log` in the same directory as the script.
