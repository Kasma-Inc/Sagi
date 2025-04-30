from mcp.server.fastmcp import FastMCP
from prompt_template import (
    FINANCIAL_PPT_FACTS_PROMPT,
    FINANCIAL_PPT_PLAN_PROMPT,
    GENERAL_FACTS_PROMPT,
    GENERAL_PPT_PLAN_PROMPT,
    GNERAL_PLAN_PROMPT,
)

# Create an MCP server
mcp = FastMCP("domain_specific_mcp")

# Define our domain-specific prompt dictionaries
PROMPT_TEMPLATES = {
    "general": {
        "facts_prompt": GENERAL_FACTS_PROMPT,
        "plan_prompt": GNERAL_PLAN_PROMPT,
    },
    "financial-ppt": {
        "facts_prompt": FINANCIAL_PPT_FACTS_PROMPT,
        "plan_prompt": FINANCIAL_PPT_PLAN_PROMPT,
    },
    "general-ppt": {
        "facts_prompt": GENERAL_FACTS_PROMPT,
        "plan_prompt": GENERAL_PPT_PLAN_PROMPT,
    },
}


# Tool to get general-purpose prompts
@mcp.tool()
async def get_general_prompts() -> dict:
    """
    Get general-purpose prompt templates for structured thinking and problem-solving.
    If no specific prompt template is found, use this general template.
    """
    return PROMPT_TEMPLATES["general"]


# Tool to get PowerPoint presentation prompts
@mcp.tool()
async def get_financial_ppt_prompts() -> dict:
    """
    Get Financial PowerPoint(PPT) prompt templates for creating finance-related brilliant presentations.
    """
    return PROMPT_TEMPLATES["financial-ppt"]


@mcp.tool()
async def get_ppt_plan() -> str:
    """
    Get General PowerPoint(PPT) prompt templates for creating brilliant presentations.
    """
    return PROMPT_TEMPLATES["general-ppt"]


def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
