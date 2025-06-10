from mcp.server.fastmcp import FastMCP
from prompt_template import (
    FINANCIAL_PPT_FACTS_PROMPT,
    FINANCIAL_PPT_FACTS_PROMPT_CN,
    FINANCIAL_PPT_PLAN_PROMPT,
    FINANCIAL_PPT_PLAN_PROMPT_CN,
    GENERAL_FACTS_PROMPT,
    GENERAL_FACTS_PROMPT_CN,
    GENERAL_PLAN_PROMPT_CN,
    GENERAL_PPT_PLAN_PROMPT,
    GENERAL_PPT_PLAN_PROMPT_CN,
    GENERAL_REPORT_PLAN_PROMPT,
    GENERAL_REPORT_PLAN_PROMPT_CN,
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
    "general-report": {
        "facts_prompt": GENERAL_FACTS_PROMPT,
        "plan_prompt": GENERAL_REPORT_PLAN_PROMPT,
    },
}


PROMPT_TEMPLATES_CN = {
    "general": {
        "facts_prompt": GENERAL_FACTS_PROMPT_CN,
        "plan_prompt": GENERAL_PLAN_PROMPT_CN,
    },
    "financial-ppt": {
        "facts_prompt": FINANCIAL_PPT_FACTS_PROMPT_CN,
        "plan_prompt": FINANCIAL_PPT_PLAN_PROMPT_CN,
    },
    "general-ppt": {
        "facts_prompt": GENERAL_FACTS_PROMPT_CN,
        "plan_prompt": GENERAL_PPT_PLAN_PROMPT_CN,
    },
    "general-report": {
        "facts_prompt": GENERAL_FACTS_PROMPT_CN,
        "plan_prompt": GENERAL_REPORT_PLAN_PROMPT_CN,
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


@mcp.tool()
async def get_report_plan() -> str:
    """
    Get General Report prompt templates for creating brilliant reports.
    """
    return PROMPT_TEMPLATES["general-report"]


@mcp.tool()
async def get_general_prompts_cn() -> dict:
    """
    模式: 中文
    获取用于结构化思维和问题解决的通用提示模板。
    如果没有找到特定的提示模板，请使用此通用模板。
    """
    return PROMPT_TEMPLATES_CN["general"]


@mcp.tool()
async def get_financial_ppt_prompts_cn() -> dict:
    """
    模式: 中文
    获取用于创建金融相关精彩演示文稿的金融PowerPoint(PPT)提示模板。
    """
    return PROMPT_TEMPLATES_CN["financial-ppt"]


@mcp.tool()
async def get_ppt_plan_cn() -> str:
    """
    模式: 中文
    获取用于创建精彩演示文稿的通用PowerPoint(PPT)提示模板。
    """
    return PROMPT_TEMPLATES_CN["general-ppt"]


@mcp.tool()
async def get_report_plan_cn() -> str:
    """
    模式: 中文
    获取用于创建精彩报告的通用报告提示模板。
    """
    return PROMPT_TEMPLATES_CN["general-report"]


def main() -> None:
    """Run the MCP server."""
    try:
        mcp.run()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
