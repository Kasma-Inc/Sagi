GENERAL_FACTS_PROMPT = """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that facts will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""


GNERAL_PLAN_PROMPT = """Fantastic. To address this request we have assembled the following team:

{team}

USER QUERY: {task}

You are a professional planning assistant. 
Based on the team composition, user query, and known and unknown facts, please devise a plan for addressing the USER QUERY. Remember, there is no requirement to involve all team members -- a team member particular expertise may not be needed for this task.

Each plan group should contain the following elements:
1. name: A short title for this group task
2. description: Detailed explanation of the group objective.
3. data_collection_task: Specific instructions for gathering data needed for this group task (optional)
4. code_executor_task: Description of what code executor should do, JUST DETAILED DESCRIPTION IS OK, NOT ACTUAL CODE BLOCK.(optional)
"""


FINANCIAL_PPT_FACTS_PROMPT = """Below I will present you a request related to creating a financial PowerPoint presentation. Before we begin addressing the request, please answer the following pre-survey to the best of your ability. Keep in mind that you possess the knowledge of an expert financial analyst and strategist (e.g., CFA charterholder level) combined with rigorous analytical capabilities, so draw upon your deep understanding of finance, markets, and data interpretation. Here is the request:

{task}

Here is the pre-survey:
1. Please list any specific financial data, company names, reporting periods, target metrics, or constraints that are GIVEN directly in the request itself. It is possible that there are none.
2. Please list any specific financial data points, market information, economic indicators, regulatory details, or company-specific information that may need to be looked up. Specify potential authoritative sources (e.g., SEC filings like 10-K/10-Q, Bloomberg, Refinitiv Eikon, company Investor Relations websites, central bank databases, specific industry reports, named analyst research).
3. Please list any financial figures, ratios, forecasts, valuations, or analyses that may need to be derived (e.g., calculating financial ratios like P/E or ROE, performing a Discounted Cash Flow (DCF) analysis, running a sensitivity analysis, calculating CAGR, projecting future performance based on assumptions, consolidating data from multiple sources).
4. Please list any assumptions, qualitative assessments, market sentiments, or potential trends that are based on your financial expertise, recalled knowledge, or well-reasoned estimations (e.g., assumptions about future interest rates, estimations of market growth, qualitative assessment of management quality, potential M&A synergies based on industry knowledge).

When answering this survey, keep in mind that facts will typically be specific names, dates, financial figures, ratios, economic data points, etc. Your answer should use headings:
    
    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES / ASSUMPTIONS

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""

FINANCIAL_PPT_PLAN_PROMPT = """Excellent. To create the financial presentation outlined in the request, we have assembled the following team:

{team}

USER QUERY: {task}

You are a professional financial PPT creation planning assistant. 
Based on the team composition, user query, please generate a detailed financial PowerPoint presentation creation plan following these structural requirements:

Each plan group should contain the following elements:
1. name: A short title for this group task
2. description: Detailed explanation of the group objective and financial content
3. data_collection_task: Specific instructions for gathering financial data needed for this group task (optional)
4. code_executor_task: Description of what code should do to process data, GENERATE slide, APPEND it to PPT and SAVE PPT (optional)

Plan requirements:
- Each group task should GENERATE one slide, APPEND it to the PPT and SAVE PPT.
- Include common financial PPT elements such as titles, financial metrics, bullet point lists, visual elements (charts, graphs, financial tables, etc.)
- Code sections should primarily use Python (with visualization libraries like matplotlib, seaborn, plotly, pandas_datareader, etc.) or bash to install dependencies
- Ensure code is practical and executable, capable of completing financial data processing and slide generation tasks
- Focus on financial analysis, stock performance, company valuations, market trends, investment recommendations, and financial forecasts

Please output in JSON format, conforming to the following structure:

  groups: [
      name: Group Task Name,
      description: Group Task Description,
      data_collection_task: Data Collection Task Description,
      code_executor_task: Description of code task to be performed based on the collected data
    ,
    ...more groups
  ]

Example for a Tesla stock analysis presentation:
  groups: [
      name: Company Overview,
      description: Create a title slide with brief overview of Tesla, including its ticker symbol, industry, and founding date.,
      data_collection_task: Find Tesla ticker symbol (TSLA), founding date, headquarters location, and CEO information.,
      code_executor_task: Create a PPT and add a title slide with Tesla name, ticker symbol, and basic company information. Save the PPT for further additions.
    ,
    
      name: Stock Price Performance,
      description: Create a slide showing Tesla stock price performance over the past 5 years with key milestones highlighted.,
      data_collection_task: Retrieve Tesla 5-year historical stock price data from Yahoo Finance API and identify key corporate milestones.,
      code_executor_task: Use yfinance to download Tesla 5-year stock price data. Create a line chart showing the closing price over time. Mark significant company milestones on the chart. Add the chart to a new slide in the presentation with an appropriate title and axis labels. Remember to append the slide to PPT and save PPT.
    ,
      name: Financial Metrics,
      description: Create a slide analyzing Tesla key financial metrics including revenue growth, profit margins, P/E ratio, and EPS.,
      data_collection_task: Gather Tesla income statements, balance sheets, and key financial ratios for the past 3 years.,
      code_executor_task: Extract financial metrics from Tesla financial statements using yfinance. Create a formatted table showing Revenue, Net Income, Profit Margin, P/E Ratio, and EPS data for the past 3 years. Add the table to a new slide with an appropriate title and brief analysis text. Remember to append the slide to PPT and save PPT.
    ,
      name: Market Position & Competition,
      description: Create a slide analyzing Tesla market position, market share, and key competitors in the EV industry.,
      data_collection_task: Research Tesla market share in the EV industry and identify key competitors with their market shares.,
      code_executor_task: Create a pie chart showing the global EV market share distribution among Tesla and its key competitors. Add the chart to a new slide. Include a text box highlighting Tesla competitive advantages in the industry, such as brand recognition, battery technology, charging infrastructure, and software capabilities. Remember to append the slide to PPT and save PPT.
    ,
      name: Investment Recommendation,
      description: Create a slide with investment recommendation, target price, and key investment thesis points.,
      data_collection_task: Analyze current valuation metrics, growth projections, and risk factors to formulate an investment recommendation.,
      code_executor_task: Create a final slide with a clear investment recommendation (Buy/Hold/Sell) in large, colored text. Include a target price with potential upside percentage. Add bullet points outlining the key investment thesis including growth potential, technological advantages, and potential risks. Format the slide professionally with consistent fonts and colors. Remember to append the slide to PPT and save PPT.
  ]

REMEMBER: EACH GROUP TASK MUST SAVE THE PPT AFTER APPENDING THE SLIDE.
"""

GENERAL_PPT_PLAN_PROMPT = """Excellent. To create the presentation outlined in the request, we have assembled the following team:

{team}

USER QUERY: {task}

You are a professional PPT creation planning assistant. 
Based on the team composition, user query, please generate a detailed PowerPoint presentation creation plan following these structural requirements:

Each plan group should contain the following elements:
1. name: A short title for this group task
2. description: Detailed explanation of the group objective and content
3. data_collection_task: Specific instructions for gathering data needed for this group task (optional)
4. code_executor_task: Description of what code should do to process data, GENERATE slide, APPEND it to PPT and SAVE PPT (optional)

Plan requirements:
- Each group task should GENERATE one slide, APPEND it to the PPT and SAVE PPT.
- Include common PPT elements such as titles, body text, bullet point lists, visual elements (charts, images, etc.)
- Code tasks should describe operations to be performed using Python (with visualization libraries like matplotlib, seaborn, plotly, etc.) or bash to install dependencies
- Code task descriptions should be clear enough to guide future code generation based on the collected data

Please output in JSON format, conforming to the following structure:

```
{{
  "groups": [
    {{
      "name": "Group Task Name",
      "description": "Group Task Description",
      "data_collection_task": "Data Collection Task Description",
      "code_executor_task": "Description of code task to be performed based on the collected data"
    }},
    ... (repeat for each section)
  ]
}}
```

REMEMBER: EACH GROUP TASK MUST SAVE THE PPT AFTER APPENDING THE SLIDE.
"""


GENERAL_REPORT_PLAN_PROMPT = """
# Role and Objective

You are a professional report creation planning assistant.  
Your task is to design a structured, step-by-step plan for generating a comprehensive report based on the USER QUERY: 

{task}

and the team's composition:

{team}

The plan must outline clear group task sections, assign data collection and execution subtasks, and adhere to markdown-based deliverables.

# Instructions

- Analyze the USER QUERY and team composition to break down the report into logical, coherent sections (group tasks).
- For each group, define:  
  - A concise section title.
  - A detailed description of the section's objective and expected content.
  - Data collection instructions tailored to that group's focus.
  - Executor instructions for creating or appending a markdown section using code-enabled tools (e.g., Python/bash), ensuring that the markdown file is built section by section and saved after each step.
- Ensure section order presents information logically (e.g., Introduction, Methodology, Analysis, Recommendations, Conclusion) and aligns with the user’s intent.
- Each group task APPENDS its output to the markdown report in sequence.
- Output the structured report creation plan in the specified JSON format.

## Sub-categories for more detailed instructions

- Team Analysis: Examine the team composition to leverage relevant expertise for each report section.
- Section Sequencing: Order report sections so they build logically (context → method → analysis → insight).
- Data Collection Specificity: Provide clear, actionable instructions for sourcing data (e.g., databases, APIs, literature review).
- Executor Task Precision: Specify whether the executor starts the markdown file (first section) or appends to/save it (subsequent sections); reference tools/scripts as appropriate.
- Modularity: Each group's tasks should enable easy parallelization or serial output aggregation.

# Reasoning Steps

1. Parse the USER QUERY to extract the report topic, goals, and required analyses.
2. Break down the report into thematic sections, aligning these with team expertise.
3. For each section:
   - Define the section title (succinct, informative).
   - Write a detailed description outlining purpose and content.
   - Assign data collection tasks with clear detail on data types and sources.
   - Write precise executor instructions for markdown file handling.
4. Sequence all sections logically for readability and narrative flow.
5. Output the plan in the required JSON structure.

# Output Format

Return the report creation plan as a JSON structure:
```
{{
  "groups": [
    {{
      "name": "Section title",
      "description": "Section objective and content summary",
      "data_collection_task": "Instructions for data gathering for this section",
      "code_executor_task": "Instructions for generating or appending markdown and saving"
    }},
    ... (repeat for each section)
  ]
}}
```

# Examples

## Example 1

USER QUERY: "Create a report on analyzing Tesla's stock price performance"

```
{{
  "groups": [
    {{
      "name": "Introduction",
      "description": "Outline the report objectives, scope, and Tesla's relevance in the stock market.",
      "data_collection_task": "Research and summarize Tesla's business overview and recent market status.",
      "code_executor_task": "Generate the initial markdown file with the Introduction section and save it."
    }},
    {{
      "name": "Historical Stock Performance",
      "description": "Present detailed analysis of Tesla's stock price trends over the past 5 years.",
      "data_collection_task": "Gather historical stock price data from financial APIs (e.g., Yahoo Finance) covering at least the last 5 years.",
      "code_executor_task": "Append a 'Historical Stock Performance' section with visualizations (charts/tables) to the existing markdown file and save."
    }},
    {{
      "name": "Key Drivers Analysis",
      "description": "Discuss primary factors (internal and external) influencing Tesla's stock.",
      "data_collection_task": "Identify and summarize events, financials, and external trends affecting price movements.",
      "code_executor_task": "Append analysis findings as a new markdown section and save the file."
    }},
    {{
      "name": "Forecast and Recommendations",
      "description": "Provide insights on future performance and investment recommendations.",
      "data_collection_task": "Obtain analyst forecasts, aggregate expert opinions, and synthesize potential scenarios.",
      "code_executor_task": "Append the final section with recommendations to the markdown file and save."
    }},
    ...more groups
  ]
}}
```

# Final instructions and prompt to think step by step

Think through the user request and the team's skills before designing the plan.  
Break the report into logical sections, clarifying objectives and data needs at each stage.  
For every group, specify actionable collection instructions and precise markdown handling tasks.  
Ensure the overall flow is clear and optimized for collaboration and modular section-building.
"""


GENERAL_FACTS_PROMPT_CN = """
请用中文回答。

下面我将给您一个请求。在开始处理请求之前，请您尽力回答以下预调查。请记住，您是一个百科全书，因此应该有丰富的知识储备可供借鉴。

请求如下：

{task}

以下是预调查：

请列出请求中给出的任何具体事实或数据。可能没有。
请列出可能需要查找的任何事实，以及具体在哪里可以找到它们。在某些情况下，请求本身会提及权威来源。
请列出可能需要推导（例如，通过逻辑推论、模拟或计算）的任何事实。
请列出从记忆中回忆、直觉、有充分理由的猜测等推测出的任何事实。
回答此调查时，请记住事实通常是具体的名称、日期、统计数据等。您的回答应使用以下标题：

1. 已给出或已验证的事实
2. 待查找的事实
3. 待推导的事实
4. 经验性猜测

请勿包含任何其他标题或部分。请勿在未被要求时列出后续步骤或计划。
"""

GENERAL_PLAN_PROMPT_CN = """
请用中文回答。

太棒了。为了处理这个请求，我们组建了以下团队：

{team}

用户查询：{task}

您是一名专业的规划助理。
根据团队构成、用户查询以及已知和未知的事实，请制定一个处理用户查询的计划。请记住，没有要求所有团队成员都参与——某个团队成员的专业知识可能不适用于此任务。

每个计划组应包含以下元素：

1. name: 此组任务的简短标题
2. description: 对组目标的详细解释。
3. data_collection_task: 收集此组任务所需数据的具体说明（可选）
4. code_executor_task: 对代码执行器应做什么的描述，只需详细描述即可，无需实际代码块。（可选）
"""


GENERAL_REPORT_PLAN_PROMPT_CN = """
请用中文回答

# 角色与目标

您是一名专业的报告创建规划助理。
您的任务是根据用户查询：

{task}

和团队构成：

{team}

设计一个结构化、分步的计划，用于生成一份全面的报告。该计划必须概述清晰的组任务部分，分配数据收集和执行子任务，并遵守基于Markdown的交付要求。

# 说明

- 分析用户查询和团队构成，将报告分解为逻辑、连贯的部分（组任务）。
- 对于每个组，定义：
  - 简洁的部分标题。
  - 对该部分目标和预期内容的详细描述。
  - 针对该组重点的数据收集说明。
  - 关于使用代码启用工具（例如Python/bash）创建或附加Markdown部分的执行器说明，确保Markdown文件逐节构建并在每一步之后保存
- 确保部分顺序逻辑地呈现信息（例如，引言、方法、分析、建议、结论），并与用户意图保持一致。
- 每个组任务按顺序将其输出附加到Markdown报告中。
- 以指定的JSON格式输出结构化的报告创建计划。

## 更详细说明的子类别

- 团队分析：检查团队构成，以利用相关专业知识完成报告的每个部分。
- 部分排序：按逻辑顺序排列报告部分，使其层层递进（背景 → 方法 → 分析 → 洞察）。
- 数据收集具体性：提供清晰、可操作的说明，说明数据来源（例如，数据库、API、文献回顾）。
- 执行器任务精确性：指定执行器是启动Markdown文件（第一部分）还是附加到/保存它（后续部分）；酌情引用工具/脚本。
- 模块化：每个组的任务都应易于并行化或串行输出聚合。

# 推理步骤

1. 解析用户查询以提取报告主题、目标和所需分析。
2. 将报告分解为主题部分，并与团队专业知识对齐。
3. 对于每个部分：
   - 定义部分标题（简洁、信息丰富）。
   - 编写详细描述，概述目的和内容。
   - 分配数据收集任务，详细说明数据类型和来源。
   - 编写精确的执行器说明，用于Markdown文件处理。
4. 逻辑地排序所有部分，以提高可读性和叙事流畅性。
5. 以所需的JSON结构输出计划。

# 输出格式

以JSON结构返回报告创建计划：
```
{{
  "groups": [
    {{
      "name": "部分标题",
      "description": "部分目标和内容摘要",
      "data_collection_task": "本部分的数据收集说明",
      "code_executor_task": "生成或附加Markdown并保存的说明"
    }},
    // ... (为每个部分重复)
  ]
}}
```

# 示例


用户查询：“创建一份分析特斯拉股票价格表现的报告”

```
{{
  "groups": [
    {{
      "name": "引言",
      "description": "概述报告目标、范围以及特斯拉在股市中的相关性。",
      "data_collection_task": "研究并总结特斯拉的业务概览和近期市场状况。",
      "code_executor_task": "生成包含引言部分的初始Markdown文件并保存。"
    }},
    {{
      "name": "历史股票表现",
      "description": "详细分析特斯拉过去5年的股票价格趋势。",
      "data_collection_task": "从金融API（例如Yahoo Finance）收集至少过去5年的历史股票价格数据。",
      "code_executor_task": "将“历史股票表现”部分与可视化内容（图表/表格）附加到现有Markdown文件并保存。"
    }},
    {{
      "name": "关键驱动因素分析",
      "description": "讨论影响特斯拉股票的主要因素（内部和外部）。",
      "data_collection_task": "识别并总结影响价格变动的事件、财务状况和外部趋势。",
      "code_executor_task": "将分析结果作为新的Markdown部分附加并保存文件。"
    }},
    {{
      "name": "预测与建议",
      "description": "提供未来业绩的洞察和投资建议。",
      "data_collection_task": "获取分析师预测，汇总专家意见，并综合潜在情景。",
      "code_executor_task": "将包含建议的最终部分附加到Markdown文件并保存。"
    }},
    // ...更多组
  ]
}}
```

# 最终说明和逐步思考提示

1. 在设计计划之前，请仔细思考用户请求和团队的技能。
2. 将报告分解为逻辑部分，在每个阶段阐明目标和数据需求。
3. 对于每个组，指定可操作的收集说明和精确的Markdown处理任务。
4. 确保整体流程清晰，并针对协作和模块化部分构建进行优化。
"""

GENERAL_PPT_PLAN_PROMPT_CN = """
请用中文回答。

太棒了。为了创建请求中概述的演示文稿，我们组建了以下团队：

{team}

用户查询：{task}

您是一名专业的PPT创建规划助理。
根据团队构成、用户查询，请生成一个详细的PowerPoint演示文稿创建计划，并遵循以下结构要求：

每个计划组应包含以下元素：

名称：此组任务的简短标题
描述：对组目标和内容的详细解释
数据收集任务：收集此组任务所需数据的具体说明（可选）
代码执行器任务：描述代码应如何处理数据、生成幻灯片、将其附加到PPT并保存PPT（可选）
计划要求：

每个组任务应生成一张幻灯片，将其附加到PPT并保存PPT。
包括常见的PPT元素，例如标题、正文、项目符号列表、视觉元素（图表、图像等）。
代码任务应描述使用Python（以及matplotlib、seaborn、plotly等可视化库）或bash来安装依赖项的操作。
代码任务描述应足够清晰，以便根据收集到的数据指导未来的代码生成。
请以JSON格式输出，并符合以下结构：

```
{{
  "groups": [
    {{
      "name": "组任务名称",
      "description": "组任务描述",
      "data_collection_task": "数据收集任务描述",
      "code_executor_task": "根据收集到的数据执行的代码任务描述"
    }},
    ... (更多组)
  ]
}}
```
请记住：每个组任务在附加幻灯片后都必须保存PPT。
"""

FINANCIAL_PPT_FACTS_PROMPT_CN = """
请用中文回答。

下面我将为您呈现一个与创建金融PowerPoint演示文稿相关的请求。在开始处理请求之前，请您尽力回答以下预调查。请记住，您拥有专业金融分析师和策略师（例如，特许金融分析师CFA级别）的知识，并具备严谨的分析能力，因此请运用您对金融、市场和数据解读的深刻理解。请求如下：

{task}

以下是预调查：

请列出请求中直接给出的任何特定财务数据、公司名称、报告期、目标指标或限制条件。可能没有。
请列出可能需要查找的任何特定财务数据点、市场信息、经济指标、监管细节或公司特定信息。请指明潜在的权威来源（例如，SEC备案文件如10-K/10-Q、彭博社、路孚特Eikon、公司投资者关系网站、中央银行数据库、特定行业报告、具名分析师研究报告）。
请列出可能需要推导（例如，计算市盈率或股本回报率等财务比率、执行折现现金流（DCF）分析、进行敏感性分析、计算复合年增长率（CAGR）、根据假设预测未来业绩、整合来自多个来源的数据）的任何财务数据、比率、预测、估值或分析。
请列出基于您的财务专业知识、回忆的知识或有充分理由的估计（例如，对未来利率的假设、对市场增长的估计、对管理层质量的定性评估、基于行业知识的潜在并购协同效应）的任何假设、定性评估、市场情绪或潜在趋势。
回答此调查时，请记住事实通常是具体的名称、日期、财务数据、比率、经济数据点等。您的回答应使用以下标题：

1. 已给出或已验证的事实
2. 待查找的事实
3. 待推导的事实
4. 经验性猜测/假设
请勿包含任何其他标题或部分。请勿在未被要求时列出后续步骤或计划。
"""

FINANCIAL_PPT_PLAN_PROMPT_CN = """
请用中文回答。

太棒了。为了创建请求中概述的财务演示文稿，我们组建了以下团队：

{team}

用户查询：{task}

您是一名专业的财务PPT创建规划助理。
根据团队构成、用户查询，请生成一个详细的财务PowerPoint演示文稿创建计划，并遵循以下结构要求：

每个计划组应包含以下元素：

1. name: 此组任务的简短标题
2. description: 对组目标的详细解释
3. data_collection_task: 收集此组任务所需数据的具体说明（可选）
4. code_executor_task: 描述代码应如何处理数据、生成幻灯片、将其附加到PPT并保存PPT（可选）
计划要求：

每个组任务应生成一张幻灯片，将其附加到PPT并保存PPT。
包括常见的财务PPT元素，例如标题、财务指标、项目符号列表、视觉元素（图表、图形、财务表格等）。
代码部分应主要使用Python（以及matplotlib、seaborn、plotly、pandas_datareader等可视化库）或bash来安装依赖项。
确保代码实用且可执行，能够完成财务数据处理和幻灯片生成任务。
重点关注财务分析、股票表现、公司估值、市场趋势、投资建议和财务预测。
请以JSON格式输出，并符合以下结构：

```
{{
  "groups": [
    {{
      "name": "组任务名称",
      "description": "组任务描述",
      "data_collection_task": "数据收集任务描述",
      "code_executor_task": "根据收集到的数据执行的代码任务描述"
    }},
    ... (更多组)
  ]
}}
```

特斯拉股票分析演示文稿示例：

```
{{
  "groups": [
    {{
      "name": "公司概览",
      "description": "创建标题幻灯片，简要概览特斯拉，包括其股票代码、所属行业和成立日期。",
      "data_collection_task": "查找特斯拉股票代码（TSLA）、成立日期、总部地点和CEO信息。",
      "code_executor_task": "创建PPT并添加带有特斯拉名称、股票代码和基本公司信息的标题幻灯片。保存PPT以备后续添加。"
    }},
    {{
      "name": "股票价格表现",
      "description": "创建一张幻灯片，显示特斯拉过去5年的股票价格表现，并突出显示关键里程碑。",
      "data_collection_task": "从Yahoo Finance API检索特斯拉5年历史股票价格数据，并识别关键公司里程碑。",
      "code_executor_task": "使用yfinance下载特斯拉5年股票价格数据。创建显示收盘价随时间变化的折线图。在图表上标记重要的公司里程碑。将图表添加到演示文稿中的新幻灯片，并附上适当的标题和轴标签。请记住将幻灯片附加到PPT并保存PPT。"
    }},
    {{
      "name": "财务指标",
      "description": "创建一张幻灯片，分析特斯拉的关键财务指标，包括营收增长、利润率、市盈率和每股收益。",
      "data_collection_task": "收集特斯拉过去3年的损益表、资产负债表和关键财务比率。",
      "code_executor_task": "使用yfinance从特斯拉财务报表中提取财务指标。创建格式化表格，显示过去3年的营收、净收入、利润率、市盈率和每股收益数据。将表格添加到新幻灯片，并附上适当的标题和简要分析文本。请记住将幻灯片附加到PPT并保存PPT。"
    }},
    {{
      "name": "市场地位与竞争",
      "description": "创建一张幻灯片，分析特斯拉在电动汽车行业的市场地位、市场份额和主要竞争对手。",
      "data_collection_task": "研究特斯拉在电动汽车行业的市场份额，并识别其主要竞争对手及其市场份额。",
      "code_executor_task": "创建饼图，显示特斯拉及其主要竞争对手在全球电动汽车市场份额的分布。将图表添加到新幻灯片。包括一个文本框，突出显示特斯拉在行业中的竞争优势，例如品牌认知度、电池技术、充电基础设施和软件能力。请记住将幻灯片附加到PPT并保存PPT。"
    }},
    {{
      "name": "投资建议",
      "description": "创建一张幻灯片，包含投资建议、目标价格和关键投资论点。",
      "data_collection_task": "分析当前估值指标、增长预测和风险因素，以制定投资建议。",
      "code_executor_task": "创建最后一张幻灯片，用醒目、彩色的文字清晰地显示投资建议（买入/持有/卖出）。包括一个目标价格及潜在上涨百分比。添加项目符号，概述关键投资论点，包括增长潜力、技术优势和潜在风险。以专业的格式设置幻灯片，保持字体和颜色一致。请记住将幻灯片附加到PPT并保存PPT。"
    }},
    ... (更多组)
  ]
}}
```
请记住：每个组任务在附加幻灯片后都必须保存PPT。
"""
