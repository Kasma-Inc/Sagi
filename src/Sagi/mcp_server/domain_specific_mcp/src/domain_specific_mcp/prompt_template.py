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

  groups: [
      name: Group Task Name,
      description: Group Task Description,
      data_collection_task: Data Collection Task Description,
      code_executor_task: Description of code task to be performed based on the collected data
    ,
      ...more groups
  ]

REMEMBER: EACH GROUP TASK MUST SAVE THE PPT AFTER APPENDING THE SLIDE.
"""
