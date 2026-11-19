from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import yfinance as yf
from enum import Enum
from langchain_core.tools import tool
from typing import Optional


load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)



                                                                                       # 1. SHARED AGENT STATE 
class AGENTState (TypedDict):
    """The central memory matrix shared across all nodes."""
    user_request: str
    stock_analysis: str
    portfolio_allocation: str
    security_selection: str
    next_step: str
    final_report: str

                                                                                       # 2. TOOLS(for stock_analyser_node)

@tool
def fetch_specific_stock_metrics(ticker: str) -> str:
    """
    Fetches comprehensive real-time financial valuation, safety, and performance metrics 
    for a specific company ticker symbol (e.g., 'AAPL', 'TSLA', 'NVDA').
    Use this tool when the user asks about a specific stock asset.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info
        metrics = {
            "Company Name": info.get("longName", ticker),
            "Current Price": f"${info.get('regularMarketPrice', 'N/A')}",
            "Trailing P/E Ratio": info.get("trailingPE", "N/A"),
            "Debt-to-Equity Ratio": info.get("debtToEquity", "N/A"),
            "Return on Equity (ROE)": f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get("returnOnEquity") else "N/A",
            "Beta (Volatility Index)": info.get("beta", "N/A")
        }
        return "\n".join([f"{k}: {v}" for k, v in metrics.items()])
    except Exception as e:
        return f"Error fetching stock data for {ticker}: {str(e)}"

@tool
def fetch_macroeconomic_benchmarks() -> str:
    """
    Fetches broad macroeconomic index and bond yield data, including the S&P 500 (SPY), 
    Total Bond Market (BND), and the 10-Year Treasury Yield (^TNX).
    Use this tool when the user provides general cash amounts or financial planning goals 
    without naming a specific company.
    """
    try:
        context = "--- LIVE GLOBAL MACRO BACKDROP ---\n"
        for symbol, name in [("SPY", "S&P 500 ETF"), ("BND", "Total Bond ETF"), ("^TNX", "10-Yr Treasury Yield")]:
            ticker = yf.Ticker(symbol)
            price = ticker.info.get("regularMarketPrice", "N/A")
            context += f"{name} ({symbol}) Current Level/Price: {price}\n"
        return context
    except Exception as e:
        return f"Error fetching macroeconomic benchmarks: {str(e)}"
    



@tool #used by security selector node, fetches market price of top stocks
def screen_top_market_securities(custom_tickers: Optional[List[str]] = None) -> str:
    """
    Fetches real-time fundamental data, current pricing, valuation metrics, and health indicators 
    for a list of high-quality market securities. Use this tool inside the security selector node 
    to choose real, active stocks that match the macro asset allocation percentages.
    """
    # Default list of institutional heavyweights across Equities and Fixed Income if none provided
    tickers = custom_tickers if custom_tickers else ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "VOO", "BND", "AGG"]
    
    print(f"-> Tool running: Screening real-time market data for {tickers}...")
    screen_results = ["--- LIVE SECURITY SCREENER METRICS ---"]
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker.upper().strip())
            info = stock.info
            
            
            name = info.get("longName", ticker)
            price = info.get("regularMarketPrice", "N/A")
            forward_pe = info.get("forwardPE", "N/A")
            profit_margin = f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A"
            fifty_day_ma = info.get("fiftyDayAverage", "N/A")
            market_cap = info.get("marketCap", "N/A")
            roe = info.get("returnOnEquity", "N/A")
            beta = info.get("beta", "N/A")
            revenue_growth = info.get("revenueGrowth", "N/A")
            
            screen_results.append(
                        f"[{ticker}] {name}\n"
                        f"   - Current Price: ${price} (50-Day Avg: ${fifty_day_ma})\n"
                        f"   - Forward P/E: {forward_pe}\n"
                        f"   - Profit Margin: {profit_margin}\n"
                        f"   - Revenue Growth: {revenue_growth}\n"
                        f"   - Return on Equity: {roe}\n"
                        f"   - Beta: {beta}\n"
                        f"   - Market Cap: {market_cap}\n"
                    )
        except Exception as e:
            screen_results.append(f"[{ticker}] Error gathering live feed: {str(e)}")
            
    return "\n".join(screen_results)
    



                                                                                        # 3. NODES




# ----------------- NODE: STOCK ANALYSIS---------------- 
def stock_analyser_node(state: AGENTState) -> dict:
    
    user_goal = state.get("user_request", "")
    
    tools_list = [fetch_specific_stock_metrics, fetch_macroeconomic_benchmarks]
    tools_map = {tool.name: tool for tool in tools_list}
    
    llm_with_tools = llm.bind_tools(tools_list)
    
    #human and system message
    messages = [
        SystemMessage(content=(
            "You are an autonomous Financial Data Agent. Your goal is to review the user's investment query "
            "and determine exactly what financial data is required to run a high-quality analysis.\n\n"
            "Look at the tools available to you. If the user mentions a specific stock, invoke the stock metrics tool. "
            "If they give an open-ended goal, fetch the macro benchmarks. Review the outputs of your tool calls, "
            "and once you have gathered the data context you need, synthesize a deep qualitative equity/macro report "
            "for the downstream advisor node. Do not output math or specific cash allocations."
        )),
        HumanMessage(content=user_goal)
    ]
    
    max_iterations = 5
    iterations = 0
    while iterations < max_iterations:
        iterations += 1

        print("Invoking LLM to determine next step...")
        response = llm_with_tools.invoke(messages) #llm_with_tools ALWAYS responds with an AImessage which consists of a tool_calls array which consists of all the tools that the LLM requested for
        messages.append(response)
        
        # Check if the LLM chose to call a tool or if it is ready to give a final text analysis
        if not response.tool_calls: #tool not called by LLM(tool_calls array is empty), all info is gathered (this if statement will always be skipped on the first iteration of the while loop)
            return {"stock_analysis": response.content} #infinite loop ends
            
        # iterate the tool_calls array
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"-> LLM requesting tool execution: {tool_name}({tool_args})")
            
            target_tool = tools_map[tool_name]
            tool_output = target_tool.invoke(tool_args) #now the LLM requests the tools and python runs it
            
            # Append the tool message
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])) #response by the tools are appended in the message, and again sent to LLM so it can now use human,system,ai,tool messages to create the final response. THIS IS STANDARD BEHAVIOUR OF TOOLS AND LLM

    return {"stock_analysis": "Unable to complete analysis within iteration limit."}






# -----------------NODE: PORTFOLIO ADVISOR----------------- 
class MacroAllocation(BaseModel): #basemodel = pydantic
    equities_percentage: float = Field(description="Total allocation percentage for Equities (0-100).")
    fixed_income_percentage: float = Field(description="Total allocation percentage for Bonds/Fixed Income (0-100).")
    cash_percentage: float = Field(description="Total allocation percentage for Liquid Cash/Money Markets (0-100).")
    macro_reasoning: str = Field(description="A concise 2-sentence structural rationale for this macro mix based on the market conditions.")


#ABOVE PYDANTIC SCHEMA IS POPULATED BY THE NODE BELOW (llm.with_structured_output)


def portfolio_advisor_node(state: AGENTState) -> dict:
    print("\n--- [Executing Node]: Portfolio Advisor (Macro Allocation) ---")
    
    analysis_report = state.get("stock_analysis", "")
    user_goal = state.get("user_request", "")
    
    # Force the LLM to strictly return data matching our Pydantic schema
    structured_llm = llm.with_structured_output(MacroAllocation)
    
    system_prompt = (
        "You are a Chief Investment Strategist. Your single responsibility is to establish the top-level "
        "macro asset allocation (Equities vs Fixed Income vs Cash) based on a qualitative market analysis report.\n\n"
        "Do NOT pick individual stocks or tickers yet. Your allocation percentages MUST add up exactly to 100%."
    )
    
    human_prompt = f"User Investment Target: {user_goal}\n\nFinancial Context Report:\n{analysis_report}"
    
    # Invoke the model
    macro_result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    # Store the result as a structured string for the downstream Security Selector
    formatted_allocation = (
        f"Macro Asset Split:\n"
        f"- Equities: {macro_result.equities_percentage}%\n"
        f"- Fixed Income: {macro_result.fixed_income_percentage}%\n"
        f"- Cash: {macro_result.cash_percentage}%\n\n"
        f"Strategy Rationale: {macro_result.macro_reasoning}"
    )


    
    print(f"-> Strategic Mix Determined:\n{formatted_allocation}")
    return {"portfolio_allocation": formatted_allocation}
    

# -----------------NODE: SECURITY SELECTOR(15% APPLE, 10% NVDA) ------------------------
def security_selector_node(state: AGENTState) -> dict:
    print("\n--- [Executing Node]: Security Selector (Micro Allocation Loop) ---")
    
    macro_mix = state.get("portfolio_allocation", "")
    analysis_report = state.get("stock_analysis", "")
    
    # 1. Bind the newly created screener tool specifically to this worker node's scope
    tools_list = [screen_top_market_securities]
    tools_map = {tool.name: tool for tool in tools_list}
    llm_with_tools = llm.bind_tools(tools_list)
    
    system_prompt = (
        "You are an expert Equity Research Analyst and Asset Selector.\n\n"
        "Your single objective is to break down the high-level macro percentages into specific ticker weightings.\n"
        "To make accurate choices, you MUST invoke the 'screen_top_market_securities' tool first to read live "
        "fundamental data for potential company stocks or bond ETFs.\n\n"
        "CRITICAL MATHEMATICAL RULES:\n"
        "1. Every ticker percentage you assign MUST represent the percentage of the TOTAL portfolio (out of 100% total), NOT a percentage of a sub-asset class.\n"
        "2. The sum of all your individual ticker percentages must exactly match the macro splits (e.g., if Equities is 80%, your stock tickers must add up to exactly 80%).\n"
        "3. When printing dollar amounts, calculate them relative to the absolute total portfolio value (e.g., 20% of a $100,000 portfolio is $20,000)."
    )
    
    human_prompt = f"Target Macro Mix From Advisor:\n{macro_mix}\n\nSupporting Financial Context:\n{analysis_report}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    max_iterations = 3
    iterations = 0
    
    # 2. Autonomous loop for tool utilization and final synthesis
    while iterations < max_iterations:
        iterations += 1
        print(f"Invoking Selector LLM (Iteration {iterations}/{max_iterations})...")
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # If no tools are called, the LLM has successfully synthesized the data
        if not response.tool_calls:
            print("-> Micro asset selection finalized with real-time data.")
            return {"security_selection": response.content}
            
        # Dynamically execute requested tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"-> Selector requesting live market data feed via: {tool_name}")
            
            target_tool = tools_map[tool_name]
            tool_output = target_tool.invoke(tool_args)
            
            messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
            
    return {"security_selection": "Security picker timed out before finishing micro selection matrix."}
    


# -----------------NODE: FINAL REPORT GENERATION ------------------------
def report_generator_node(state: AGENTState) -> dict:
    print("\n--- [Executing Node]: Final Report Generation ---")
    
    # Extract all gathered assets from the shared central blackboard
    user_goal = state.get("user_request", "")
    analysis = state.get("stock_analysis", "")
    macro_split = state.get("portfolio_allocation", "")
    micro_split = state.get("security_selection", "")
    
    system_prompt = (
        "You are an institutional Investment Publisher. Your job is to format raw quantitative "
        "and qualitative data from your specialist sub-agents into a premium, beautifully typeset "
        "financial portfolio prospectus report for your client.\n\n"
        "Organize the report elegantly using clean Markdown headers (##), strong bolding (**), "
        "and clear bulleted lists. Make sure it reads like a polished document from an elite wealth management firm."
    )
    
    human_prompt = (
        f"Compile and format the final Investment Prospectus using this asset data:\n\n"
        f"### 1. CLIENT INVESTMENT OBJECTIVE\n{user_goal}\n\n"
        f"### 2. MACRO MARKET ANALYSIS\n{analysis}\n\n"
        f"### 3. STRATEGIC ASSET ALLOCATION OVERVIEW\n{macro_split}\n\n"
        f"### 4. TACTICAL TICKER SELECTION MATRIX\n{micro_split}\n"
    )
    
    # Standard linear invocation (no tools needed here, just pure synthesis)
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])
    
    print("-> Final Prospectus successfully generated!")
    return {"final_report": response.content}
    





                                                                                        # ORCHESTRATOR

# -----------------NODE: ORCHESTRATOR ------------------------------
def orchestrator_node(state: AGENTState) -> dict:
    if not state.get("stock_analysis"):
        return{"next_step" : "stock_analyser"}
    elif not state.get("portfolio_allocation"):
        return{"next_step" : "portfolio_advisor"}
    elif not state.get("security_selection"):
        return{"next_step": "security_selector"}
    elif not state.get("final_report"):
        return{"next_step" : "report_generator"}
    else:
        return{"next_step" : "FINISH"}
#ROUTE DECIDER FUNCTION
def route_next(state:  AGENTState):
    return state["next_step"]




