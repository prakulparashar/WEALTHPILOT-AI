from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import yfinance as yf
from enum import Enum
from langchain_core.tools import tool
from typing import Optional


load_dotenv()

llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)



                                                                                       # 1. SHARED AGENT STATE 
class AGENTState (TypedDict):
    """The central memory matrix shared across all nodes."""
    user_request: str
    stock_analysis: str
    portfolio_allocation: str
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
class AssetAllocation(BaseModel):
    asset_class: Literal["Equities", "Fixed Income", "Cash", "Alternative Assets"] = Field(
        description="The category of the financial asset."
    )
    ticker: str = Field(description="Ticker symbol or identifier (e.g., 'AAPL', 'BND', 'CASH').")
    percentage: float = Field(description="The allocation percentage of the total portfolio (e.g., 45.5).")
    justification: str = Field(description="One-sentence structural reasoning based on the stock analysis.")

class PortfolioStrategy(BaseModel):
    risk_profile: Literal["Conservative", "Moderate", "Aggressive"] = Field(
        description="Assessed risk tolerance baseline based on the market backdrop."
    )
    allocations: List[AssetAllocation] = Field(description="List of specific asset distributions.")

def portfolio_advisor_node(state: AGENTState) -> dict:
    print("\n--- [Executing Node]: Portfolio Advisor ---")
    
    market_context = state.get("stock_analysis", "")
    user_goal = state.get("user_request", "")
    
    #  Bind the Pydantic schema to your LLM to force structured response
    llm_structured = llm.with_structured_output(PortfolioStrategy)
    
    system_prompt = (
        "You are an expert Portfolio Risk Management Consultant. Your job is to translate qualitative "
        "stock or macroeconomic analysis reports into concrete, quantitative asset allocation models.\n\n"
        "Review the background report carefully. Ensure that the total allocation percentages add up "
        "exactly to 100%. Provide deep, clear structural justifications for every single asset you pick."
    )
    
    human_prompt = f"User Request: {user_goal}\n\nFinancial Context Report:\n{market_context}"
    
    #  Invoke the structured model
    structured_response = llm_structured.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return {"portfolio_allocation": str(structured_response.model_dump_json(indent=2))}
    





# -----------------NODE: FINAL REPORT GENERATION ------------------------
def report_generator_node(state: AGENTState) -> dict:
    pass

# -----------------NODE: ORCHESTRATOR ------------------------------
def orchestrator_node(state: AGENTState) -> dict:
    if not state.get("stock_analysis"):
        return{"next_step" : "stock_analyser"}
    elif not state.get("portfolio_allocation"):
        return{"next_step" : "portfolio_advisor"}
    elif not state.get("final_report"):
        return{"next_step" : "report_generator"}
    else:
        return{"next_step" : "FINISH"}
#ROUTE DECIDER FUNCTION
def route_next(state:  AGENTState):
    return state["next_step"]




                                                                                        # FINAL GRAPH
agent_graph = StateGraph(AGENTState)

agent_graph.add_node("orchestrator", orchestrator_node)
agent_graph.add_node("stock_analyser", stock_analyser_node)
agent_graph.add_node("portfolio_advisor", portfolio_advisor_node)
agent_graph.add_node("report_generator", report_generator_node)

agent_graph.add_edge(START, "orchestrator")

#all report back to the orchestrator, MAIN RULE OF OUR MULTI-AGENT PIPELINE, EVERY SUB AGENT HAS A PARENT 
agent_graph.add_edge("stock_analyser", "orchestrator")
agent_graph.add_edge("portfolio_advisor", "orchestrator")
agent_graph.add_edge("report_generator", "orchestrator")

#PARENT ORCHESTRATOR DECIDES WHICH NODE TO REACH BY CHECKING -> 1. IF stock_analysis STATE IS EMPTY, REACH THE stock_analyser_node WHICH POPULATES THE stock_analysis STATE AND REPORTS BACK TO ORCHESTRATOR    2. NOW stock_analysis STATE IS FILLED BUT portfolio_allocation STATE IS EMPTY, SO NOW ORCHESTRATOR REACHED THE portfolio_analyser_node WHICH POPULATES THE portfolio_allocation STATE AND AGAIN THE NODE REPORTS BACK TO THE ORCHESTRATOR.     3.NOW THE ORCHESTRATOR SEES THAT BOTH THE STATES ARE FILLED, SO IT REACHES THE report_generation_node WICH FILLS THE final_report STATE and again reports to orchestrator.      4.NOW THE ORCHESTRATOR SEES THAT ALL THREE STATES ARE FILLED, SO IT FINALLY REACHES THE END. 
agent_graph.add_conditional_edges(
    "orchestrator",
    route_next,{                               ##ROUTE DECIDER FUNCTION used here
        "stock_analyser": "stock_analyser",
        "portfolio_advisor": "portfolio_advisor",
        "report_generator": "report_generator",
        "FINISH": END
    }
)


agent_pipeline = agent_graph.compile()

