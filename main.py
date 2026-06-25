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




                                                                                        # FINAL GRAPH
agent_graph = StateGraph(AGENTState)

agent_graph.add_node("orchestrator", orchestrator_node)
agent_graph.add_node("stock_analyser", stock_analyser_node)
agent_graph.add_node("portfolio_advisor", portfolio_advisor_node)
agent_graph.add_node("security_selector", security_selector_node)
agent_graph.add_node("report_generator", report_generator_node)

agent_graph.add_edge(START, "orchestrator")

#all report back to the orchestrator, MAIN RULE OF OUR MULTI-AGENT PIPELINE, EVERY SUB AGENT HAS A PARENT 
agent_graph.add_edge("stock_analyser", "orchestrator")
agent_graph.add_edge("portfolio_advisor", "orchestrator")
agent_graph.add_edge("security_selector", "orchestrator")
agent_graph.add_edge("report_generator", "orchestrator")

#PARENT ORCHESTRATOR DECIDES WHICH NODE TO REACH BY CHECKING -> 1. IF stock_analysis STATE IS EMPTY, REACH THE stock_analyser_node WHICH POPULATES THE stock_analysis STATE AND REPORTS BACK TO ORCHESTRATOR    2. NOW stock_analysis STATE IS FILLED BUT portfolio_allocation STATE IS EMPTY, SO NOW ORCHESTRATOR REACHED THE portfolio_analyser_node WHICH POPULATES THE portfolio_allocation STATE AND AGAIN THE NODE REPORTS BACK TO THE ORCHESTRATOR.     3.NOW THE ORCHESTRATOR SEES THAT BOTH THE STATES ARE FILLED, SO IT REACHES THE report_generation_node WICH FILLS THE final_report STATE and again reports to orchestrator.      4.NOW THE ORCHESTRATOR SEES THAT ALL THREE STATES ARE FILLED, SO IT FINALLY REACHES THE END. 
agent_graph.add_conditional_edges(
    "orchestrator",
    route_next,{                               ##ROUTE DECIDER FUNCTION used here
        "stock_analyser": "stock_analyser",
        "portfolio_advisor": "portfolio_advisor",
        "security_selector": "security_selector",
        "report_generator": "report_generator",
        "FINISH": END
    }
)


agent_pipeline = agent_graph.compile()


if __name__ == "__main__":
    # 1. Define the initial state with a test investment request
    initial_input = {
        "user_request": "I have $100,000 to invest. I want an aggressive strategy focused heavily on growth stocks for a 10-year horizon, but I want a 15% cash safety net to protect against high market volatility.",
        "stock_analysis": "",
        "portfolio_allocation": "",
        "security_selection": "",
        "next_step": "",
        "final_report": ""
    }
    
    print("=================== INITIATING WEALTH MANAGEMENT PIPELINE ===================")
    
    # 2. Run the compiled graph pipeline synchronously
    final_state = agent_pipeline.invoke(initial_input)
    
    print("\n======================= PIPELINE EXECUTION COMPLETE =======================")
    print("\nGenerated Investment Prospectus:\n")
    
    # 3. Print the final compiled document from the publisher node
    print(final_state.get("final_report"))

