from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
import yfinance as yf

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


                                                                                        # 2. NODES


# NODE: ORCHESTRATOR 
def orchestrator_node(state: AGENTState) -> AGENTState:
    if not state.get("stock_analysis"):
        return{"next_step" : "stock_analyser"}
    elif not state.get("portfolio_allocation"):
        return{"next_step" : "portfolio_advisor"}
    elif not state.get("final_report"):
        return{"next_step" : "report_generator"}
    else:
        return{"next_step" : "FINISH"}
    



# NODE: STOCK ANALYSER 
import yfinance as yf
from langchain_core.messages import SystemMessage, HumanMessage

def stock_analyser_node(state: AGENTState) -> dict:
    print("\n--- [Executing Node]: Stock Analyser ---")
    

    user_goal = state.get("user_request", "")
    
   
    print("Fetching deep market metrics via yfinance...")
    try:
        #  SPY (S&P 500 ETF) 
        spy_ticker = yf.Ticker("SPY")
        spy_info = spy_ticker.info
        
       
        market_metrics = {
            "Current Price": f"${spy_info.get('regularMarketPrice', 'N/A')}",
            "Trailing P/E Ratio": spy_info.get("trailingPE", "N/A"),
            "Forward P/E Ratio": spy_info.get("forwardPE", "N/A"),
            "Price-to-Book (P/B) Ratio": spy_info.get("priceToBook", "N/A"),
            "Dividend Yield": f"{spy_info.get('dividendYield', 0) * 100:.2f}%" if spy_info.get("dividendYield") else "N/A",
            "52-Week High": f"${spy_info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52-Week Low": f"${spy_info.get('fiftyTwoWeekLow', 'N/A')}",
            "50-Day Moving Average": f"${spy_info.get('fiftyDayAverage', 'N/A')}",
            "Beta (Volatility Index)": spy_info.get("beta", "N/A")
        }
        
        # Format the dictionary into a highly readable, structured text block for the LLM
        financial_context = "--- LIVE MACRO MARKET DATA BACKGROUND ---\n"
        for key, value in market_metrics.items():
            financial_context += f"{key}: {value}\n"
            
    except Exception as e:
        financial_context = f"--- LIVE MACRO MARKET DATA BACKGROUND ---\n[Warning]: Failed to fetch real-time metrics due to network/API error: {str(e)}\n"

    # 3. Establish the strict specialist persona
    system_instruction = (
        "You are a Senior Equity Research Analyst. Your sole job is to synthesize the provided "
        "live market data backdrop and the user's personal investment goals to discover viable market sectors, "
        "macro-level trends, and industry-specific growth trajectories. Highlight core industry risks, "
        "valuation warnings (based on P/E or P/B imbalances), or volatility factors (indicated by Beta).\n\n"
        "CRITICAL CONSTRAINT: Do NOT calculate specific portfolio percentage splits, numeric asset "
        "weightings, or absolute cash allocations. Leave all asset allocation math entirely to the downstream advisor node. "
        "Provide your entire research findings as deep qualitative text analysis."
    )
    
    # 4. Construct the comprehensive data-driven Human Message payload
    user_prompt_with_data = (
        f"{financial_context}\n"
        f"--- USER INVESTMENT REQUEST ---\n"
        f"User Intent: {user_goal}\n\n"
        f"Instructions: Use the macro dataset context provided above to ground your analysis. "
        f"Provide a sophisticated, qualitative equity research assessment based on these real-time variables."
    )
    
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=user_prompt_with_data)
    ]
    
    # 5. Invoke Groq
    response = llm.invoke(messages)
    
    # 6. Push the data-rich summary directly onto the shared blackboard
    return {"stock_analysis": response.content}
















# NODE: PORTFOLIO ADVISOR 
def portfolio_advisor_node(state: AGENTState) -> AGENTState:
    pass
# NODE: FINAL REPORT GENERATION 
def report_generator_node(state: AGENTState) -> AGENTState:
    pass
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

