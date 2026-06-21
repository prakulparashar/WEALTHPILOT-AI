from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END



# 1. SHARED AGENT STATE 
class AGENTState (TypedDict):
    """The central memory matrix shared across all nodes."""
    user_request: str
    stock_analysis: str
    portfolio_allocation: str
    next_step: str
    final_report: str





# NODE: ORCHESTRATOR 
def orchestrator_node(state: AGENTState) -> AGENTState:
    pass
# NODE: STOCK ANALYSER 
def stock_analyser_node(state: AGENTState) -> AGENTState:
    pass
# PORTFOLIO ADVISOR 
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
    route_next,{
        "stock_analyser": "stock_analyser",
        "portfolio_advisor": "portfolio_advisor",
        "report_generator": "report_generator",
        "FINISH": END
    }
)


agent_pipeline = agent_graph.compile()

