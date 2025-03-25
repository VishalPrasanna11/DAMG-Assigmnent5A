# backend/langraph/orchestrator.py
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import langchain
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import operator

# Import our agents
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent
from agents.websearch_agent import WebSearchAgent

# Define our state 
class ResearchState(TypedDict):
    query: str
    year: Optional[int]
    quarter: Optional[int]
    messages: List[Any]
    rag_results: Optional[Dict[str, Any]]
    snowflake_results: Optional[Dict[str, Any]]
    websearch_results: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    active_agents: List[str]

class ResearchOrchestrator:
    def __init__(self, use_rag: bool = True, use_snowflake: bool = True, use_websearch: bool = True):
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
        
        # Initialize agents if needed
        self.rag_agent = RagAgent() if use_rag else None
        self.snowflake_agent = SnowflakeAgent() if use_snowflake else None
        self.websearch_agent = WebSearchAgent() if use_websearch else None
        
        # Track which agents are active
        self.active_agents = []
        if use_rag:
            self.active_agents.append("rag")
        if use_snowflake:
            self.active_agents.append("snowflake")
        if use_websearch:
            self.active_agents.append("websearch")
        
        # Build the LangGraph workflow
        self.workflow = self._build_graph()
        
    def _build_graph(self):
        # Define our graph
        graph = StateGraph(ResearchState)
        
        # Router node - determines which agents to call
        @graph.node("router")
        def router(state: ResearchState) -> ResearchState:
            # Initialize the state with a system message
            system_message = SystemMessage(content="""
            You are a research assistant that routes queries to specialized agents.
            Based on the user's query, determine which agents should be called.
            """)
            
            # Add messages to state
            if len(state["messages"]) == 0:
                state["messages"] = [system_message, HumanMessage(content=state["query"])]
            
            return state
        
        # RAG node - handles document retrieval
        @graph.node("rag_agent")
        def rag_node(state: ResearchState) -> ResearchState:
            if "rag" not in state["active_agents"]:
                return state
                
            # Process with RAG agent
            rag_results = self.rag_agent.query(
                state["query"], 
                year=state["year"], 
                quarter=state["quarter"]
            )
            
            # Update state with results
            state["rag_results"] = rag_results
            return state
        
        # Snowflake node - handles financial metrics
        @graph.node("snowflake_agent")
        def snowflake_node(state: ResearchState) -> ResearchState:
            if "snowflake" not in state["active_agents"]:
                return state
                
            # Process with Snowflake agent
            snowflake_results = self.snowflake_agent.query(
                state["query"], 
                year=state["year"], 
                quarter=state["quarter"]
            )
            
            # Update state with results
            state["snowflake_results"] = snowflake_results
            return state
        
        # Web search node - handles latest information
        @graph.node("websearch_agent")
        def websearch_node(state: ResearchState) -> ResearchState:
            if "websearch" not in state["active_agents"]:
                return state
                
            # Process with Web Search agent
            websearch_results = self.websearch_agent.query(
                state["query"]
            )
            
            # Update state with results
            state["websearch_results"] = websearch_results
            return state
        
        # Synthesis node - combines all information
        @graph.node("synthesis")
        def synthesis_node(state: ResearchState) -> ResearchState:
            # Build prompt for synthesis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a financial research assistant specialized in NVIDIA. 
                Synthesize information from multiple sources to create a comprehensive report.
                Include relevant information from all available sources.
                Structure your response clearly with sections for Historical Data, 
                Financial Metrics, and Latest Insights when available.
                """),
                ("human", "Please create a comprehensive report answering the following query: {query}"),
            ])
            
            # Prepare information from each agent
            historical_data = state.get("rag_results", {}).get("response", "No historical data available.")
            financial_metrics = state.get("snowflake_results", {}).get("response", "No financial metrics available.")
            latest_insights = state.get("websearch_results", {}).get("response", "No recent insights available.")
            
            # Create the complete context
            context = {
                "query": state["query"],
                "historical_data": historical_data,
                "financial_metrics": financial_metrics,
                "latest_insights": latest_insights
            }
            
            # Generate synthesis
            chain = prompt | self.llm
            response = chain.invoke(context)
            
            # Create final report
            final_report = {
                "content": response.content,
                "sections": {
                    "historical_data": {"content": historical_data} if "rag" in state["active_agents"] else None,
                    "financial_metrics": state.get("snowflake_results", {}) if "snowflake" in state["active_agents"] else None,
                    "latest_insights": {"content": latest_insights} if "websearch" in state["active_agents"] else None
                }
            }
            
            # Update state
            state["final_report"] = final_report
            state["messages"].append(AIMessage(content=response.content))
            
            return state
        
        # Define edges - the flow between nodes
        # After routing, each agent processes in parallel
        graph.add_edge("router", "rag_agent")
        graph.add_edge("router", "snowflake_agent")
        graph.add_edge("router", "websearch_agent")
        
        # After all agents finish, synthesize results
        graph.add_edge("rag_agent", "synthesis")
        graph.add_edge("snowflake_agent", "synthesis")
        graph.add_edge("websearch_agent", "synthesis")
        
        # Set entry and terminal nodes
        graph.set_entry_point("router")
        graph.set_finish_point("synthesis")
        
        # Compile the graph
        return graph.compile()
    
    def run(self, query: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the research orchestrator to generate a comprehensive report.
        
        Args:
            query: The research question
            year: Optional year filter
            quarter: Optional quarter filter
            
        Returns:
            Dictionary with the final research report
        """
        # Initialize state
        initial_state = {
            "query": query,
            "year": year,
            "quarter": quarter,
            "messages": [],
            "rag_results": None,
            "snowflake_results": None,
            "websearch_results": None,
            "final_report": None,
            "active_agents": self.active_agents
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Return the final report
        return final_state["final_report"]