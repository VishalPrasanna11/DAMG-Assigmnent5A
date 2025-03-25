# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from agents.rag_agent import RagAgent
from agents.snowflake_agent import SnowflakeAgent 
from agents.websearch_agent import WebSearchAgent
from langraph.orchestrator import ResearchOrchestrator

app = FastAPI(title="NVIDIA Research Assistant API")

class ResearchQuery(BaseModel):
    query: str
    year: Optional[int] = None
    quarter: Optional[int] = None
    agents: List[str] = ["rag", "snowflake", "websearch"]

@app.post("/research")
async def generate_research(request: ResearchQuery):
    try:
        # Initialize orchestrator with selected agents
        orchestrator = ResearchOrchestrator(
            use_rag="rag" in request.agents,
            use_snowflake="snowflake" in request.agents,
            use_websearch="websearch" in request.agents
        )
        
        # Generate research report
        print(f"Running orchestrator with query: {request.query}, year: {request.year}, quarter: {request.quarter}")
        result = orchestrator.run(
            query=request.query,
            year=request.year,
            quarter=request.quarter
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
