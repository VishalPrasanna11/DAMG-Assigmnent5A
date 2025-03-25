# backend/agents/websearch_agent.py
import os
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class WebSearchAgent:
    def __init__(self):
        # Initialize with your preferred search API
        # For example, using SerpAPI
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY environment variable not set")
        
        self.search_url = "https://serpapi.com/search"
        self.llm = ChatOpenAI(temperature=0)
        
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Query web search API for latest information on NVIDIA related to the query.
        
        Args:
            query_text: The query text
            
        Returns:
            Dictionary with search results and synthesized information
        """
        # Augment query to focus on NVIDIA and recent information
        augmented_query = f"NVIDIA {query_text} latest news"
        
        # Prepare search parameters
        params = {
            "q": augmented_query,
            "api_key": self.api_key,
            "engine": "google",
            "num": 5,  # Limit to 5 results
            "tbm": "nws"  # News search
        }
        
        # Execute search
        response = requests.get(self.search_url, params=params)
        search_results = response.json()
        
        # Extract organic results
        news_results = search_results.get("news_results", [])
        
        # Format results
        formatted_results = []
        for result in news_results:
            formatted_results.append({
                "title": result.get("title", ""),
                "link": result.get("link", ""),
                "snippet": result.get("snippet", ""),
                "source": result.get("source", ""),
                "date": result.get("date", "")
            })
        
        # Generate insights from the search results
        insights = self._generate_insights(formatted_results, query_text)
        
        # Return results
        return {
            "results": formatted_results,
            "response": insights
        }
        
    def _generate_insights(self, results: List[Dict[str, Any]], query_text: str) -> str:
        """Generate insights from search results"""
        # Format results for the prompt
        context = ""
        for i, result in enumerate(results, 1):
            context += f"{i}. Title: {result['title']}\n"
            context += f"   Source: {result['source']}\n"
            context += f"   Date: {result['date']}\n"
            context += f"   Snippet: {result['snippet']}\n\n"
        
        # Create prompt for insights
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a research analyst specializing in NVIDIA.
            Analyze the following recent news articles about NVIDIA to answer the query.
            Focus on extracting the most relevant and recent insights.
            Provide a balanced perspective considering multiple sources.
            """),
            ("human", "Recent news articles about NVIDIA:\n{context}\n\nQuery: {query}")
        ])
        
        # Generate insights
        chain = prompt | self.llm
        response = chain.invoke({
            "context": context,
            "query": query_text
        })
        
        return response.content