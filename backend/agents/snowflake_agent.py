# backend/agents/snowflake_agent.py
import os
from typing import Dict, Any, List, Optional
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class SnowflakeAgent:
    def __init__(self):
        # Initialize Snowflake connection
        self.conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema=os.getenv('SNOWFLAKE_SCHEMA')
        )
        self.llm = ChatOpenAI(temperature=0)
        
    def query(self, query_text: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Query Snowflake for NVIDIA valuation metrics.
        
        Args:
            query_text: The query text
            year: Optional year filter
            quarter: Optional quarter filter
            
        Returns:
            Dictionary with financial metrics and visualization
        """
        # Build SQL query with filters
        sql = "SELECT * FROM NVIDIA_VALUATION_METRICS"
        conditions = []
        
        if year is not None:
            conditions.append(f"YEAR = {year}")
        if quarter is not None:
            conditions.append(f"QUARTER = {quarter}")
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(sql)
        
        # Convert to DataFrame
        df = pd.DataFrame.from_records(
            iter(cursor), 
            columns=[col[0] for col in cursor.description]
        )
        
        # Generate visualization based on the query
        chart_img = self._generate_chart(df, query_text)
        
        # Generate text analysis
        analysis = self._generate_analysis(df, query_text)
        
        # Return results
        return {
            "data": df.to_dict(orient="records"),
            "response": analysis,
            "chart": chart_img
        }
        
    def _generate_chart(self, df: pd.DataFrame, query_text: str) -> str:
        """Generate an appropriate chart based on the data and query"""
        # Simple logic to determine chart type
        if "trend" in query_text.lower() or "over time" in query_text.lower():
            # Create time series chart
            plt.figure(figsize=(10, 6))
            
            # If we have time-based columns
            if "YEAR" in df.columns and "QUARTER" in df.columns:
                # Create period label (e.g., "2022-Q1")
                df["PERIOD"] = df["YEAR"].astype(str) + "-Q" + df["QUARTER"].astype(str)
                
                # Plot relevant metrics
                for col in df.columns:
                    if col not in ["YEAR", "QUARTER", "PERIOD"] and df[col].dtype in [float, int]:
                        plt.plot(df["PERIOD"], df[col], marker='o', label=col)
                
                plt.title(f"NVIDIA Metrics Over Time")
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
            
        else:
            # Create bar chart of recent metrics
            plt.figure(figsize=(10, 6))
            
            # Get the most recent period
            if len(df) > 0:
                recent_df = df.iloc[-1:]
                
                # Plot bar chart of numeric columns
                numeric_cols = [col for col in recent_df.columns 
                               if col not in ["YEAR", "QUARTER", "PERIOD"] 
                               and recent_df[col].dtype in [float, int]]
                
                if numeric_cols:
                    recent_df[numeric_cols].transpose().plot(kind='bar', legend=False)
                    plt.title(f"NVIDIA Recent Financial Metrics")
                    plt.tight_layout()
        
        # Save figure to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 for embedding in response
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
        
    def _generate_analysis(self, df: pd.DataFrame, query_text: str) -> str:
        """Generate text analysis of financial metrics"""
        # Convert dataframe to string representation
        if len(df) > 0:
            data_str = df.to_string()
        else:
            data_str = "No data available for the specified filters."
        
        # Create prompt for analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a financial analyst specializing in NVIDIA.
            Analyze the following financial metrics from Snowflake and answer the query.
            Provide insights about trends, notable changes, and implications.
            Only use the information in the provided data.
            """),
            ("human", "Financial metrics data:\n{data}\n\nQuery: {query}")
        ])
        
        # Generate analysis
        chain = prompt | self.llm
        response = chain.invoke({
            "data": data_str,
            "query": query_text
        })
        
        return response.content