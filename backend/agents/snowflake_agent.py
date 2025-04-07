# backend/agents/snowflake_agent.py
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class SnowflakeAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Validate required environment variables
        required_vars = [
            "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_WAREHOUSE", "SNOWFLAKE_DATABASE", "SNOWFLAKE_SCHEMA"
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"{var} environment variable not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(temperature=0)
    
    def query(self, query_text: str, year: Optional[int] = None, quarter: Optional[int] = None) -> Dict[str, Any]:
        """
        Query Snowflake for NVIDIA financial data based on query text, year, and quarter.
        
        Args:
            query_text: The query text
            year: Optional year filter
            quarter: Optional quarter filter
            
        Returns:
            Dictionary with structured data, charts, and generated response
        """
        try:
            # Build SQL query with filters
            sql_query = "SELECT * FROM NVDA_FINANCIAL_DATA"
            
            where_clauses = []
            if year is not None:
                where_clauses.append(f"YEAR(Report_Date) = {year}")
            if quarter is not None:
                where_clauses.append(f"QUARTER(Report_Date) = {quarter}")
                
            if where_clauses:
                sql_query += " WHERE " + " AND ".join(where_clauses)
            
            # Query Snowflake
            df = self._query_snowflake(sql_query)
            
            if df.empty:
                return {
                    "response": f"No financial data found for NVIDIA with the specified filters.",
                    "chart": None,
                    "sources": ["Snowflake Database"]
                }
            
            # Generate chart if data is available
            chart_path = None
            if not df.empty:
                chart_path = self._generate_chart(df, "Market_Cap")
            
            # Generate analysis with LLM
            analysis = self._generate_analysis(df, query_text)
            
            return {
                "response": analysis,
                "chart": chart_path,
                "sources": ["Snowflake Database - NVDA_FINANCIAL_DATA"]
            }
            
        except Exception as e:
            return {
                "response": f"Error querying Snowflake: {str(e)}",
                "chart": None,
                "sources": []
            }
    
    def _query_snowflake(self, query: str) -> pd.DataFrame:
        """Execute query against Snowflake and return results as DataFrame"""
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA")
        )
        
        try:
            df = pd.read_sql(query, conn)
            return df
        finally:
            conn.close()
    
    def _generate_chart(self, df, metric="Market_Cap") -> str:
        """Generate chart for visualization"""
        # Convert dates if needed
        if 'Report_Date' in df.columns:
            df['Report_Date'] = pd.to_datetime(df['Report_Date'])
            date_col = 'Report_Date'
        elif 'ASOFDATE' in df.columns:
            df['ASOFDATE'] = pd.to_datetime(df['ASOFDATE'])
            date_col = 'ASOFDATE'
        else:
            # Find a date column
            date_cols = [col for col in df.columns if 'date' in col.lower()]
            if date_cols:
                date_col = date_cols[0]
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                return None
        
        # Find the metric column (case insensitive)
        metric_col = None
        for col in df.columns:
            if col.lower() == metric.lower():
                metric_col = col
                break
        
        if not metric_col:
            metric_options = [col for col in df.columns if 'cap' in col.lower() or 'value' in col.lower()]
            if metric_options:
                metric_col = metric_options[0]
            else:
                return None
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Build the figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[date_col], df[metric_col], marker="o", linewidth=2, color="#007acc")

        # Title and axes
        ax.set_title(f"NVIDIA {metric_col} Over Time", fontsize=14)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(metric_col, fontsize=12)
        plt.xticks(rotation=45)
        ax.grid(True, linestyle="--", alpha=0.6)

        # Format y-axis with billion/trillion scaling
        def billions(x, pos):
            if x >= 1e12:
                return f"${x*1.0/1e12:.1f}T"
            elif x >= 1e9:
                return f"${x*1.0/1e9:.1f}B"
            else:
                return f"${x:,.0f}"
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(billions))

        # Add data labels
        for i, row in df.iterrows():
            ax.annotate(
                f'{row[metric_col]/1e9:.1f}B',
                (row[date_col], row[metric_col]),
                textcoords="offset points",
                xytext=(0, 8),
                ha='center',
                fontsize=8,
                color='gray'
            )

        # Save the chart
        plt.tight_layout()
        chart_path = f"nvda_{metric_col.lower()}_chart.png"
        plt.savefig(chart_path)
        plt.close()
        
        return chart_path
        
    def _generate_analysis(self, df: pd.DataFrame, query_text: str) -> str:
        """Generate analysis of financial data using LLM"""
        # Format dataframe as string for context
        df_str = df.head(10).to_string()
        
        # Create prompt for analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a financial analyst specializing in NVIDIA.
            Analyze the provided financial data table to answer the user's query.
            Focus on key metrics, trends, and notable changes.
            Provide insights in a clear, concise manner suitable for financial analysis.
            """),
            ("human", """
            NVIDIA Financial Data:
            {data}
            
            User Query: {query}
            
            Provide a detailed analysis based on this financial data.
            """)
        ])
        
        # Generate analysis
        chain = prompt | self.llm
        response = chain.invoke({
            "data": df_str,
            "query": query_text
        })
        
        return response.content