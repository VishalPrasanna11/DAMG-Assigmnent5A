from dotenv import load_dotenv
import os
import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

# Load credentials
load_dotenv()

# ✅ Query helper
def query_snowflake(query: str) -> pd.DataFrame:
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA")
    )
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Enhanced dynamic chart generator
def generate_chart(df, metric="MARKETCAP") -> str:
    df = df.sort_values("ASOFDATE")
    df["ASOFDATE"] = pd.to_datetime(df["ASOFDATE"])

    # Build the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["ASOFDATE"], df[metric], marker="o", linewidth=2, color="#007acc")

    # Title and axes
    ax.set_title(f"NVIDIA {metric} Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
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
            f'{row[metric]/1e9:.1f}B',
            (row["ASOFDATE"], row[metric]),
            textcoords="offset points",
            xytext=(0, 8),
            ha='center',
            fontsize=8,
            color='gray'
        )

    # Save the chart
    plt.tight_layout()
    chart_path = f"{metric.lower()}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return f"📊 Chart saved as {chart_path}"

# ✅ LangChain Tool
@tool
def get_nvidia_financials(input: str) -> str:
    """
    Get NVIDIA financials from Snowflake for a given year and quarter.
    Input: "year=2024, quarter=1"
    """
    try:
        year = input.split("year=")[1].split(",")[0].strip()
        quarter = input.split("quarter=")[1].strip()

        query = f"""
        SELECT * FROM NVIDIA_FINANCIALS 
        WHERE YEAR(ASOFDATE) = {year} AND QUARTER(ASOFDATE) = {quarter}
        """

        df = query_snowflake(query)

        if df.empty:
            return f"No data found for year {year} and quarter {quarter}."

        # Textual summary
        row = df.iloc[0]
        summary = (
            f"NVIDIA Financials for Q{quarter} {year}:\n"
            f"- ASOFDATE: {row['ASOFDATE']}\n"
            f"- ENTERPRISEVALUE: {row['ENTERPRISEVALUE']:,}\n"
            f"- MARKETCAP: {row['MARKETCAP']:,}\n"
            f"- PERATIO: {row['PERATIO']:.2f}\n"
            f"- PBRATIO: {row['PBRATIO']:.2f}\n"
            f"- PSRATIO: {row['PSRATIO']:.2f}\n"
            f"- PEGRATIO: {row['PEGRATIO']:.4f}\n"
            f"- FORWARDPERATIO: {row['FORWARDPERATIO']:.2f}"
        )

        # Save enhanced chart
        chart_msg = generate_chart(df, metric="MARKETCAP")

        return summary + f"\n\n{chart_msg}"

    except Exception as e:
        return f"Error parsing input or querying Snowflake: {e}"

# 🔮 Language Model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 🤖 Agent
agent = initialize_agent(
    tools=[get_nvidia_financials],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔁 Prompt user from terminal
user_prompt = input("Your question: ")
response = agent.invoke(user_prompt)
print(response["output"])
