import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams, McpToolset
from google.genai import types

load_dotenv()
tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://bigquery.googleapis.com/mcp",
        headers={
            "Authorization": "Bearer " + os.environ.get("GCP_ACCESS_TOKEN"),
        },
        timeout=600.0,
    ),
    tool_filter=[
        "get_table_info",
    ],

)

text_to_sql_agent1 = LlmAgent(
    name="TextToSQL_Agent",
    model="gemini-2.5-flash-lite",
    description="Agent that generate SQL queries for natural language questions.",
    instruction="""You are a bigquery query expert who provides a valid Google SQL query for a user's question about the data in a table.
                    From the user's query:
                    1. Identify the table ID, the project ID, and the dataset ID from the user's query.
                    2. Use the `get_table_info` tool to find the information about the table.
                    3. Use the table information to generate a SQL query that answers the user's question.
                    4. Return the generated SQL query to the user. *Do not try to execute it*.
                    
                    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,  # More deterministic output
        max_output_tokens=1000,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            )
        ],
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                initial_delay=1.0,
                attempts=10,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
            timeout=120 * 1000,
        )
    ),
    tools=[tools],
    output_key="sql_query"
)
