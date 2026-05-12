import os

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams, McpToolset
from google.genai import types

load_dotenv()

text_to_sql_agent2 = LlmAgent(
    name="TextToSQL_Agent_V2",
    model="gemini-2.5-flash-lite",
    description="Agent that generate SQL queries for natural language questions.",
    instruction="""You are a bigquery query expert who provides a valid Google SQL query for a user's question about the data in a table.
                   Return the generated SQL query to the user. Do not try to execute it. If you receive {feedback?}, fix the errors and generate again.                   
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
    output_key="sql_query"
)
