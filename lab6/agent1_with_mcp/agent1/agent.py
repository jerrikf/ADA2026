from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams, McpToolset
from google.genai import types

load_dotenv()
"""Creates an ADK Agent equipped with tools from the MCP Server."""
tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:8000/mcp", timeout=120.0
    ),
    # Optional: Filter which tools from the MCP server are exposed
    # tool_filter=['list_directory', 'read_file']
)

root_agent = LlmAgent(
    name="weather_time_agent",
    model="gemini-2.5-flash-lite",
    description="Agent that provides weather and time information for cities.",
    instruction="You help users with time and weather information for various cities.",
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
)
