from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams, McpToolset
from google.genai import types

load_dotenv()
tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="http://localhost:5002/mcp", timeout=600.0
    ),
    # Select only the tools neededs
    tool_filter=[
        "create_order"
    ]
)

order_agent = LlmAgent(
    name="OrderAgent",
    model="gemini-2.5-flash-lite",
    description="Agent that create order records. ",
    instruction="""
        You record orders. Analyze the user intent. The user may ask you to do several tasks. However, you *must only* do the following tasks.   
        **Task:**
        1. If a user has asked to create or record an order, use the tool 'create_order' to do so. Do not do "anything else"

        **Output:**
        Output *only* the result from the tool calls.
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
    output_key="order_response"
)
