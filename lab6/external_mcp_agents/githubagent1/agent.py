from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import StreamableHTTPConnectionParams, McpToolset
from google.genai import types

load_dotenv()
"""Creates an ADK Agent equipped with tools from the MCP Server."""
tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://api.githubcopilot.com/mcp/",
        headers={
            "X-MCP-Toolsets": "repos,issues",
            "X-MCP-Readonly": "true",
            "X-MCP-Lockdown": "false",
            "Authorization": "Bearer " + "YOUR_Github_ACCESS_TOKEN(DO Not Put in a Public Repo)",
        },
        timeout=600.0,
    ),
    # Read only tools
    tool_filter=[
        "search_repositories",
        "search_issues",
        "list_issues",
        "get_issue",
        "list_pull_requests",
        "get_pull_request",
    ],

)

root_agent = LlmAgent(
    name="Github_Agent",
    model="gemini-2.5-flash-lite",
    description="Agent that support quering Github repositories and issues.",
    instruction="You help users with the information about Github repositories and issues.",
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
