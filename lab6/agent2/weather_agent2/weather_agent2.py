import asyncio
import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.adk.agents.llm_agent import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.types import ThinkingConfig

APP_NAME = "weather_app"
USER_ID = "1234"
SESSION_ID = "session1234"
load_dotenv()


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "den bosch":
        return {
            "status": "success",
            "report": "The weather in Den Bosch is sunny with 15°C."
        }
    elif city.lower() == "kandy":
        return {
            "status": "success",
            "report": "The weather in New York is sunny with 30°C."
        }
    return {
        "status": "error",
        "error_message": f"Weather for '{city}' unavailable."
    }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg."""
    city_timezones = {
        "den bosch": "Europe/Amsterdam",
        "london": "Europe/London",
        "kandy": "Asia/Colombo",
        "paris": "Europe/Paris"
    }

    if city.lower() in city_timezones:
        try:
            tz = ZoneInfo(city_timezones[city.lower()])
            now = datetime.datetime.now(tz)
            return {
                "status": "success",
                "report": f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
            }
        except Exception as e:
            print(e)
            pass

    return {
        "status": "error",
        "error_message": f"Time information for '{city}' unavailable."
    }


# Step 1: Create a ThinkingConfig
thinking_config = ThinkingConfig(
    include_thoughts=True,  # Ask the model to include its thoughts in the response
    thinking_budget=512  # Limit the 'thinking' to 256 tokens (adjust as needed)
)
print("ThinkingConfig:", thinking_config)

# Step 2: Instantiate BuiltInPlanner
planner = BuiltInPlanner(
    thinking_config=thinking_config
)
print("BuiltInPlanner created.")

# Step 3: Wrap the planner in an LlmAgent
weather_agent2 = LlmAgent(
    model="gemini-2.5-flash-lite",  # Set your model name
    name="weather_and_time_agent",
    instruction="You are an agent that returns time and weather",
    planner=planner,
    tools=[get_weather, get_current_time],
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
    )
)

# Session and Runner
session_service = InMemorySessionService()

# Use asyncio.run() to execute the async session creation**
session = asyncio.run(session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID,
    session_id=SESSION_ID
))
print(session)
runner = Runner(agent=weather_agent2, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        print(f"Content: {event.content}")
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)


call_agent("If it's raining in Den Bosch right now, what is the current temperature?")
