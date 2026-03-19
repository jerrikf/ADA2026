import datetime
from zoneinfo import ZoneInfo
import logging

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)
# 1. Create the server
mcp = FastMCP(name="Weather Info MCP Server")


# 2. Add a tool
@mcp.tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """

    print(f"get_weather: Received city '{city}'")
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


# 2. Add a tool
@mcp.tool
def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    print(f"get_current_time: Received city '{city}'")

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
        except Exception  as e:
            print(e)
            # This message IS sent to client
            raise ToolError("Error getting current time")

    return {
        "status": "error",
        "error_message": f"Time information for '{city}' unavailable."
    }


# 3. Add a static resource
@mcp.resource("resource://config")
def get_config() -> dict:
    """Provides the application's configuration."""
    return {"version": "1.0", "author": "ADA"}


# 4. Add a resource template for dynamic content
@mcp.resource("greetings://{name}")
def personalized_greeting(name: str) -> str:
    """Generates a personalized greeting for the given name."""
    return f"Hello, {name}! Welcome to the weather MCP server."

# Basic prompt returning a string
@mcp.prompt
def ask_about_topic(city: str) -> str:
    """Provide the current time of a place."""
    return f"Can you please give me the time of '{city}'?"

# 5. Make the server runnable
if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)  # HTTP Transport (Streamable), server url  http://IP:8000/mcp
    # mcp.run()  # STDIO
