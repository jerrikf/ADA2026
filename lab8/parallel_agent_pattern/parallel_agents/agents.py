from google.adk.tools import google_search, ToolContext, LongRunningFunctionTool
from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from dotenv import load_dotenv
from typing import Any
from google.genai import types

load_dotenv()
# Content Credit https://codelabs.developers.google.com/adkcourse/instructions#3
# Specialist Agent 1
museum_finder_agent = Agent(
    name="museum_finder_agent", model="gemini-2.5-flash-lite", tools=[google_search],
    instruction="You are a museum expert. Find the best museum based on the user's query. Output only the museum's name.",
    output_key="museum_result"
)

# Specialist Agent 2
concert_finder_agent = Agent(
    name="concert_finder_agent", model="gemini-2.5-flash-lite", tools=[google_search],
    instruction="You are an events guide. Find a concert based on the user's query. Output only the concert name and artist.",
    output_key="concert_result"
)

# We can reuse our foodie_agent for the third parallel task!
# Just need to give it a new output_key for this workflow.
# restaurant_finder_agent = foodie_agent.copy(update={"output_key": "restaurant_result"})
restaurant_finder_agent = Agent(
    name="restaurant_finder_agent",
    model="gemini-2.5-flash-lite",
    tools=[google_search],
    instruction="""You are an expert food critic. Your goal is to find the best restaurant based on a user's request.

    When you recommend a place, you must output *only* the name of the establishment.
    For example, if the best sushi is at 'Jin Sho', you should output only: Jin Sho
    """,
    output_key="restaurant_result"  # Set the correct output key for this workflow
)

#  The ParallelAgent runs all three specialists at once
parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    sub_agents=[museum_finder_agent, concert_finder_agent, restaurant_finder_agent]
)

# Agent to synthesize the parallel results
synthesis_agent = Agent(
    name="synthesis_agent", model="gemini-2.5-flash-lite",
    instruction="""You are a helpful assistant. Combine the following research results into a clear, bulleted list for the user.
    - Museum: {museum_result}
    - Concert: {concert_result}
    - Restaurant: {restaurant_result}
    """,
    output_key="synthesis_result"
)


# # The SequentialAgent runs the parallel search, then the synthesis
parallel_planner_agent = SequentialAgent(
    name="parallel_planner_agent",
    sub_agents=[parallel_research_agent, synthesis_agent],
    description="A workflow that finds multiple things in parallel and then summarizes the results."
)

root_agent = parallel_planner_agent

# ******** Human-in-the-loop Demo ****************
# def confirm_recommendation(museum: str, concert: str, restaurant: str) -> dict[str, str]:
#     """Confirm the recommendation."""
#     print(f"Confirm recommendation for {museum} and {concert}.")
#     return {
#         'status': 'Recommended',
#     }
#
#
# def ask_for_approval(
#         museum: str, concert: str, restaurant: str, tool_context: ToolContext
# ) -> dict[str, Any]:
#     """Ask for approval for the travel selection."""
#     return {
#         'status': 'pending',
#         'museum': museum,
#         'concert': concert,
#         'restaurant': restaurant,
#         'ticketId': 'travel-ticket-001',
#     }
#
#
# selection_agent = Agent(
#     model='gemini-2.5-flash',
#     name='selection_agent',
#     instruction="""
#       You are an agent whose job is to recommend museum, concert, and restaurant to the travel manager.
#       Select a single museum, a single concert, and a single restaurant from {synthesis_result} and ask for approval
#       from the travel manager. If the manager approves, you will
#       call confirm_recommendation() to recommend the selection to the user. If the manager
#       rejects, you will inform the user of the rejection.
# """,
#     tools=[confirm_recommendation, LongRunningFunctionTool(func=ask_for_approval)],
#     generate_content_config=types.GenerateContentConfig(temperature=0.1),
# )
#
# parallel_planner_agent2 = SequentialAgent(
#     name="parallel_planner_agent",
#     sub_agents=[parallel_research_agent, synthesis_agent, selection_agent],
#     description="A workflow that finds multiple things in parallel and then summarizes the results."
# )
#
# root_agent = parallel_planner_agent2
