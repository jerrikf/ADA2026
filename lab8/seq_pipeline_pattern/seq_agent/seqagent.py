from google.adk.agents import SequentialAgent, LlmAgent, LoopAgent
from google.adk.tools import exit_loop, AgentTool
from google.cloud.aiplatform.metadata.schema.system.context_schema import Pipeline

from seq_agent.seq_exe_agent import biquery_executor_agent
from seq_agent.table_info_agent import table_info_agent
from seq_agent.text_to_sql_agent import text_to_sql_agent1
from seq_agent.text_to_sql_agent_v2 import text_to_sql_agent2

# ********** Pattern 1: Sequential Pipeline Example **********
# Orchestrate the Sequential Pipeline
seq_agent = SequentialAgent(
    name="BiqQueryPipeline",
    sub_agents=[text_to_sql_agent1, biquery_executor_agent]
)

root_agent = seq_agent

# ********** Pattern 2: Sequential Pipeline with Critic Example **********

# The Critic
# bigquery_sql_critic = LlmAgent(
#     name="BiQuery_SQL_Critic",
#     instruction="""
#                 You are a bigquery query expert. Your task is to check if {sql_query} is valid bigquery SQL.
#                 **Task:**
#                 Use the table information in {table_schema} to verify that the SQL query includes valid fields.
#                 IF correct:
#                 You MUST output 'PASS' and call the 'exit_loop' function.
#                 ELSE (the critique contains actionable feedback):
#                 Output error details.
#                 """,
#     tools=[exit_loop],
#     output_key="feedback"
# )
#
# correction_loop = LoopAgent(
#     name="SQLValidationLoop",
#     # Agent order is crucial
#     sub_agents=[
#         text_to_sql_agent2, table_info_agent,
#         bigquery_sql_critic,
#     ],
#     max_iterations=2  # Limit loops
# )
#
# seq_agent2 = SequentialAgent(
#     name="BiqQueryPipelineWithCritic",
#     sub_agents=[correction_loop, biquery_executor_agent]
# )
#
# root_agent = seq_agent2


# ********** Pattern 3: Sequential Pipeline with Critic with Sub-Agent Workers **********

# The Critic
# bigquery_sql_critic2 = LlmAgent(
#     name="BiQuery_SQL_Critic",
#     instruction="""
#                 You are a bigquery query expert. Your task is to check if {sql_query} is valid bigquery SQL.
#                 **Task:**
#                 Use Table_Information_Provider_Agent to get the table information. Then, use that information to verify that the SQL query includes valid fields.
#                 IF correct:
#                 You MUST output 'PASS' and call the 'exit_loop' function.
#                 ELSE (the critique contains actionable feedback):
#                 Output error details.
#                 """,
#     tools=[exit_loop,AgentTool(table_info_agent)],
#     output_key="feedback"
# )
#
# correction_loop2 = LoopAgent(
#     name="SQLValidationLoop",
#     # Agent order is crucial
#     sub_agents=[
#         text_to_sql_agent2, bigquery_sql_critic2,
#     ],
#     max_iterations=2  # Limit loops
# )
#
# seq_agent3 = SequentialAgent(
#     name="BiqQueryPipelineWithCritic",
#     sub_agents=[correction_loop2, biquery_executor_agent]
# )
#
# root_agent = seq_agent3