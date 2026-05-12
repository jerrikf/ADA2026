from google.adk.agents import SequentialAgent, ParallelAgent
from google.adk.agents.llm_agent import LlmAgent

from agents.delivery_agent import delivery_agent
from agents.inventory_agent import inventory_agent
from agents.order_agent import order_agent

# ************ Pattern Coordinator ******************
# Orchestrate the Order Approval Workflow
order_creation_workflow_agent = SequentialAgent(
    name="OrderManagementWorkflowAgent",
    sub_agents=[order_agent, inventory_agent]
)

# The Coordinator (Dispatcher)
coordinator_agent = LlmAgent(
    name="CoordinatorAgent",
    # The instructions guide the routing logic
    instruction="Analyze user intent. Route order management requests to OrderManagementWorkflowAgent and order delivery management requests to DeliveryServiceAgent.",
    sub_agents=[order_creation_workflow_agent, delivery_agent]
)
root_agent = coordinator_agent

# ********** Pattern Fan-out ************************
# # Fan-out (The Swarm)
# parallel_order_delivery = ParallelAgent(
#     name="OrderAndDeliveryManager",
#     sub_agents=[order_agent, inventory_agent, delivery_agent]
# )
#
# # Gather/Synthesize
# parallel_order_response_aggregator = LlmAgent(
#     name="OrderAndDeliveryResponseAggregator",
#     instruction="""You are a helpful assistant. Combine the following tool results into a clear, bulleted list for the user.
#     - Order Response: {order_response}
#     - Inventory Response: {inventory_response}
#     - Delivery Response: {delivery_response}
#     """)
#
# # Wrap in a sequence
# parallel_workflow_agent = SequentialAgent(name="WorkflowOrganizer",
#                              description="A workflow that finds multiple things in parallel and then summarizes the results.",
#                              sub_agents=[parallel_order_delivery, parallel_order_response_aggregator])
#
# root_agent = parallel_workflow_agent
