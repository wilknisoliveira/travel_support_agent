from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition

from support_bot_agent.agent import Assistant, assistant_runnable, tools
from support_bot_agent.utils.state import State
from support_bot_agent.utils.tools.flights_tools import fetch_user_flight_information
from support_bot_agent.utils.utilities import create_tool_node_with_feedback

def user_info(state: State):
    return {
        "user_info": fetch_user_flight_information.invoke({})
    }

builder = StateGraph(State)
builder.add_node("fetch_user_flight_information", user_info)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_feedback(tools))
builder.add_edge(START, "fetch_user_flight_information")
# Get the user infor at begging
builder.add_edge("fetch_user_flight_information", "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"]
)

print("Graph Mermaid Code: ")
try:
    print(graph.get_graph().draw_mermaid())
    print("\n")
except Exception as e:
    print(f'Display error!: {e}')