from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from support_bot_agent.agent import Assistant, assistant_runnable, sensitive_tools, sensitive_tool_names, safe_tools
from support_bot_agent.utils.state import State
from support_bot_agent.utils.tools.flights_tools import fetch_user_flight_information
from support_bot_agent.utils.utilities import create_tool_node_with_feedback

def user_info(state: State):
    return {
        "user_info": fetch_user_flight_information.invoke({})
    }

# Define a custom route tools condition
def route_tools(state: State):
    next_node = tools_condition(state)
    # If no tools are invoked, return to the user
    if next_node == END:
        return END
    ai_message = state["messages"][-1]
    first_tool_call = ai_message.tool_calls[0]
    if first_tool_call["name"] in sensitive_tool_names:
        return "sensitive_tools"
    return "safe_tools"


builder = StateGraph(State)
builder.add_node("fetch_user_info", user_info)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("safe_tools", create_tool_node_with_feedback(safe_tools))
builder.add_node("sensitive_tools", create_tool_node_with_feedback(sensitive_tools))

builder.add_edge(START, "fetch_user_info")
# Get the user infor at begging
builder.add_edge("fetch_user_info", "assistant")
# Use the custom instead of tools_condition
builder.add_conditional_edges(
    "assistant",
    route_tools,
    ["safe_tools", "sensitive_tools", END]
)
builder.add_edge("safe_tools", "assistant")
builder.add_edge("sensitive_tools", "assistant")

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_tools"]
)

print("Graph Mermaid Code: ")
try:
    print(graph.get_graph().draw_mermaid())
    print("\n")
except Exception as e:
    print(f'Display error!: {e}')