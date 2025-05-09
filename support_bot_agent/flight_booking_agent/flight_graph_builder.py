from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, pop_dialog_state
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback
from support_bot_agent.flight_booking_agent.flight_booking_agent import update_flight_safe_tools, \
    update_flight_runnable, update_flight_sensitive_tools


def route_update_flight(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_tool_names = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_tool_names for tc in tool_calls):
        return "update_flight_safe_tools"
    return "update_flight_sensitive_tools"

def create_flight_subgraph(builder: StateGraph):
    builder.add_node(
        "enter_update_flight",
        create_entry_node("Flight Updates & Booking Assistant", "update_flight")
    )
    builder.add_node("update_flight", Assistant(update_flight_runnable))
    builder.add_edge("enter_update_flight", "update_flight")

    builder.add_node("update_flight_sensitive_tools", create_tool_node_with_fallback(update_flight_sensitive_tools))
    builder.add_node("update_flight_safe_tools", create_tool_node_with_fallback(update_flight_safe_tools))
    builder.add_edge("update_flight_sensitive_tools", "update_flight")
    builder.add_edge("update_flight_safe_tools", "update_flight")
    builder.add_conditional_edges(
        "update_flight",
        route_update_flight,
        ["update_flight_sensitive_tools", "update_flight_safe_tools", "leave_skill", END]
    )

    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")