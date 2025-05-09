from typing import Final

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, LEAVE_SKILL
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback
from support_bot_agent.flight_booking_agent.flight_booking_agent import update_flight_safe_tools, \
    update_flight_runnable, update_flight_sensitive_tools

UPDATE_FLIGHT_SAFE_TOOLS = "update_flight_safe_tools"
UPDATE_FLIGHT_SENSITIVE_TOOLS = "update_flight_sensitive_tools"
ENTER_UPDATE_FLIGHT = "enter_update_flight"
UPDATE_FLIGHT: Final = "update_flight"

def route_update_flight(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return LEAVE_SKILL
    safe_tool_names = [t.name for t in update_flight_safe_tools]
    if all(tc["name"] in safe_tool_names for tc in tool_calls):
        return UPDATE_FLIGHT_SAFE_TOOLS
    return UPDATE_FLIGHT_SENSITIVE_TOOLS
def create_flight_subgraph(builder: StateGraph):
    builder.add_node(
        ENTER_UPDATE_FLIGHT,
        create_entry_node("Flight Updates & Booking Assistant", "update_flight")
    )
    builder.add_node(UPDATE_FLIGHT, Assistant(update_flight_runnable))
    builder.add_edge(ENTER_UPDATE_FLIGHT, UPDATE_FLIGHT)
    builder.add_node(UPDATE_FLIGHT_SENSITIVE_TOOLS, create_tool_node_with_fallback(update_flight_sensitive_tools))
    builder.add_node(UPDATE_FLIGHT_SAFE_TOOLS, create_tool_node_with_fallback(update_flight_safe_tools))
    builder.add_edge(UPDATE_FLIGHT_SENSITIVE_TOOLS, UPDATE_FLIGHT)
    builder.add_edge(UPDATE_FLIGHT_SAFE_TOOLS, UPDATE_FLIGHT)
    builder.add_conditional_edges(
        UPDATE_FLIGHT,
        route_update_flight,
        [UPDATE_FLIGHT_SENSITIVE_TOOLS, UPDATE_FLIGHT_SAFE_TOOLS, LEAVE_SKILL, END]
    )
