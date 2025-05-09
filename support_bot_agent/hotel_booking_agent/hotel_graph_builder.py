from typing import Final

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.hotel_booking_agent.hotel_booking_agent import book_hotel_safe_tools, book_hotel_runnable, \
    book_hotel_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, pop_dialog_state, LEAVE_SKILL
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback

BOOK_HOTEL_SAFE_TOOLS = "book_hotel_safe_tools"
BOOK_HOTEL_SENSITIVE_TOOLS = "book_hotel_sensitive_tools"
ENTER_BOOK_HOTEL = "enter_book_hotel"
BOOK_HOTEL: Final = "book_hotel"

def route_book_hotel(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return LEAVE_SKILL
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return BOOK_HOTEL_SAFE_TOOLS
    return BOOK_HOTEL_SENSITIVE_TOOLS
def create_hotel_subgraph(builder: StateGraph):
    builder.add_node(
        ENTER_BOOK_HOTEL, create_entry_node("Hotel Booking Assistant", "book_hotel")
    )
    builder.add_node(BOOK_HOTEL, Assistant(book_hotel_runnable))
    builder.add_edge(ENTER_BOOK_HOTEL, BOOK_HOTEL)
    builder.add_node(
        BOOK_HOTEL_SAFE_TOOLS,
        create_tool_node_with_fallback(book_hotel_safe_tools),
    )
    builder.add_node(
        BOOK_HOTEL_SENSITIVE_TOOLS,
        create_tool_node_with_fallback(book_hotel_sensitive_tools),
    )


    builder.add_edge(BOOK_HOTEL_SENSITIVE_TOOLS, BOOK_HOTEL)
    builder.add_edge(BOOK_HOTEL_SAFE_TOOLS, BOOK_HOTEL)
    builder.add_conditional_edges(
        BOOK_HOTEL,
        route_book_hotel,
        [LEAVE_SKILL, BOOK_HOTEL_SAFE_TOOLS, BOOK_HOTEL_SENSITIVE_TOOLS, END],
    )