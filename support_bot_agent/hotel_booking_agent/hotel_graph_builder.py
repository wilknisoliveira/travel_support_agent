from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.hotel_booking_agent.hotel_booking_agent import book_hotel_safe_tools, book_hotel_runnable, \
    book_hotel_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, pop_dialog_state
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback

def route_book_hotel(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_hotel_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_hotel_safe_tools"
    return "book_hotel_sensitive_tools"

def create_hotel_subgraph(builder: StateGraph):
    builder.add_node(
        "enter_book_hotel", create_entry_node("Hotel Booking Assistant", "book_hotel")
    )
    builder.add_node("book_hotel", Assistant(book_hotel_runnable))
    builder.add_edge("enter_book_hotel", "book_hotel")
    builder.add_node(
        "book_hotel_safe_tools",
        create_tool_node_with_fallback(book_hotel_safe_tools),
    )
    builder.add_node(
        "book_hotel_sensitive_tools",
        create_tool_node_with_fallback(book_hotel_sensitive_tools),
    )


    builder.add_edge("book_hotel_sensitive_tools", "book_hotel")
    builder.add_edge("book_hotel_safe_tools", "book_hotel")
    builder.add_conditional_edges(
        "book_hotel",
        route_book_hotel,
        ["leave_skill", "book_hotel_safe_tools", "book_hotel_sensitive_tools", END],
    )