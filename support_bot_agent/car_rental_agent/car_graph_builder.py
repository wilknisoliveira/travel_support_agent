from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.car_rental_agent.car_rental_agent import book_car_rental_safe_tools, book_car_rental_runnable, \
    book_car_rental_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, pop_dialog_state
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback


def route_book_car_rental(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    safe_tool_names = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_tool_names for tc in tool_calls):
        return "book_car_rental_safe_tools"
    return "book_car_rental_sensitive_tools"

def create_car_rental_subgraph(builder: StateGraph):
    builder.add_node(
        "enter_book_car_rental",
        create_entry_node("Car Rental Assistant", "book_car_rental"),
    )
    builder.add_node("book_car_rental", Assistant(book_car_rental_runnable))
    builder.add_edge("enter_book_car_rental", "book_car_rental")
    builder.add_node(
        "book_car_rental_safe_tools",
        create_tool_node_with_fallback(book_car_rental_safe_tools),
    )
    builder.add_node(
        "book_car_rental_sensitive_tools",
        create_tool_node_with_fallback(book_car_rental_sensitive_tools),
    )

    builder.add_edge("book_car_rental_sensitive_tools", "book_car_rental")
    builder.add_edge("book_car_rental_safe_tools", "book_car_rental")
    builder.add_conditional_edges(
        "book_car_rental",
        route_book_car_rental,
        [
            "book_car_rental_safe_tools",
            "book_car_rental_sensitive_tools",
            "leave_skill",
            END,
        ],
    )