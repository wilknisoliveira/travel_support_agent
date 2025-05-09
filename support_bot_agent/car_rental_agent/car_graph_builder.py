from typing import Final

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.car_rental_agent.car_rental_agent import book_car_rental_safe_tools, book_car_rental_runnable, \
    book_car_rental_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, LEAVE_SKILL
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback

BOOK_CAR_RENTAL_SAFE_TOOLS = "book_car_rental_safe_tools"
BOOK_CAR_RENTAL_SENSITIVE_TOOLS = "book_car_rental_sensitive_tools"
ENTER_BOOK_CAR_RENTAL = "enter_book_car_rental"
BOOK_CAR_RENTAL: Final = "book_car_rental"

def route_book_car_rental(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return LEAVE_SKILL
    safe_tool_names = [t.name for t in book_car_rental_safe_tools]
    if all(tc["name"] in safe_tool_names for tc in tool_calls):
        return BOOK_CAR_RENTAL_SAFE_TOOLS
    return BOOK_CAR_RENTAL_SENSITIVE_TOOLS

def create_car_rental_subgraph(builder: StateGraph):
    builder.add_node(
        ENTER_BOOK_CAR_RENTAL,
        create_entry_node("Car Rental Assistant", BOOK_CAR_RENTAL),
    )
    builder.add_node(BOOK_CAR_RENTAL, Assistant(book_car_rental_runnable))
    builder.add_edge(ENTER_BOOK_CAR_RENTAL, BOOK_CAR_RENTAL)
    builder.add_node(
        BOOK_CAR_RENTAL_SAFE_TOOLS,
        create_tool_node_with_fallback(book_car_rental_safe_tools),
    )
    builder.add_node(
        BOOK_CAR_RENTAL_SENSITIVE_TOOLS,
        create_tool_node_with_fallback(book_car_rental_sensitive_tools),
    )

    builder.add_edge(BOOK_CAR_RENTAL_SENSITIVE_TOOLS, BOOK_CAR_RENTAL)
    builder.add_edge(BOOK_CAR_RENTAL_SAFE_TOOLS, BOOK_CAR_RENTAL)
    builder.add_conditional_edges(
        BOOK_CAR_RENTAL,
        route_book_car_rental,
        [
            BOOK_CAR_RENTAL_SAFE_TOOLS,
            BOOK_CAR_RENTAL_SENSITIVE_TOOLS,
            LEAVE_SKILL,
            END,
        ],
    )