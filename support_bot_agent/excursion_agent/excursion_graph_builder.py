from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.excursion_agent.excursion_agent import book_excursion_safe_tools, book_excursion_runnable, \
    book_excursion_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, pop_dialog_state
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback


def route_book_excursion(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return "book_excursion_safe_tools"
    return "book_excursion_sensitive_tools"

def create_excursion_subgraph(builder: StateGraph):
    builder.add_node(
        "enter_book_excursion",
        create_entry_node("Trip Recommendation Assistant", "book_excursion"),
    )
    builder.add_node("book_excursion", Assistant(book_excursion_runnable))
    builder.add_edge("enter_book_excursion", "book_excursion")
    builder.add_node(
        "book_excursion_safe_tools",
        create_tool_node_with_fallback(book_excursion_safe_tools),
    )
    builder.add_node(
        "book_excursion_sensitive_tools",
        create_tool_node_with_fallback(book_excursion_sensitive_tools),
    )

    builder.add_edge("book_excursion_sensitive_tools", "book_excursion")
    builder.add_edge("book_excursion_safe_tools", "book_excursion")
    builder.add_conditional_edges(
        "book_excursion",
        route_book_excursion,
        ["book_excursion_safe_tools", "book_excursion_sensitive_tools", "leave_skill", END],
    )