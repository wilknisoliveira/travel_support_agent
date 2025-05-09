from typing import Final

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph

from support_bot_agent.excursion_agent.excursion_agent import book_excursion_safe_tools, book_excursion_runnable, \
    book_excursion_sensitive_tools
from support_bot_agent.utils.agent import CompleteOrEscalate, Assistant
from support_bot_agent.utils.state import State, LEAVE_SKILL
from support_bot_agent.utils.utilities import create_entry_node, create_tool_node_with_fallback

BOOK_EXCURSION_SAFE_TOOLS = "book_excursion_safe_tools"
BOOK_EXCURSION_SENSITIVE_TOOLS = "book_excursion_sensitive_tools"
ENTER_BOOK_EXCURSION = "enter_book_excursion"
BOOK_EXCURSION: Final = "book_excursion"

def route_book_excursion(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == CompleteOrEscalate.__name__ for tc in tool_calls)
    if did_cancel:
        return LEAVE_SKILL
    tool_names = [t.name for t in book_excursion_safe_tools]
    if all(tc["name"] in tool_names for tc in tool_calls):
        return BOOK_EXCURSION_SAFE_TOOLS
    return BOOK_EXCURSION_SENSITIVE_TOOLS

def create_excursion_subgraph(builder: StateGraph):
    builder.add_node(
        ENTER_BOOK_EXCURSION,
        create_entry_node("Trip Recommendation Assistant", BOOK_EXCURSION),
    )
    builder.add_node(BOOK_EXCURSION, Assistant(book_excursion_runnable))
    builder.add_edge(ENTER_BOOK_EXCURSION, BOOK_EXCURSION)
    builder.add_node(
        BOOK_EXCURSION_SAFE_TOOLS,
        create_tool_node_with_fallback(book_excursion_safe_tools),
    )
    builder.add_node(
        BOOK_EXCURSION_SENSITIVE_TOOLS,
        create_tool_node_with_fallback(book_excursion_sensitive_tools),
    )

    builder.add_edge(BOOK_EXCURSION_SENSITIVE_TOOLS, BOOK_EXCURSION)
    builder.add_edge(BOOK_EXCURSION_SAFE_TOOLS, BOOK_EXCURSION)
    builder.add_conditional_edges(
        BOOK_EXCURSION,
        route_book_excursion,
        [BOOK_EXCURSION_SAFE_TOOLS, BOOK_EXCURSION_SENSITIVE_TOOLS, LEAVE_SKILL, END]
    )