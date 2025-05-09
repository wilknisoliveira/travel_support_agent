from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from support_bot_agent.car_rental_agent.car_graph_builder import create_car_rental_subgraph
from support_bot_agent.car_rental_agent.car_rental_agent import ToBookCarRental
from support_bot_agent.excursion_agent.excursion_agent import ToBookExcursion
from support_bot_agent.excursion_agent.excursion_graph_builder import create_excursion_subgraph
from support_bot_agent.flight_booking_agent.flight_booking_agent import ToFlightBookingAssistant
from support_bot_agent.flight_booking_agent.flight_graph_builder import create_flight_subgraph
from support_bot_agent.hotel_booking_agent.hotel_booking_agent import ToHotelBookingAssistant
from support_bot_agent.hotel_booking_agent.hotel_graph_builder import create_hotel_subgraph
from support_bot_agent.primary_agent.primary_agent import assistant_runnable, primary_assistant_tools
from support_bot_agent.utils.agent import Assistant
from support_bot_agent.utils.state import State
from support_bot_agent.flight_booking_agent.flights_tools import fetch_user_flight_information
from support_bot_agent.utils.utilities import create_tool_node_with_fallback


def user_info(state: State):
    return {
        "user_info": fetch_user_flight_information.invoke({})
    }

def route_primary_assistant(state: State):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == ToFlightBookingAssistant.__name__:
            return "enter_update_flight"
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return "enter_book_car_rental"
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return "enter_book_hotel"
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return "enter_book_excursion"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")

# Each specialized agent can directly respond to the user
# When user responds, return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    "primary_assistant",
    "update_flight",
    "book_car_rental",
    "book_hotel",
    "book_excursion",
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return "primary_assistant"
    return dialog_state[-1]

builder = StateGraph(State)
builder.add_node("fetch_user_info", user_info)
# Get the user info at begging
builder.add_edge(START, "fetch_user_info")

# Building subgraph
create_flight_subgraph(builder)
create_car_rental_subgraph(builder)
create_hotel_subgraph(builder)
create_excursion_subgraph(builder)

builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node("primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools))

# Use the custom instead of tools_condition
builder.add_conditional_edges(
    "primary_assistant",
    route_primary_assistant,
    [
        "enter_update_flight",
        "enter_book_car_rental",
        "enter_book_hotel",
        "enter_book_excursion",
        "primary_assistant_tools",
        END,
    ],
)
builder.add_edge("primary_assistant_tools", "primary_assistant")
builder.add_conditional_edges("fetch_user_info", route_to_workflow)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[
        "update_flight_sensitive_tools",
        "book_car_rental_sensitive_tools",
        "book_hotel_sensitive_tools",
        "book_excursion_sensitive_tools",
    ],
)

print("Graph Mermaid Code: ")
try:
    print(graph.get_graph().draw_mermaid())
    print("\n")
except Exception as e:
    print(f'Display error!: {e}')