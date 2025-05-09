from typing import Literal, Final

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition

from support_bot_agent.car_rental_agent.car_graph_builder import create_car_rental_subgraph, ENTER_BOOK_CAR_RENTAL, \
    BOOK_CAR_RENTAL, BOOK_CAR_RENTAL_SENSITIVE_TOOLS
from support_bot_agent.car_rental_agent.car_rental_agent import ToBookCarRental
from support_bot_agent.excursion_agent.excursion_agent import ToBookExcursion
from support_bot_agent.excursion_agent.excursion_graph_builder import create_excursion_subgraph, ENTER_BOOK_EXCURSION, \
    BOOK_EXCURSION, BOOK_EXCURSION_SENSITIVE_TOOLS
from support_bot_agent.flight_booking_agent.flight_booking_agent import ToFlightBookingAssistant
from support_bot_agent.flight_booking_agent.flight_graph_builder import create_flight_subgraph, ENTER_UPDATE_FLIGHT, \
    UPDATE_FLIGHT, UPDATE_FLIGHT_SENSITIVE_TOOLS
from support_bot_agent.hotel_booking_agent.hotel_booking_agent import ToHotelBookingAssistant
from support_bot_agent.hotel_booking_agent.hotel_graph_builder import create_hotel_subgraph, ENTER_BOOK_HOTEL, \
    BOOK_HOTEL, BOOK_HOTEL_SENSITIVE_TOOLS
from support_bot_agent.primary_agent.primary_agent import assistant_runnable, primary_assistant_tools
from support_bot_agent.utils.agent import Assistant
from support_bot_agent.utils.state import State, pop_dialog_state, LEAVE_SKILL
from support_bot_agent.flight_booking_agent.flights_tools import fetch_user_flight_information
from support_bot_agent.utils.utilities import create_tool_node_with_fallback

PRIMARY_ASSISTANT_TOOLS = "primary_assistant_tools"
PRIMARY_ASSISTANT: Final = "primary_assistant"
FETCH_USER_INFO = "fetch_user_info"

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
            return ENTER_UPDATE_FLIGHT
        elif tool_calls[0]["name"] == ToBookCarRental.__name__:
            return ENTER_BOOK_CAR_RENTAL
        elif tool_calls[0]["name"] == ToHotelBookingAssistant.__name__:
            return ENTER_BOOK_HOTEL
        elif tool_calls[0]["name"] == ToBookExcursion.__name__:
            return ENTER_BOOK_EXCURSION
        return PRIMARY_ASSISTANT_TOOLS
    raise ValueError("Invalid route")

# Each specialized agent can directly respond to the user
# When user responds, return to the currently active workflow
def route_to_workflow(
    state: State,
) -> Literal[
    PRIMARY_ASSISTANT,
    UPDATE_FLIGHT,
    BOOK_CAR_RENTAL,
    BOOK_HOTEL,
    BOOK_EXCURSION,
]:
    """If we are in a delegated state, route directly to the appropriate assistant."""
    dialog_state = state.get("dialog_state")
    if not dialog_state:
        return PRIMARY_ASSISTANT
    return dialog_state[-1]

builder = StateGraph(State)
builder.add_node(FETCH_USER_INFO, user_info)
# Get the user info at begging
builder.add_edge(START, FETCH_USER_INFO)

# Building subgraph
create_flight_subgraph(builder)
create_car_rental_subgraph(builder)
create_hotel_subgraph(builder)
create_excursion_subgraph(builder)

builder.add_node(LEAVE_SKILL, pop_dialog_state)
builder.add_edge(LEAVE_SKILL, PRIMARY_ASSISTANT)

builder.add_node(PRIMARY_ASSISTANT, Assistant(assistant_runnable))
builder.add_node(PRIMARY_ASSISTANT_TOOLS, create_tool_node_with_fallback(primary_assistant_tools))

# Use the custom instead of tools_condition
builder.add_conditional_edges(
    PRIMARY_ASSISTANT,
    route_primary_assistant,
    [
        ENTER_UPDATE_FLIGHT,
        ENTER_BOOK_CAR_RENTAL,
        ENTER_BOOK_HOTEL,
        ENTER_BOOK_EXCURSION,
        PRIMARY_ASSISTANT_TOOLS,
        END,
    ],
)
builder.add_edge(PRIMARY_ASSISTANT_TOOLS, PRIMARY_ASSISTANT)
builder.add_conditional_edges(FETCH_USER_INFO, route_to_workflow)

memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=[
        UPDATE_FLIGHT_SENSITIVE_TOOLS,
        BOOK_CAR_RENTAL_SENSITIVE_TOOLS,
        BOOK_HOTEL_SENSITIVE_TOOLS,
        BOOK_EXCURSION_SENSITIVE_TOOLS,
    ],
)

print("Graph Mermaid Code")
print("--------START---------")
try:
    print(graph.get_graph().draw_mermaid())
except Exception as e:
    print(f'Display error!: {e}')
finally:
    print("--------END---------")
    print("\n")