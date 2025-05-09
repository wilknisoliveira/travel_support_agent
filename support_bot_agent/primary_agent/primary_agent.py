from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults

from support_bot_agent.car_rental_agent.car_rental_agent import ToBookCarRental
from support_bot_agent.excursion_agent.excursion_agent import ToBookExcursion
from support_bot_agent.flight_booking_agent.flight_booking_agent import ToFlightBookingAssistant
from support_bot_agent.flight_booking_agent.flights_tools import search_flights
from support_bot_agent.hotel_booking_agent.hotel_booking_agent import ToHotelBookingAssistant

from support_bot_agent.primary_agent.lookup_company_policy_tools import lookup_policy

from support_bot_agent.utils.agent import llm

# Top-level assistant for general purpose and route specialized tasks
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            "Your primary role is to search for flight information and company policies to answer customer queries. "
            "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
            "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
            " Only the specialized assistants are given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user flight information:\n<Flights>\n{user_info}\n</Flights>"
            "\nCurrent time: {time}."
        ),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now)

primary_assistant_tools = [
    TavilySearchResults(max_results=1),
    search_flights,
    lookup_policy
]

assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools + [
        ToFlightBookingAssistant,
        ToBookCarRental,
        ToHotelBookingAssistant,
        ToBookExcursion
    ]
)