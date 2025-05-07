from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

from support_bot_agent.utils.state import State
from support_bot_agent.utils.tools.car_rental_tools import (
    search_car_rentals,
    book_car_rental,
    update_car_rental,
    cancel_car_rental
)
from support_bot_agent.utils.tools.excursions_tools import (
    search_trip_recommendations,
    book_excursion,
    update_excursion,
    cancel_excursion
)
from support_bot_agent.utils.tools.flights_tools import (
    fetch_user_flight_information,
    search_flights,
    update_ticket_to_new_flight,
    cancel_ticket
)
from support_bot_agent.utils.tools.hotel_tools import (
    search_hotels,
    book_hotel,
    update_hotel,
    cancel_hotel
)
from support_bot_agent.utils.tools.lookup_company_policy_tools import lookup_policy


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

llm = ChatGoogleGenerativeAI(temperature=1, model="gemini-2.0-flash-001")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for Swiss Airlines. "
            " Use the provided tools to search for flights, company policies, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}."
        ),
        ("placeholder", "{messages}")
    ]
).partial(time=datetime.now())

safe_tools = [
    TavilySearchResults(max_results=1),
    fetch_user_flight_information,
    search_flights,
    lookup_policy,
    search_car_rentals,
    search_hotels,
    search_trip_recommendations,
]

sensitive_tools = [
    update_ticket_to_new_flight,
    cancel_ticket,
    book_car_rental,
    update_car_rental,
    cancel_car_rental,
    book_hotel,
    update_hotel,
    cancel_hotel,
    book_excursion,
    update_excursion,
    cancel_excursion,
]

sensitive_tool_names = {t.name for t in sensitive_tools}

assistant_runnable = primary_assistant_prompt | llm.bind_tools(safe_tools + sensitive_tools)