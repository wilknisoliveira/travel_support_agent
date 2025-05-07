import uuid
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

load_dotenv()

from support_bot_agent.db import update_dates, db
from support_bot_agent.graph import graph
from support_bot_agent.utils.utilities import print_event

tutorial_questions = [
        "Hi there, what time is my flight?",
        "Am i allowed to update my flight to something sooner? I want to leave later today.",
        "Update my flight to sometime next week then",
        "The next available option is great",
        "what about lodging and transportation?",
        "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
        "OK could you place a reservation for your recommended hotel? It sounds nice.",
        "yes go ahead and book anything that's moderate expense and has availability.",
        "Now for a car, what are my options?",
        "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
        "Cool so now what recommendations do you have on excursions?",
        "Are they available while I'm there?",
        "interesting - i like the museums, what options are there? ",
        "OK great pick one and book it for my second day there.",
    ]

if __name__ == '__main__':
    db = update_dates(db)
    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "passenger_id": "3442 587242",
            "thread_id": thread_id
        }
    }

    _printed = set()

    while True:
        question: str = input("\n\nUser input: ")

        if question.lower() == 'close':
            print("See you!")
            break

        events = graph.stream(
            {"messages": ("user", question)},
            config,
            stream_mode="values"
        )
        for event in events:
            print_event(event, _printed)

        snapshot = graph.get_state(config)
        while snapshot.next:
            try:
                user_input = input(
                    "Do you approve of the above actions? Type 'y' to continue;"
                    " otherwise, explain your requested changed.\n\n"
                )
            except:
                user_input = "y"

            if user_input.strip() == "y":
                # Continue
                result = graph.invoke(
                    None,
                    config
                )
            else:
                result = graph.invoke(
                    {
                        "messages": [
                            ToolMessage(
                                tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input."
                            )
                        ]
                    },
                    config
                )
            snapshot = graph.get_state(config)
