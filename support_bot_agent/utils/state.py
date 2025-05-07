from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import AnyMessage, add_messages

# User info is used to pass custom params, as user confirmation or passenger_id
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str