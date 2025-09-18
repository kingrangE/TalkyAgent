from typing import TypedDict,Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
class TalkyState(TypedDict):
    messages : Annotated[list, add_messages]
    llm : ChatOpenAI
    