from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import TavilySearchResults



from langchain_experimental.utilities import PythonREPL
import json
from langchain_ollama import ChatOllama


tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

@tool
def json_validator(string: str):
    """Validates if the given string is a correctly formatted JSON."""
    try:
        json.loads(string)
        return "Valid JSON"
    except json.JSONDecodeError as e:
        return "Invalid JSON: " +  str(e)

llmOllama = ChatOllama(
    model="llama3.2:latest",
    base_url="http://host.docker.internal:11434",

    temperature=0,
    # other params...
)

research_agent = create_react_agent(
    llmOllama,
    tools=[tavily_search],
    state_modifier="You are best at researching a topic. You should do a thorough research on the given topic."
)

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

graph_builder = StateGraph(GraphState)
graph_builder.add_node("research_node", research_agent)

graph_builder.add_edge(START, "research_node")

# end todo
graph = graph_builder.compile()