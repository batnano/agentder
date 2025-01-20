from langchain_ollama import ChatOllama
import requests
import json
from typing import Annotated
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import YoutubeLoader

class State(TypedDict):
    messages: Annotated[list, add_messages]

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")


graph_builder = StateGraph(State)

llmOllama = ChatOllama(
    model="qwen2.5:32b",
    base_url="https://ollama.batnano.fr",
    temperature=6,


)

def invoke_n8n_webhook(method, url, function_name, payload):
    """
    Helper function to make a GET or POST request to a webhook.

    Args:
        method (str): HTTP method ('GET' or 'POST')
        url (str): The API endpoint
        function_name (str): The name of the tool the AI agent invoked
        payload (dict): The payload for POST requests

    Returns:
        str: The API response in JSON format or an error message
    """
    headers = {
        "Content-Type": "application/json"
    }

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=payload)
        else:
            return f"Unsupported method: {method}"

        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Exception when calling {function_name}: {e}"

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~ n8n AI Agent Tool Functions ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




@tool(parse_docstring=True)
def get_youtube_transcript(url:str):
    """
        Gets the transcript of a youtube video. Good for summarizing youtube videos

        Args:
            url: A Youtube url

        Returns:
            Transcript of the youtube video.
    """
    loader = YoutubeLoader.from_youtube_url(
        url, add_video_info=False,language=["fr","en"]
    )
    return loader.load()

@tool(parse_docstring=True)
def check_table_availability(payload:dict):

    """
    Checks the availability of a table for a specified number of guests at a specific time and date.

    Args:
        payload: Dict of the customer info (hour in HH-MM format, date in YYYY-MM-DD format, guests number)

    Returns:
        Response from the webhook call indicating table availability.
    """
    return invoke_n8n_webhook(
        "POST",
        "https://batnano-n8n-4292efb2cdf2.herokuapp.com/webhook/0119b32c-b257-4bcb-a78c-a4016640b844",
        "check_table_availability",
        payload=payload
    )

@tool(parse_docstring=True)
def reserve_table(payload:dict):
    """
        Creates a reservation with the specified details for a table at a specific time and date for a specified number of guests.

        Args:
        payload: Dict of the customer infos (hour in HH-MM format, date in YYYY-MM-DD format, guests number, phone, email, name)

        Returns:
            Response from the webhook call indicating table availability.
        """
    return invoke_n8n_webhook(
        "POST",
        "https://batnano-n8n-a0b56247e135.herokuapp.com/webhook/0119b32c-b257-4bcb-a78c-a4016640b844",
        "reserve_table",
        payload=payload
    )


tools = [check_table_availability,reserve_table,get_youtube_transcript]
llm = llmOllama
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
prompt = f"""
current datetime {formatted_datetime}
You are a helpful agent that makes reservation for the restaurant 'l'imprévu'.
You first check for table availability with the check_table_availability function then you reserve a table with reserve_table function.
You always ask for confirmation before reserving the table
If there is no table available you propose the alternatives but you do not reserve without customer consent
You also summarize youtube videos if a youtube video url is submitted
"""
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
