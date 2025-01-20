import streamlit as st
from agent import graph_builder
from typing import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="Chat with AI Agent",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="""You are a helpful agent that makes reservation for the restaurant 'l'imprévu'.
You first check for table availability with the check_table_availability function then you reserve a table with reserve_table function.
You always ask for confirmation before reserving the table.
If there is no table available you propose the alternatives but you do not reserve without customer consent.
You also summarize youtube videos if a youtube video url is submitted.""")
    ]

# Create the header
st.title("🤖 Restaurant L'Imprevu Assistant")
st.markdown("---")

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

# Get user input
if prompt := st.chat_input("How can I help you with your reservation?"):
    # Create a HumanMessage and add to chat history
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Create initial state with full chat history
            state = State(messages=st.session_state.messages)
            
            # Create and run the graph
            graph = graph_builder.compile()
            result = graph.invoke(state)
            
            # Get all new messages from the result
            new_messages = result["messages"][len(st.session_state.messages):]
            
            # Display and store new messages
            for message in new_messages:
                if isinstance(message, AIMessage):
                    st.markdown(message.content)
                    st.session_state.messages.append(message)

# Add a clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = [
        SystemMessage(content="""You are a helpful agent that makes reservation for the restaurant 'l'imprévu'.
You first check for table availability with the check_table_availability function then you reserve a table with reserve_table function.
You always ask for confirmation before reserving the table.
If there is no table available you propose the alternatives but you do not reserve without customer consent.
You also summarize youtube videos if a youtube video url is submitted.""")
    ]
    st.rerun()