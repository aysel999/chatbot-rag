import streamlit as st
from openai import OpenAIError

from chatbot_rag.chat.service import ChatService


st.title("Chatbot RAG")

service = ChatService()

message = st.chat_input("Ask something...")

if message:
    st.chat_message("user").write(message)

    try:
        response = service.ask(message)
    except OpenAIError:
        st.error(
            "I can't get a response from OpenAI right now. "
            "Please check your API billing and try again."
        )
    else:
        st.chat_message("assistant").write(response)