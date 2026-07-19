from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from operator import itemgetter

from dotenv import load_dotenv
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

- Your response language must always be English.
- Do not reply in Marathi, Hindi, or any other language unless the user explicitly requests it.
- You may use transcripts written in any language, but translate the information into English before answering.

Context:
{context}
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parser = StrOutputParser()

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def clear_chat_history():
    store.clear()

def create_chatbot(retriever):

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever
            | format_docs
        )
        | prompt
        | llm
        | parser
    )

    chatbot = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    return chatbot
