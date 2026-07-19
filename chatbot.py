from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed_url = urlparse(url)

    # https://youtu.be/VIDEO_ID
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    # https://www.youtube.com/watch?v=VIDEO_ID
    if parsed_url.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]

    raise ValueError("Invalid YouTube URL")


ytt_api = YouTubeTranscriptApi()

while True:
    url = input("Enter YouTube URL: ").strip()

    try:
        video_id = extract_video_id(url)
        print(f"\nVideo ID: {video_id}")

        transcripts = ytt_api.list(video_id)

        first_transcript = next(iter(transcripts))

        fetched = first_transcript.fetch()

        transcript = " ".join(
            snippet.text
            for snippet in fetched.snippets
        )

        # print(type(fetched))
        # print(fetched)

        print("\nTranscript loaded successfully!")
        print(f"Length: {len(transcript)} characters\n")
        break

    except ValueError:
        print("❌ Invalid YouTube URL. Try again.\n")

    except TranscriptsDisabled:
        print("❌ Transcript is disabled for this video. Try another video.\n")

    except Exception as e:
        print(f"❌ Error: {e}\n")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks = splitter.create_documents([transcript])


embedding = HuggingFaceEmbeddings(
    model = "sentence-transformers/all-MiniLM-L6-v2"
)


vector_store = FAISS.from_documents(chunks,embedding)

vector_store.index_to_docstore_id

retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={'k':4})

from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful AI assistant.

- Answer greetings naturally.
- Answer questions using only the provided transcript context.
- If the transcript doesn't contain the answer, say "I don't know."

Context:
{context}
"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])




load_dotenv(r"E:\Langchain\.env")

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

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


from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

session_id = "user1"

print()
print("=" * 100)
print("YouTube RAG Chatbot")
print("Type 'exit' to quit")
print("=" * 100)

while True:
    question = input("\nYou: ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    response = chatbot.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    print("\nAI:", response)