import streamlit as st
from urllib.parse import urlparse

from backend import process_video
from chat_engine import create_chatbot, clear_chat_history

def is_valid_youtube_url(url):
    try:
        parsed = urlparse(url)

        valid_domains = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtu.be",
            "www.youtu.be",
        }

        if parsed.netloc not in valid_domains:
            return False

        # Handle https://youtu.be/VIDEO_ID
        if parsed.netloc in {"youtu.be", "www.youtu.be"}:
            return parsed.path.strip("/") != ""

        # Handle https://www.youtube.com/watch?v=...
        if parsed.path == "/watch":
            return "v" in parsed.query

        # Handle /shorts/VIDEO_ID and /embed/VIDEO_ID
        if parsed.path.startswith("/shorts/") or parsed.path.startswith("/embed/"):
            return True

        return False

    except Exception:
        return False

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide"
)

# ==========================
# Session State
# ==========================

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_history" not in st.session_state:
    st.session_state.video_history = {}

if "current_video" not in st.session_state:
    st.session_state.current_video = None

if "video_url" not in st.session_state:
    st.session_state.video_url = ""

# ==========================
# UI
# ==========================

st.title("🎥 YouTube RAG Chatbot")

if not st.session_state.video_processed:
    st.markdown("""
### 🤖 Chat with Any YouTube Video

Paste any YouTube URL to:

✅ Generate AI Notes
✅ Ask Questions
✅ Summarize Videos
✅ Remember Chat History

---
""")

# Sidebar
with st.sidebar:
    st.title("💬 Conversations")
    search = st.text_input(
    "🔍 Search Chats",
    placeholder="Search...",
    key="search_chat"
)

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chatbot = None
        st.session_state.video_processed = False
        st.session_state.messages = []
        st.session_state.current_video = None

    

        # Clear the URL input
        st.session_state.video_url = ""
        st.rerun()

    st.divider()

    # Search filter
    videos = list(st.session_state.video_history.keys())

    search_text = search.strip().lower()

    if search_text:
        videos = [
            video
            for video in videos
            if search_text in video.lower()
        ]

    # st.divider()

    for video in videos:

        col1, col2 = st.columns([0.82, 0.18], gap="small")
        # Open chat
        with col1:
            if st.button(video, key=f"open_{video}", use_container_width=True):

                st.session_state.current_video = video
                st.session_state.chatbot = st.session_state.video_history[video]["chatbot"]
                st.session_state.messages = st.session_state.video_history[video]["messages"]
                st.session_state.video_processed = True

                st.rerun()

        # Delete chat
        with col2:
            if st.button("🗑️", key=f"delete_{video}", use_container_width=True):

                del st.session_state.video_history[video]

                if st.session_state.current_video == video:
                    st.session_state.current_video = None
                    st.session_state.chatbot = None
                    st.session_state.messages = []
                    st.session_state.video_processed = False
                    st.session_state.video_url = ""

                st.rerun()


#Video URL

video_url = st.text_input(
    "Enter a YouTube URL",
    key="video_url"
)
#Process Video

if st.button("Process Video"):

    if not video_url:
        st.warning("Please enter a YouTube URL.")

    else:
        with st.spinner("Processing video..."):

            if not is_valid_youtube_url(video_url):
                st.error("❌ Please enter a valid YouTube URL.")
                st.stop()
            retriever, title = process_video(video_url)

            # Clear previous AI memory
            # clear_chat_history()

            chatbot = create_chatbot(retriever)

            st.session_state.chatbot = chatbot
            st.session_state.video_processed = True

            # Save this video
            st.session_state.video_history[title] = {
            "chatbot": chatbot,
            "messages": []
}

        st.session_state.current_video = title

            # Current chat starts empty
        st.session_state.messages = []

        

    st.success("Video processed successfully!")

# Ask Questions

if st.session_state.video_processed:

    # Show previous chats
    for chat in st.session_state.messages:

        with st.chat_message("user", avatar="🧑"):
            st.markdown("### You")
            st.write(chat["question"])

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("### YouTube AI")
            st.write(chat["answer"])

    # Chat input (always at the bottom)
    question = st.chat_input("Ask a question about the video")

    if question:

        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.invoke(
                {"question": question},
                config={
                    "configurable": {
                        "session_id": st.session_state.current_video
                    }
                }
            )

            chat = {
            "question": question,
            "answer": response
        }

        st.session_state.messages.append(chat)

        st.session_state.video_history[
            st.session_state.current_video
        ]["messages"].append(chat)

        with st.chat_message("assistant"):
            st.write(response)



