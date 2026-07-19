from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from yt_dlp import YoutubeDL
from dotenv import load_dotenv

load_dotenv()
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


def process_video(url):

    video_id = extract_video_id(url)

    transcripts = ytt_api.list(video_id)

    first_transcript = next(iter(transcripts))

    fetched = first_transcript.fetch()

    transcript = " ".join(
        snippet.text
        for snippet in fetched.snippets
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding
    )
    

    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k':4}
    )

    title = get_video_title(url)

    return retriever, title

def get_video_title(url):
    ydl_opts = {
    "quiet": True,
    "noplaylist": True
}

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    print(info)   # Add this line

    return info["title"]