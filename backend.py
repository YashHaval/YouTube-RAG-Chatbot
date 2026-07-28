import os
import tempfile
from urllib.parse import urlparse, parse_qs

from dotenv import load_dotenv
from groq import Groq
from yt_dlp import YoutubeDL

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import RequestBlocked

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ytt_api = YouTubeTranscriptApi()


def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in (
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com",
    ):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]

    raise ValueError("Invalid YouTube URL")


def get_video_title(url):
    ydl_opts = {
        "quiet": True,
        "noplaylist": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return info.get("title", "Unknown Title")


def download_audio(url):
    temp_dir = tempfile.mkdtemp()

    output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        audio_path = os.path.splitext(
            ydl.prepare_filename(info)
        )[0] + ".mp3"

    return audio_path


def whisper_transcribe(audio_path):

    with open(audio_path, "rb") as audio_file:

        transcription = groq_client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3",
            response_format="text",
        )

    return transcription


def process_video(url):

    video_id = extract_video_id(url)

    try:
        print("Trying YouTube Transcript API...")

        transcripts = ytt_api.list(video_id)

        first_transcript = next(iter(transcripts))

        fetched = first_transcript.fetch()

        transcript = " ".join(
            snippet.text
            for snippet in fetched.snippets
        )

        print("Transcript API Success!")

    except RequestBlocked:

        print("Transcript API Blocked!")
        print("Switching to Groq Whisper...")

        audio_path = download_audio(url)

        transcript = whisper_transcribe(audio_path)

        print("Whisper Transcription Completed!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.create_documents([transcript])

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    title = get_video_title(url)

    return retriever, title