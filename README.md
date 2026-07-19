# 🎥 YouTube RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) application that lets users chat with any YouTube video. Simply paste a YouTube URL to generate AI-powered notes, ask questions, summarize videos, and interact with the content through natural language.

## 🚀 Features

- 🎥 Process any YouTube video using its URL
- 🤖 AI-powered question answering
- 📝 Automatic video summarization
- 📚 Generate structured notes
- 💬 Multi-chat conversation history
- 🔍 Search previous conversations
- 🗑️ Delete chat history
- ⚡ Fast semantic search using FAISS
- 🎯 Context-aware answers using Retrieval-Augmented Generation (RAG)

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### LLM
- Groq API
- Llama 3.3 70B Versatile

### Embeddings
- HuggingFace
- sentence-transformers/all-MiniLM-L6-v2

### Vector Database
- FAISS

### Framework
- LangChain

### Data Source
- YouTube Transcript API
- yt-dlp

---

## 📂 Project Structure

```
YouTube-RAG-Chatbot/
│
├── app.py                 # Streamlit UI
├── backend.py             # Video processing & FAISS creation
├── chat_engine.py         # RAG chain
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YouTube-RAG-Chatbot.git

cd YouTube-RAG-Chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. User enters a YouTube URL.
2. The transcript is extracted.
3. The transcript is split into chunks.
4. HuggingFace embeddings are generated.
5. Chunks are stored in a FAISS vector database.
6. Relevant context is retrieved.
7. Groq Llama 3.3 generates context-aware responses.

---

## 🔮 Future Improvements

- Persistent chat history
- PDF chat export
- Voice input
- Text-to-Speech
- Google Authentication
- Cloud database integration
- Multi-user support

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Yash Haval**

- LinkedIn: https://www.linkedin.com/in/yashhaval2304
- GitHub: https://github.com/YashHaval