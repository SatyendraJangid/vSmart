# PDF RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) application that lets you chat with your PDF documents using Google Gemini and FAISS vector search.

## Features

- **PDF Ingestion** — Upload and process multiple PDFs at once.
- **In-Memory FAISS Vector Store** — Fast similarity search with HuggingFace `all-MiniLM-L6-v2` embeddings; no local disk persistence.
- **Google Gemini LLM** — Powered by `gemini-flash-latest` for fast, high-quality answers.
- **Streaming Responses** — Real-time, token-by-token answer generation.
- **Chat History** — Persistent chat sessions saved as JSON, with auto-generated titles.
- **Follow-up Suggestions** — AI-generated follow-up questions after each answer.
- **Response Time Metrics** — Logs Total, DB/Retrieval, and LLM processing times to the terminal.

## Tech Stack

| Component       | Technology                              |
| --------------- | --------------------------------------- |
| Frontend        | Streamlit                               |
| LLM             | Google Gemini (`gemini-flash-latest`)   |
| Embeddings      | HuggingFace (`all-MiniLM-L6-v2`)       |
| Vector Store    | FAISS (in-memory)                       |
| Orchestration   | LangChain (LCEL)                        |
| PDF Parsing     | PyPDF                                   |

## Project Structure

```
├── app.py              # Streamlit UI and chat logic
├── rag_pipeline.py     # RAG chain: PDF processing, embeddings, QA, metrics
├── chat_manager.py     # Chat session CRUD (JSON-based)
├── utils.py            # Utility helpers (title generation)
├── requirements.txt    # Pinned Python dependencies
├── .env                # API keys (not committed)
└── chats/              # Saved chat sessions (not committed)
```

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd TrainingProject
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

## Usage

1. **Upload PDFs** — Use the sidebar file uploader.
2. **Process** — Click "Process PDFs" to index the documents into the in-memory FAISS store.
3. **Ask Questions** — Type your question in the chat input.
4. **View Sources** — Each answer includes the source PDF file names.
5. **Check Metrics** — Response time metrics (Total, DB/Retrieval, LLM) are logged to the terminal running Streamlit.

## Response Time Metrics

The application tracks and logs the following metrics for every query:

| Metric          | Description                                    |
| --------------- | ---------------------------------------------- |
| **Total**       | End-to-end response time                       |
| **DB/Retrieval**| Time spent searching the FAISS vector store    |
| **LLM**        | Time spent on Gemini LLM generation             |

Example log output:

```
>>> DEBUG_LOG: Response Metrics => Total: 2.35s | DB/Retrieval: 0.12s | LLM: 2.20s
```

## License

This project is for training and educational purposes.
