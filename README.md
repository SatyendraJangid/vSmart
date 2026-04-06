# PDF Chat Assistant with RAG (Retrieval-Augmented Generation)

A powerful Streamlit-based application that allows users to upload PDF documents and seamlessly chat with their content. This project leverages the latest LangChain framework and Google's Gemini models to provide intelligent, context-aware responses with extremely low latency.

## Features

- **Document Processing**: Upload one or multiple PDF files. Text extraction and intelligent chunking using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
- **RAG Architecture**: Highly optimized Retrieval-Augmented Generation that ensures the AI answers *only* using the provided context, reducing hallucinations.
- **Advanced Embeddings & Vector Store**: Uses `GoogleGenerativeAIEmbeddings` with a local lightweight `Chroma` database for fast, cached document search.
- **MMR Optimized Retrieval**: Implements Maximum Marginal Relevance (MMR) for retrieving diverse and highly relevant chunks.
- **Google Gemini Integration**: Powered by Google's `gemini-2.0-flash` for lightning-fast comprehension and generation.
- **Chat Management**: Supports multiple persistent chat sessions stored locally in JSON format, allowing users to revisit older conversations effortlessly.
- **Smart Follow-up Suggestions**: After every response, the system dynamically analyzes the context and provides three relevant follow-up questions for an interactive experience.
- **Source Tracing**: Answers include bulleted source citations so you know exactly which uploaded document provided the context.

## Prerequisites

- Python 3.8+
- An active API key from Google (Gemini/Google MakerSuite) for LLM and Embeddings access.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <project-directory>
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   Make sure you have requirements listed correctly, normally using `pip`:
   ```bash
   pip install streamlit langchain-community langchain-text-splitters langchain-google-genai langchain-classic chromadb pypdf
   ```

4. **Environment Variables Configuration:**
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Start the application:**
   Run the following command to start the Streamlit UI:
   ```bash
   streamlit run rag_pipeline.py
   ```
   *(Note: Adjust the target python file to your main Streamlit file if the entry point differs)*

2. **In the Web Interface:**
   - **Upload PDFs**: Drag and drop or browse to select your PDF documents.
   - **Process**: Wait a moment for the documents to be embedded and saved to the session.
   - **Chat**: Ask any question pertaining to the document content.
   - **Follow Up**: Click on any of the magically generated follow-up questions to explore topics deeper.

## File Structure

- `rag_pipeline.py` - Core logic for document loading, splitting, vector setup, retriever pipeline, and LLM orchestration.
- `chat_manager.py` - Simple and efficient utility handling UUID-based chat sessions and managing saved files in the `chats/` directory.
- `utils.py` - Contains text helpers including dynamic chat title generation based on initial queries.
- `temp_<chat_id>/` - Temporary folder per chat session containing the specific uploaded PDFs.
- `db_<chat_id>/` - Persistent Chromadb vector storage unique to that specific chat session.

## License

This project is licensed under the MIT License.
