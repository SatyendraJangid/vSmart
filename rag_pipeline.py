import streamlit as st
import os
import logging
import sys
import warnings
from dotenv import load_dotenv
import time

# Aggressive warning suppression
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
    force=True
)
logger = logging.getLogger(__name__)

def debug_log(msg):
    logger.info(msg)
    print(f"\n>>> DEBUG_LOG [pipeline]: {msg}", file=sys.stderr, flush=True)

load_dotenv()

# Basic Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# LCEL Imports
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# =========================
# EMBEDDINGS (HuggingFace - Cached)
# =========================
@st.cache_resource
def get_embeddings():
    """Returns a cached HuggingFace embeddings model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

# =========================
# PDF PROCESSING
# =========================
def process_pdfs(uploaded_files, chat_id):
    debug_log(f"Processing {len(uploaded_files)} PDF(s)")
    documents = []
    temp_dir = f"temp_{chat_id}"
    os.makedirs(temp_dir, exist_ok=True)

    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = file.name
        documents.extend(docs)
        try: os.remove(path)
        except: pass
            
    try: os.rmdir(temp_dir)
    except: pass
    return documents

# =========================
# TEXT SPLITTING
# =========================
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# =========================
# VECTOR STORE (FAISS)
# =========================
def create_vectorstore(docs):
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embedding=embeddings)

# =========================
# LLM (Fast Flash model - Cached)
# =========================
@st.cache_resource
def _get_base_llm():
    """Returns a cached base LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        verbose=True
    )

def get_llm(streaming=False, callbacks=None):
    """Returns an LLM instance, optionally with streaming."""
    if not streaming:
        return _get_base_llm()
    
    # For streaming, we need a fresh object to attach callbacks
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        streaming=True,
        callbacks=callbacks,
        verbose=True
    )

# =========================
# QA CHAIN
# =========================
def create_qa_chain(vectorstore, stream_handler=None):
    llm = get_llm(streaming=bool(stream_handler), callbacks=[stream_handler] if stream_handler else None)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an assistant for question-answering tasks. "
            "Use the provided context to answer the question. "
            "If the answer is not contained within the context or if you are unsure, "
            "simply state that you do not have enough details to answer based on the provided documents. "
            "Do not use any outside knowledge or provide information not found in the context."
        )),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_retrieval_data(input_text):
        start = time.time()
        docs = retriever.invoke(input_text)
        end = time.time()
        return {
            "context": format_docs(docs),
            "source_documents": docs,
            "db_time": end - start
        }

    def get_llm_answer(x):
        start = time.time()
        # We need to pass the context and input to the prompt
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": x["retrieval"]["context"],
            "input": x["input"]
        })
        end = time.time()
        return {
            "answer": answer,
            "llm_time": end - start
        }

    # LCEL RAG Chain with embedded timing
    chain = (
        RunnableParallel({
            "retrieval": get_retrieval_data,
            "input": RunnablePassthrough()
        })
        | RunnableParallel({
            "generation": get_llm_answer,
            "retrieval": lambda x: x["retrieval"]
        })
    )
    return chain

# =========================
# QA EXECUTION (Unified Interface)
# =========================
def run_qa(chain, query):
    start_total = time.time()
    result = chain.invoke(query)
    end_total = time.time()
    
    return {
        "answer": result['generation']['answer'],
        "source_documents": result['retrieval']['source_documents'],
        "metrics": {
            "total_time": end_total - start_total,
            "db_time": result['retrieval']['db_time'],
            "llm_time": result['generation']['llm_time']
        }
    }

# =========================
# SUGGESTIONS
# =========================
def generate_suggestions(llm, answer):
    prompt = f"Based on the answer: {answer}, generate 3 short follow-up questions as bullet points."
    return llm.invoke(prompt)