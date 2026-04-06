import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.prompts import PromptTemplate

# =========================
# EMBEDDINGS (FAST + CACHED)
# =========================
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

# =========================
# PDF PROCESSING
# =========================
def process_pdfs(uploaded_files, chat_id):
    documents = []
    os.makedirs(f"temp_{chat_id}", exist_ok=True)

    for file in uploaded_files:
        path = os.path.join(f"temp_{chat_id}", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = file.name

        documents.extend(docs)

    return documents

# =========================
# TEXT SPLITTING
# =========================
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )
    return splitter.split_documents(documents)

# =========================
# VECTOR STORE (CACHED)
# =========================
@st.cache_resource
def create_vectorstore(docs, chat_id):
    embeddings = get_embeddings()
    persist_dir = f"db_{chat_id}"
    return Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)

def load_vectorstore(chat_id):
    persist_dir = f"db_{chat_id}"
    if os.path.exists(persist_dir) and os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0:
        embeddings = get_embeddings()
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return None

# =========================
# RETRIEVER (MMR OPTIMIZED)
# =========================
def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

# =========================
# LLM (FAST)
# =========================
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0
    )

# =========================
# PROMPT
# =========================
def get_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""
Answer using ONLY the given context.

- Give the source of the information in the format of bullet points with the file name. For example:
- [source_file.pdf]
- If the context does not contain the answer, say "The answer is not in the provided context." Do not try to make up an answer.
- If not found, say "I don't have the idea about what you are asking for. Sorry for inconvenience. Ask the relevant question. 
Thank you for your understanding."

Context:
{context}

Question:
{question}

Answer:
"""
    )

# =========================
# QA CHAIN
# =========================
def create_qa_chain(retriever, stream_handler=None):
    condense_llm = get_llm()
    
    if stream_handler:
        streaming_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            callbacks=[stream_handler]
        )
    else:
        streaming_llm = get_llm()

    return ConversationalRetrievalChain.from_llm(
        llm=streaming_llm,
        condense_question_llm=condense_llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": get_prompt()}
    )

# =========================
# SUGGESTIONS
# =========================
def generate_suggestions(llm, answer):
    prompt = f"""
Based on the following answer, generate 3 directly related follow-up questions the user might logically ask next.
Make them concise.

Answer Context:
{answer}

Return only the 3 questions as bullet points without any other introductory text.
"""
    return llm.invoke(prompt)