import streamlit as st
import os
import logging
import sys
import warnings

# Aggressive warning suppression
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

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

# Guaranteed visibility debug log
def debug_log(msg):
    logger.info(msg)
    print(f"\n>>> DEBUG_LOG: {msg}", file=sys.stderr, flush=True)

from rag_pipeline import process_pdfs, split_docs, create_vectorstore, create_qa_chain, generate_suggestions
from chat_manager import create_chat, save_chat, load_chats, load_chat
from utils import generate_chat_title
from langchain_core.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="PDF RAG Assistant", layout="wide")

# Custom UI Styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}
.reportview-container .main .block-container{
    padding-top: 2rem;
}
.chat-row {
    margin-bottom: 2rem;
}
/* ChatGPT style sidebar buttons */
.stButton > button {
    border-radius: 5px;
    height: 3em;
    width: 100%;
    border: 1px solid #4d4d4f;
    background-color: transparent;
    color: white;
    text-align: left;
    padding-left: 10px;
    margin-bottom: 5px;
}
.stButton > button:hover {
    background-color: #2a2b32;
    border: 1px solid #4d4d4f;
}
/* Sticky Sidebar Top Container */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:nth-child(1) {
    position: sticky;
    top: 0;
    background-color: transparent;
    z-index: 1000;
    padding-bottom: 5px;
    border-bottom: 1px solid #4d4d4f;
    margin-bottom: 10px;
}
/* Ensure sidebar headers and text are visible */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
    color: white !important;
}
/* Restore gap in sidebar */
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
/* Hide default streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Initialize Session State
if "chat_id" not in st.session_state:
    st.session_state.chat_id = create_chat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "title" not in st.session_state:
    st.session_state.title = "New Chat"
if "vectorstores" not in st.session_state:
    st.session_state.vectorstores = {} # Dictionary mapping chat_id to in-memory vectorstore

# Sidebar: Chat History and File Upload
with st.sidebar:
    # Use a container for the sticky part
    # We will target the first child of the sidebar's vertical block with CSS
    with st.container():
        st.markdown('<h2 style="margin-top: 0; margin-bottom: 10px;">PDF Assistant</h2>', unsafe_allow_html=True)
        
        if st.button("New Chat", use_container_width=True):
            st.session_state.chat_id = create_chat()
            st.session_state.messages = []
            st.session_state.title = "New Chat"
            st.rerun()

        st.markdown('<p style="margin-top: 15px; margin-bottom: 5px; font-weight: bold;">Upload Documents</p>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
        
        if st.button("Process PDFs", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    docs = process_pdfs(uploaded_files, st.session_state.chat_id)
                    chunks = split_docs(docs)
                    vectorstore = create_vectorstore(chunks)
                    st.session_state.vectorstores[st.session_state.chat_id] = vectorstore
                    # Use session state for messages to prevent pushing sticky container
                    st.session_state.processing_success = "Indexing complete!"
            else:
                st.session_state.processing_error = "Please upload PDFs first."
    
    # Transient messages outside sticky container
    if "processing_success" in st.session_state:
        st.success(st.session_state.processing_success)
        del st.session_state.processing_success
    if "processing_error" in st.session_state:
        st.error(st.session_state.processing_error)
        del st.session_state.processing_error

    # Chat History (Now scrollable naturally as the rest of the sidebar)
    st.markdown('<p style="margin-top: 10px; margin-bottom: 5px; color: #8e8ea0; font-size: 0.8rem; text-transform: uppercase;">Chat History</p>', unsafe_allow_html=True)
    chats = load_chats()
    for cat in chats:
        btn_label = f"{cat['title']}"
        # Indicate if the chat has a vectorstore loaded in memory
        if cat['chat_id'] in st.session_state.vectorstores:
            btn_label += " (Active)"
        
        if st.sidebar.button(btn_label, key=cat['chat_id'], use_container_width=True):
            st.session_state.chat_id = cat['chat_id']
            chat_data = load_chat(cat['chat_id'])
            st.session_state.messages = chat_data.get('messages', [])
            st.session_state.title = chat_data.get('title', 'New Chat')
            st.rerun()

# Main Chat Area
chat_title_placeholder = st.title(st.session_state.title)

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick Suggestion Buttons
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    if "suggestions" in last_msg and last_msg["suggestions"]:
        st.write("**Suggestions:**")
        cols = st.columns(min(3, len(last_msg["suggestions"])))
        for idx, sug in enumerate(last_msg["suggestions"]):
            if cols[idx].button(sug, key=f"sug_{idx}_{len(st.session_state.messages)}"):
                st.session_state.pending_question = sug
                st.rerun()

question = st.chat_input("Ask a question about your documents...")

if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        
    # Set chat title if first message
    if len(st.session_state.messages) == 1:
        st.session_state.title = generate_chat_title(question)
        chat_title_placeholder.title(st.session_state.title)
        save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        
        with st.status("Processing...", expanded=True) as status:
            debug_log("Step: Accessing In-Memory Vectorstore")
            vs = st.session_state.vectorstores.get(st.session_state.chat_id)
            
            if not vs:
                debug_log("Error: No vectorstore in memory")
                msg = ("No documents have been processed for this chat in the current session. "
                       "Since embeddings are not stored locally, please re-upload and process your PDFs in the sidebar.")
                message_placeholder.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                status.update(label="No documents in memory", state="error", expanded=False)
            else:
                try:
                    debug_log("Step: Creating QA Chain")
                    status.write("Initializing modern AI assistant...")
                    rag_chain = create_qa_chain(vs, stream_handler)
                    
                    debug_log(f"Step: Invoking RAG chain for question: {question}")
                    status.write("Searching and generating answer...")
                    
                    from rag_pipeline import run_qa
                    result = run_qa(rag_chain, question)
                    
                    debug_log("Step: RAG chain invocation complete")
                    status.write("Finalizing response...")
                    
                    answer = result['answer']
                    context_docs = result.get('source_documents', [])
                    
                    if context_docs and "**Sources:**" not in answer:
                        answer += "\n\n**Sources:**\n"
                        source_names = list(set([doc.metadata.get('source', 'Unknown') for doc in context_docs]))
                        for s in source_names:
                            answer += f"- `{s}`\n"
                    
                    # Log metrics to terminal
                    metrics = result.get('metrics', {})
                    if metrics:
                        debug_log(
                            f"Response Metrics => "
                            f"Total: {metrics['total_time']:.2f}s | "
                            f"DB/Retrieval: {metrics['db_time']:.2f}s | "
                            f"LLM: {metrics['llm_time']:.2f}s"
                        )

                    message_placeholder.markdown(answer)
                    
                    # Generate suggestions
                    try:
                        # Handle different chain types for suggestions
                        if hasattr(rag_chain, 'combine_docs_chain'):
                            base_llm = rag_chain.combine_docs_chain.llm
                        else:
                            base_llm = rag_chain.llm
                        
                        suggestions_text = generate_suggestions(base_llm, answer).content
                        suggestions = [s.strip('- ').strip() for s in suggestions_text.split('\n') if s.strip() and not s.lower().startswith('here')]
                        suggestions = [s for s in suggestions if s][:3]
                    except Exception as e:
                        debug_log(f"Suggestions error: {e}")
                        suggestions = []

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "suggestions": suggestions
                    })
                    save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                    status.update(label="Answer generated", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    debug_log(f"Error during QA: {str(e)}")
                    st.error(f"Error during QA: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                    save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                    status.update(label="Error occurred", state="error", expanded=False)