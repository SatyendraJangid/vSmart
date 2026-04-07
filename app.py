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

from rag_pipeline import process_pdfs, split_docs, create_vectorstore, load_vectorstore, get_retriever, create_qa_chain, generate_suggestions
from chat_manager import create_chat, save_chat, load_chats, load_chat
from utils import generate_chat_title
from langchain_core.callbacks.base import BaseCallbackHandler

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="wide")

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

# Sidebar: Chat History and File Upload
with st.sidebar:
    st.title("📄 PDF Assistant")
    
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if st.button("Process PDFs"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = process_pdfs(uploaded_files, st.session_state.chat_id)
                chunks = split_docs(docs)
                vectorstore = create_vectorstore(chunks, st.session_state.chat_id)
                st.success("Indexing complete!")

        else:
            st.error("Please upload at least one PDF first.")

    st.divider()
    
    st.header("2. Chat History")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_id = create_chat()
        st.session_state.messages = []
        st.session_state.title = "New Chat"
        st.rerun()
        
    st.subheader("Previous Chats")
    chats = load_chats()
    for cat in chats:
        if st.sidebar.button(f"{cat['title']}", key=cat['chat_id'], use_container_width=True):
            st.session_state.chat_id = cat['chat_id']
            chat_data = load_chat(cat['chat_id'])
            st.session_state.messages = chat_data.get('messages', [])
            st.session_state.title = chat_data.get('title', 'New Chat')
            st.rerun()

# Main Chat Area
st.title(st.session_state.title)

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Quick Suggestion Buttons
if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
    last_msg = st.session_state.messages[-1]
    if "suggestions" in last_msg and last_msg["suggestions"]:
        st.write("💭 **Suggestions:**")
        cols = st.columns(len(last_msg["suggestions"]))
        for idx, sug in enumerate(last_msg["suggestions"]):
            # Use smaller custom buttons and append question text as next prompt on click
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
        save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
        st.rerun() # Refresh title sidebar

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        stream_handler = StreamHandler(message_placeholder)
        
        with st.status("🔍 Processing your request...", expanded=True) as status:
            debug_log("Step: Loading Vectorstore")
            status.write("Loading knowledge base...")
            vs = load_vectorstore(st.session_state.chat_id)
            
            if not vs:
                debug_log("Error: No vectorstore found")
                msg = "No documents have been processed for this chat yet. Please upload and process PDFs in the sidebar."
                message_placeholder.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
                save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                status.update(label="❌ No documents found", state="error", expanded=False)
            else:
                try:
                    debug_log("Step: Creating QA Chain")
                    status.write("Initializing AI assistant...")
                    retriever = get_retriever(vs)
                    qa_chain = create_qa_chain(retriever, stream_handler)
                    
                    status.write("Formatting chat history...")
                    chat_history = [(m["content"], st.session_state.messages[i+1]["content"]) 
                                    for i, m in enumerate(st.session_state.messages[:-1]) 
                                    if m["role"] == "user" and i+1 < len(st.session_state.messages) and st.session_state.messages[i+1]["role"] == "assistant"]

                    debug_log(f"Step: Invoking QA chain for question: {question}")
                    status.write("Searching documents and generating answer...")
                    result = qa_chain.invoke({
                        "question": question,
                        "chat_history": chat_history
                    })
                    
                    debug_log("Step: QA chain invocation complete")
                    status.write("Finalizing response...")
                    
                    answer = result['answer']
                    sources = result.get('source_documents', [])
                    generated_q = result.get('generated_question', 'N/A')
                    debug_log(f"Generated standalone question: {generated_q}")
                    debug_log(f"Retrieved {len(sources)} source documents")




                    
                    if sources:
                        answer += "\n\n**Sources:**\n"
                        source_names = list(set([doc.metadata.get('source', 'Unknown') for doc in sources]))
                        for s in source_names:
                            answer += f"- `{s}`\n"
                    
                    message_placeholder.markdown(answer)
                    
                    # Generate suggestions
                    try:
                        suggestions_text = generate_suggestions(qa_chain.llm, answer).content
                        import re
                        suggestions = [s.strip('- ').strip() for s in suggestions_text.split('\n') if s.strip() and not s.lower().startswith('here')]
                        suggestions = [s for s in suggestions if s][:3]
                    except Exception as e:
                        suggestions = []

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "suggestions": suggestions
                    })
                    save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                    status.update(label="✅ Answer generated", state="complete", expanded=False)
                    st.rerun()
                except Exception as e:
                    debug_log(f"Error during QA: {str(e)}")
                    st.error(f"Error during QA: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
                    save_chat(st.session_state.chat_id, st.session_state.title, st.session_state.messages)
                    status.update(label="❌ Error occurred", state="error", expanded=False)

