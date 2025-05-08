
# streamlit_app.py
import os
import streamlit as st
from dotenv import load_dotenv
from embeddings import build_faiss_index_from_txt
from llm import create_agent_with_memory

# Load environment
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="ITC PORTAL CHATBOT",
    page_icon=":robot_face:",
    layout="centered"
)
st.title("ITC PORTAL CHATBOT 🤖")

# Sidebar for indexing txt file
st.sidebar.header("Load & Index Text Content")
## upload file
uploaded_file = st.sidebar.file_uploader(
    "Upload a .txt file:",
    type=["txt"],
    help="Upload a text file to index"
)
build_index_button = st.sidebar.button("Build Retrieval Agent")

if uploaded_file and build_index_button:
    st.sidebar.info("Indexing text and initializing agent...")
    with st.spinner("Building FAISS index and agent..."):
        # Build index
        vectorstore = build_faiss_index_from_txt(uploaded_file)
        # Create agent
        st.session_state.agent_executor = create_agent_with_memory(vectorstore)
    st.sidebar.success("Agent ready! Ask your questions below.")

# Initialize chat history
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome! How can I help?"}
    ]

chat_placeholder = st.empty()

def display_chat():
    with chat_placeholder.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                _, col = st.columns([1, 5])
                with col:
                    st.chat_message("user").write(msg["content"])
            else:
                col, _ = st.columns([5, 1])
                with col:
                    st.chat_message("assistant").write(msg["content"])

# Show chat
display_chat()

# User input
user_input = st.chat_input("Ask me anything about the loaded content...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    display_chat()
    if st.session_state.agent_executor:
        with st.spinner("Thinking..."):
            result = st.session_state.agent_executor.invoke({"input": user_input})
            response = result.get("output")
    else:
        response = "Please build the agent first by indexing the text file."
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    display_chat()



