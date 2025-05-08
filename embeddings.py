# embeddings.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables for OpenAI API key
load_dotenv()

# Constants
INDEX_DIR = "faiss_index"

def build_faiss_index_from_txt(
    txt_path: str,
) -> FAISS:
    """
    Reads text from a given .txt file, splits it into chunks using RecursiveCharacterTextSplitter,
    and builds a FAISS index with OpenAI embeddings.

    Args:
        txt_path (str): Path to the input text file.
    Returns:
        FAISS: A FAISS vectorstore instance.
    """
    # Read raw text
    if hasattr(txt_path, "read"):
        # Streamlit UploadedFile or file-like
        raw_bytes = txt_path.read()
        # If bytes, decode
        full_text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, (bytes, bytearray)) else raw_bytes
    else:
        # File path
        with open(txt_path, "r", encoding="utf-8") as f:
            full_text = f.read()

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )

    # Split text into chunks
    chunks = text_splitter.split_text(full_text)

    # Create embeddings and FAISS vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # Save index locally
    vectorstore.save_local(INDEX_DIR)
    return vectorstore


