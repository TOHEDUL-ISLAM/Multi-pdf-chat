import tempfile
import time
import json
import requests
import concurrent.futures
from typing import List, Dict, Any
import pickle
import logging
import sys
from pathlib import Path

import streamlit as st
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# --- Error Handling Decorator ---
def handle_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# --- Performance Optimization ---
@handle_exceptions
def configure_torch():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'classes'):
        try:
            torch.classes.__path__ = []
        except Exception as e:
            logger.warning(f"Failed to configure torch classes: {e}")

# --- Constants & Config ---
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 2000
MAX_WORKERS = 1
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# --- Initialize embeddings with caching and error handling ---
@st.cache_resource(show_spinner=False)
@handle_exceptions
def get_embeddings():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Initializing embeddings on device: {device}")
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v2-moe",
        model_kwargs={
            'device': device,
            'trust_remote_code': True
        }
    )

# --- Session State Management ---
@handle_exceptions
def initialize_session_state():
    if 'vector_store' not in st.session_state:
        embeddings = get_embeddings()
        if embeddings:
            st.session_state.vector_store = FAISS.from_texts(
                ["initialization"], embeddings
            )
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'processed_pdfs' not in st.session_state:
        st.session_state.processed_pdfs = set()
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0

# --- Enhanced PDF Processor ---
class EnhancedPDFProcessor:
    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            add_start_index=True
        )
    
    @handle_exceptions
    def process_pdf(self, file) -> bool:
        logger.info(f"Processing PDF: {file.name}")
        
        # Validate file size
        if file.size > 200 * 1024 * 1024:  # 200MB
            raise ValueError("File size exceeds 200MB limit")
            
        for attempt in range(MAX_RETRIES):
            try:
                with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp_file:
                    tmp_file.write(file.getbuffer())
                    loader = PDFPlumberLoader(tmp_file.name)
                    documents = loader.load()
                
                chunks = self.text_splitter.split_documents(documents)
                
                if not chunks:
                    raise ValueError("No text content extracted from PDF")
                
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        'source': file.name,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chars_count': len(chunk.page_content),
                        'words_count': len(chunk.page_content.split())
                    })
                
                st.session_state.vector_store.add_documents(chunks)
                st.session_state.processed_pdfs.add(file.name)
                
                logger.info(f"Successfully processed {file.name}")
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {file.name}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    raise

# --- Main Streamlit App ---
@handle_exceptions
def main() -> None:
    configure_torch()
    
    st.set_page_config(
        page_title="Multi-PDF Chat",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.header("PDF Manager")
        st.info("1. Upload PDFs below\n2. Select which to use\n3. Ask questions in the chat!")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs (Max 200MB each)",
            type="pdf",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                processor = EnhancedPDFProcessor()
                new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_pdfs]
                
                if new_files:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(processor.process_pdf, f) for f in new_files]
                        results = [f.result() for f in futures]
                    
                    if all(results):
                        st.success("PDFs processed successfully!")
                        st.session_state.error_count = 0
                    else:
                        st.error("Some PDFs failed to process. Please try again.")
                        st.session_state.error_count += 1
        
        selected_pdfs = st.multiselect(
            "Select PDFs for context:",
            options=list(st.session_state.processed_pdfs),
            default=list(st.session_state.processed_pdfs)
        )
        
        if st.button("Clear Chat History"):
            st.session_state.history = []
            logger.info("Chat history cleared")
        
        if st.button("Reset All PDFs"):
            st.session_state.vector_store = FAISS.from_texts(
                ["initialization"], get_embeddings()
            )
            st.session_state.processed_pdfs = set()
            st.session_state.error_count = 0
            logger.info("All PDFs reset")
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}", exc_info=True)
        st.error("The application encountered a critical error. Please check the logs or contact support.")


