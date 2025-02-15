import os
import time
import json
import requests
import shutil
import concurrent.futures
from typing import List, Dict, Any

import streamlit as st
import torch
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_chroma import Chroma
from langchain.vectorstores import Chroma

#from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

#from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

# --- Performance Optimization: Disable torch classes file watcher if not needed ---
torch.classes.__path__ = []

# --- Constants & Config ---
PDFS_DIRECTORY = 'pdfs'
DB_DIRECTORY = 'db'
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 2000
MAX_WORKERS = 1  # Adjust based on your CPU cores

# Ensure directories exist
os.makedirs(PDFS_DIRECTORY, exist_ok=True)
os.makedirs(DB_DIRECTORY, exist_ok=True)

# OpenRouter Configuration
OPENROUTER_API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
SITE_URL = "http://localhost:8501"
SITE_NAME = "Multi-PDF Chat App"

# --- Initialize embeddings with caching ---
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v2-moe",
        model_kwargs={
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            # 'device' : 'cpu'
            'trust_remote_code': True
        }
    )
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="allenai/specter",
#         model_kwargs={'device': 'cpu'}
#     )

@st.cache_resource(show_spinner=False)
def get_vector_store() -> Chroma:
    embeddings = get_embeddings()
    store = Chroma(
        collection_name="multi_pdf_store",
        embedding_function=embeddings,
        persist_directory=DB_DIRECTORY
    )
    # Check for embedding dimension mismatch via a dummy query.
    try:
        store.similarity_search("dimension_test", k=1)
    except Exception as e:
        if "dimension" in str(e).lower():
            shutil.rmtree(DB_DIRECTORY, ignore_errors=True)
            os.makedirs(DB_DIRECTORY, exist_ok=True)
            store = Chroma(
                collection_name="multi_pdf_store",
                embedding_function=embeddings,
                persist_directory=DB_DIRECTORY
            )
        else:
            raise e
    return store

# --- PDF Processing ---
class EnhancedPDFProcessor:
    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            add_start_index=True
        )
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Any]:
        return PDFPlumberLoader(file_path).load()
    
    def process_pdf(self, file, vector_store: Chroma) -> bool:
        try:
            file_path = os.path.join(PDFS_DIRECTORY, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            documents = self.load_pdf(file_path)
            chunks = self.text_splitter.split_documents(documents)
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'source': file.name,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chars_count': len(chunk.page_content),
                    'words_count': len(chunk.page_content.split())
                })
            for attempt in range(3):
                try:
                    vector_store.add_documents(chunks)
                    vector_store.persist()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            return True
        except Exception as e:
            st.error(f"Error processing '{file.name}': {str(e)}")
            return False

def get_stored_pdfs(vector_store: Chroma) -> List[str]:
    try:
        result = vector_store._collection.get(include=['metadatas'])
        if result and 'metadatas' in result:
            return list({meta['source'] for meta in result['metadatas'] if meta and 'source' in meta})
    except Exception as e:
        st.error(f"Error retrieving stored PDFs: {str(e)}")
    return []

# --- Retrieval & Context Building (Hybrid Retrieval Only) ---
def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store: Chroma) -> List[Any]:
    """
    Retrieve context documents for a query using a hybrid retrieval approach.
    This method fuses dense (semantic) scores with BM25 lexical scores.
    """
    if not selected_pdfs:
        return []
    
    # Prepend the task instruction prefix to the query
    query_with_prefix = "search_query: " + query
    
    # Dynamically set k based on query length and number of selected PDFs.
    query_words = len(query.split())
    base_k = min(4 * len(selected_pdfs), 10)
    k = min(base_k + (query_words // 10), 15)
    filter_dict = {"source": {"$in": selected_pdfs}}
    
    # Dense retrieval using Chroma (returns (doc, score) pairs)
    dense_results = vector_store.similarity_search_with_relevance_scores(query_with_prefix, k=k, filter=filter_dict)
    
    # Build a BM25 index on the retrieved documents.
    docs = [doc for doc, _ in dense_results]
    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    # Compute BM25 scores for the query (without prefix) once.
    query_tokens = query.split()
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Fuse dense and BM25 scores.
    hybrid_results = []
    for idx, (doc, dense_score) in enumerate(dense_results):
        bm25_score = bm25_scores[idx]
        combined_score = 0.5 * dense_score + 0.5 * bm25_score  # Equal weight fusion
        hybrid_results.append((doc, combined_score))
    
    # Sort documents by the combined score in descending order.
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in hybrid_results]

def build_context(docs: List[Any]) -> str:
    if not docs:
        return ""
    groups: Dict[str, List[Any]] = {}
    for doc in docs:
        groups.setdefault(doc.metadata['source'], []).append(doc)
    context_parts = [
        f"From {source}:\n" + "\n".join(
            chunk.page_content for chunk in sorted(chunks, key=lambda c: c.metadata.get('chunk_index', 0))
        )
        for source, chunks in groups.items()
    ]
    return "\n\n".join(context_parts)

# --- LLM Interaction ---
def get_last_two_history() -> List[Dict[str, str]]:
    history = st.session_state.get("history", [])
    return history[-2:] if len(history) >= 2 else history

def generate_answer(question: str, context: str) -> str:
    messages = [{
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based on the provided PDF context and recent chat history. "
            "Provide accurate, well-structured responses and cite the specific PDF sources when possible."
        )
    }]
    for msg in get_last_two_history():
        messages.append({"role": msg["role"], "content": msg["text"]})
    prompt = (
        f"Context from relevant PDF sections:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Please provide a comprehensive answer based on the context above. "
        "If the context doesn't contain enough information to fully answer the question, please indicate that clearly."
    )
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "google/gemini-2.0-flash-thinking-exp:free",
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 66000
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload))
        data = response.json()
        if "choices" not in data:
            st.error(f"API response error: {data}")
            return f"Error generating answer: 'choices' missing. Full response: {data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def typewriter_effect(text: str, placeholder) -> None:
    output = ""
    for line in text.split('\n'):
        output += line + "\n"
        placeholder.markdown(output + "▌")
        time.sleep(0.01)
    placeholder.markdown(output)

# --- Main Streamlit App ---
def main() -> None:
    st.set_page_config(page_title="Multi-PDF Chat", layout="wide", page_icon="🤖")
    vector_store = get_vector_store()
    pdf_processor = EnhancedPDFProcessor()
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    with st.sidebar:
        st.header("PDF Manager")
        st.info("1. Upload PDFs below.\n2. They are processed automatically.\n3. Select which PDFs to use for context.\n4. Ask questions in the main chat area!")
        stored_pdfs = get_stored_pdfs(vector_store)
        uploaded_files = st.file_uploader("Upload PDFs (Max 200MB each)", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing PDFs..."):
                new_files = [f for f in uploaded_files if f.name not in stored_pdfs]
                if new_files:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        list(executor.map(lambda file: pdf_processor.process_pdf(file, vector_store), new_files))
            stored_pdfs = get_stored_pdfs(vector_store)
            st.success("PDFs processed successfully!")
        selected_pdfs = st.multiselect("Select PDFs for context:", options=stored_pdfs, default=stored_pdfs) if stored_pdfs else []
        if st.button("Clear Chat History"):
            st.session_state.history = []
        st.markdown("---")
        st.header("About")
        st.markdown("This **Multi-PDF Chat App** lets you upload multiple PDFs and ask questions. A Large Language Model references the content of these PDFs to provide context-aware answers.")
    
    st.title("Chat with Your PDFs 📄🔍")
    if selected_pdfs:
        st.subheader(f"Active PDFs: {', '.join(selected_pdfs)}")
    else:
        st.subheader("No PDFs selected. Please choose/upload PDFs in the sidebar.")
    
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
    
    user_input = st.chat_input("Ask a question about your selected PDFs...")
    if user_input and selected_pdfs:
        st.session_state.history.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Retrieving context from PDFs..."):
            docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
            context = build_context(docs)
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Generating answer..."):
                answer = generate_answer(user_input, context)
            typewriter_effect(answer, placeholder)
            st.session_state.history.append({"role": "assistant", "text": answer})

if __name__ == "__main__":
    main()
# import os
# import time
# import json
# import requests
# import shutil
# import concurrent.futures
# from typing import List, Dict, Any

# import streamlit as st
# import torch
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from angle_emb import AnglE, Prompts
# from langchain_core.embeddings import Embeddings
# from rank_bm25 import BM25Okapi  # For BM25 re-ranking

# # --------------------------------------------------------------------------------
# # Constants & Directories
# # --------------------------------------------------------------------------------
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 100
# MAX_WORKERS = 4

# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"
# SITE_NAME = "Multi-PDF Chat App"

# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # --------------------------------------------------------------------------------
# # Embeddings & Vector Store
# # --------------------------------------------------------------------------------
# class EnhancedAnglEEmbeddings(Embeddings):
#     def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1',
#                                            pooling_strategy='cls',
#                                            device=self.device)
#         self.max_length = 512
#         # For potential dimension checks (if needed)
#         self.embedding_dimension = self.angle.encode(
#             ["dimension_check"], normalize_embedding=True, max_length=self.max_length
#         )[0].shape[0]

#     def _process_text(self, text: str) -> str:
#         tokens = text.split()
#         return ' '.join(tokens[:self.max_length]) if len(tokens) > self.max_length else ' '.join(tokens)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         processed = [self._process_text(t) for t in texts]
#         embeddings = []
#         batch_size = 32
#         for i in range(0, len(processed), batch_size):
#             batch = processed[i:i+batch_size]
#             embeddings.extend(
#                 self.angle.encode(batch, normalize_embedding=True, max_length=self.max_length).tolist()
#             )
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         processed = self._process_text(text)
#         query_text = Prompts.C.format(
#             text=f"Query: {processed}\nFind relevant information about: {processed}"
#         )
#         return self.angle.encode([query_text], normalize_embedding=True, max_length=self.max_length)[0].tolist()

# @st.cache_resource(show_spinner=False)
# def get_embeddings() -> EnhancedAnglEEmbeddings:
#     return EnhancedAnglEEmbeddings()

# @st.cache_resource(show_spinner=False)
# def get_vector_store() -> Chroma:
#     embeddings = get_embeddings()
#     store = Chroma(
#         collection_name="multi_pdf_store",
#         embedding_function=embeddings,
#         persist_directory=DB_DIRECTORY
#     )
#     # Check for embedding dimension mismatch via a dummy query.
#     try:
#         store.similarity_search("dimension_test", k=1)
#     except Exception as e:
#         if "dimension" in str(e).lower():
#             shutil.rmtree(DB_DIRECTORY, ignore_errors=True)
#             os.makedirs(DB_DIRECTORY, exist_ok=True)
#             store = Chroma(
#                 collection_name="multi_pdf_store",
#                 embedding_function=embeddings,
#                 persist_directory=DB_DIRECTORY
#             )
#         else:
#             raise e
#     return store

# # --------------------------------------------------------------------------------
# # PDF Processing
# # --------------------------------------------------------------------------------
# class EnhancedPDFProcessor:
#     def __init__(self) -> None:
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         return PDFPlumberLoader(file_path).load()
    
#     def process_pdf(self, file, vector_store: Chroma) -> bool:
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
#             for i, chunk in enumerate(chunks):
#                 chunk.metadata.update({
#                     'source': file.name,
#                     'chunk_index': i,
#                     'total_chunks': len(chunks),
#                     'chars_count': len(chunk.page_content),
#                     'words_count': len(chunk.page_content.split())
#                 })
#             for attempt in range(3):
#                 try:
#                     vector_store.add_documents(chunks)
#                     vector_store.persist()
#                     break
#                 except Exception:
#                     if attempt == 2:
#                         raise
#                     time.sleep(1)
#             return True
#         except Exception as e:
#             st.error(f"Error processing '{file.name}': {str(e)}")
#             return False

# def get_stored_pdfs(vector_store: Chroma) -> List[str]:
#     try:
#         result = vector_store._collection.get(include=['metadatas'])
#         if result and 'metadatas' in result:
#             return list({meta['source'] for meta in result['metadatas'] if meta and 'source' in meta})
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# # --------------------------------------------------------------------------------
# # Retrieval & Context Building (Hybrid Retrieval Only)
# # --------------------------------------------------------------------------------
# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store: Chroma) -> List[Any]:
#     """
#     Retrieve context documents for a query from the vector store using a hybrid retrieval approach.
#     This method fuses dense (semantic) scores with BM25 lexical scores for optimal retrieval.
#     """
#     if not selected_pdfs:
#         return []
    
#     # Dynamically set k based on query length and number of selected PDFs.
#     query_words = len(query.split())
#     base_k = min(4 * len(selected_pdfs), 10)
#     k = min(base_k + (query_words // 10), 15)
#     filter_dict = {"source": {"$in": selected_pdfs}}
    
#     # Dense retrieval using Chroma (returns (doc, score) pairs)
#     dense_results = vector_store.similarity_search_with_relevance_scores(query, k=k, filter=filter_dict)
    
#     # Build a BM25 index on the retrieved documents.
#     docs = [doc for doc, _ in dense_results]
#     tokenized_docs = [doc.page_content.split() for doc in docs]
#     bm25 = BM25Okapi(tokenized_docs)
    
#     # Compute BM25 scores for the query once.
#     query_tokens = query.split()
#     bm25_scores = bm25.get_scores(query_tokens)
    
#     # Fuse dense and BM25 scores.
#     hybrid_results = []
#     for idx, (doc, dense_score) in enumerate(dense_results):
#         bm25_score = bm25_scores[idx]
#         combined_score = 0.5 * dense_score + 0.5 * bm25_score  # Equal weight fusion
#         hybrid_results.append((doc, combined_score))
    
#     # Sort documents by the combined score in descending order.
#     hybrid_results.sort(key=lambda x: x[1], reverse=True)
#     return [doc for doc, score in hybrid_results]

# def build_context(docs: List[Any]) -> str:
#     if not docs:
#         return ""
#     groups: Dict[str, List[Any]] = {}
#     for doc in docs:
#         groups.setdefault(doc.metadata['source'], []).append(doc)
#     context_parts = [
#         f"From {source}:\n" + "\n".join(chunk.page_content for chunk in sorted(chunks, key=lambda c: c.metadata.get('chunk_index', 0)))
#         for source, chunks in groups.items()
#     ]
#     return "\n\n".join(context_parts)

# # --------------------------------------------------------------------------------
# # LLM Interaction
# # --------------------------------------------------------------------------------
# def get_last_two_history() -> List[Dict[str, str]]:
#     history = st.session_state.get("history", [])
#     return history[-2:] if len(history) >= 2 else history

# def generate_answer(question: str, context: str) -> str:
#     messages = [{
#         "role": "system",
#         "content": ("You are a helpful assistant that answers questions based on the provided PDF context and recent chat history. "
#                     "Provide accurate, well-structured responses and cite the specific PDF sources when possible.")
#     }]
#     for msg in get_last_two_history():
#         messages.append({"role": msg["role"], "content": msg["text"]})
#     prompt = (
#         f"Context from relevant PDF sections:\n\n{context}\n\n"
#         f"Question: {question}\n\n"
#         "Please provide a comprehensive answer based on the context above. "
#         "If the context doesn't contain enough information to fully answer the question, please indicate that clearly."
#     )
#     messages.append({"role": "user", "content": prompt})
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",
#         "messages": messages,
#         "temperature": 0.6,
#         "max_tokens": 66000
#     }
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
#     try:
#         response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload))
#         data = response.json()
#         if "choices" not in data:
#             st.error(f"API response error: {data}")
#             return f"Error generating answer: 'choices' missing. Full response: {data}"
#         return data["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Error generating answer: {str(e)}"

# def typewriter_effect(text: str, placeholder) -> None:
#     output = ""
#     for line in text.split('\n'):
#         output += line + "\n"
#         placeholder.markdown(output + "▌")
#         time.sleep(0.01)
#     placeholder.markdown(output)

# # --------------------------------------------------------------------------------
# # Main Streamlit App
# # --------------------------------------------------------------------------------
# def main() -> None:
#     st.set_page_config(page_title="Multi-PDF Chat", layout="wide", page_icon="🤖")
#     vector_store = get_vector_store()
#     pdf_processor = EnhancedPDFProcessor()
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     with st.sidebar:
#         st.header("PDF Manager")
#         st.info("1. Upload PDFs below.\n2. They are processed automatically.\n3. Select which PDFs to use for context.\n4. Ask questions in the main chat area!")
#         stored_pdfs = get_stored_pdfs(vector_store)
#         uploaded_files = st.file_uploader("Upload PDFs (Max 200MB each)", type="pdf", accept_multiple_files=True)
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 new_files = [f for f in uploaded_files if f.name not in stored_pdfs]
#                 if new_files:
#                     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                         list(executor.map(lambda file: pdf_processor.process_pdf(file, vector_store), new_files))
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
#         selected_pdfs = st.multiselect("Select PDFs for context:", options=stored_pdfs, default=stored_pdfs) if stored_pdfs else []
#         if st.button("Clear Chat History"):
#             st.session_state.history = []
#         st.markdown("---")
#         st.header("About")
#         st.markdown("This **Multi-PDF Chat App** lets you upload multiple PDFs and ask questions. A Large Language Model references the content of these PDFs to provide context-aware answers.")
    
#     st.title("Chat with Your PDFs 📄🔍")
#     if selected_pdfs:
#         st.subheader(f"Active PDFs: {', '.join(selected_pdfs)}")
#     else:
#         st.subheader("No PDFs selected. Please choose/upload PDFs in the sidebar.")
    
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     user_input = st.chat_input("Ask a question about your selected PDFs...")
#     if user_input and selected_pdfs:
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         with st.spinner("Retrieving context from PDFs..."):
#             docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#             context = build_context(docs)
        
#         with st.chat_message("assistant"):
#             placeholder = st.empty()
#             with st.spinner("Generating answer..."):
#                 answer = generate_answer(user_input, context)
#             typewriter_effect(answer, placeholder)
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()
# import os
# import time
# import json
# import requests
# import shutil
# import concurrent.futures
# from typing import List, Dict, Any

# import streamlit as st
# import torch
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from angle_emb import AnglE, Prompts
# from langchain_core.embeddings import Embeddings

# # --------------------------------------------------------------------------------
# # 1. Constants & Configuration
# # --------------------------------------------------------------------------------
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 500   # Adjust chunk size to your needs
# CHUNK_OVERLAP = 100  # Adjust overlap for context continuity
# MAX_WORKERS = 4

# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"  # or your deployed Streamlit URL
# SITE_NAME = "Multi-PDF Chat App"

# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # --------------------------------------------------------------------------------
# # 2. Embeddings & Vector Store
# # --------------------------------------------------------------------------------
# class EnhancedAnglEEmbeddings(Embeddings):
#     """
#     Enhanced embedding system with improved query handling and document processing.
#     """

#     def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         self.device = device
#         self.angle = AnglE.from_pretrained(
#             'WhereIsAI/UAE-Large-V1',
#             pooling_strategy='cls',
#             device=self.device
#         )
#         self.max_length = 512  # Maximum sequence length

#         # Optional: We can test embedding dimension if needed:
#         self.embedding_dimension = self.angle.encode(
#             ["dimension_check"], normalize_embedding=True, max_length=self.max_length
#         )[0].shape[0]

#     def _process_text(self, text: str) -> str:
#         """Clean and prepare text for embedding."""
#         text = ' '.join(text.split())
#         tokens = text.split()
#         if len(tokens) > self.max_length:
#             text = ' '.join(tokens[:self.max_length])
#         return text

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for document chunks with preprocessing."""
#         processed_texts = [self._process_text(t) for t in texts]
        
#         batch_size = 32
#         all_embeddings = []
#         for i in range(0, len(processed_texts), batch_size):
#             batch = processed_texts[i:i + batch_size]
#             batch_embeddings = self.angle.encode(
#                 batch,
#                 normalize_embedding=True,
#                 max_length=self.max_length
#             )
#             all_embeddings.extend(batch_embeddings.tolist())
        
#         return all_embeddings

#     def embed_query(self, text: str) -> List[float]:
#         """Generate embedding for query with enhanced prompting."""
#         processed_text = self._process_text(text)
#         query_text = Prompts.C.format(
#             text=f"Query: {processed_text}\nFind relevant information about: {processed_text}"
#         )
#         embedding = self.angle.encode(
#             [query_text],
#             normalize_embedding=True,
#             max_length=self.max_length
#         )
#         return embedding[0].tolist()


# @st.cache_resource(show_spinner=False)
# def get_embeddings() -> EnhancedAnglEEmbeddings:
#     """Initialize and cache the enhanced embedding system."""
#     return EnhancedAnglEEmbeddings()

# @st.cache_resource(show_spinner=False)
# def get_vector_store() -> Chroma:
#     """
#     Initialize and cache the Chroma vector store, then do a dimension check.
#     If an exception indicates a dimension mismatch, rebuild a fresh DB.
#     """
#     embeddings = get_embeddings()
#     store = Chroma(
#         collection_name="multi_pdf_store",
#         embedding_function=embeddings,
#         persist_directory=DB_DIRECTORY
#     )
    
#     # Attempt a dummy query to verify dimension compatibility
#     dummy_text = "dimension_test"
#     try:
#         store.similarity_search(dummy_text, k=1)
#     except Exception as e:
#         # If the error message references 'dimension', we assume mismatch
#         if "dimension" in str(e).lower():
#             shutil.rmtree(DB_DIRECTORY, ignore_errors=True)
#             os.makedirs(DB_DIRECTORY, exist_ok=True)
#             store = Chroma(
#                 collection_name="multi_pdf_store",
#                 embedding_function=embeddings,
#                 persist_directory=DB_DIRECTORY
#             )
#         else:
#             # Some other error, raise it for debugging
#             raise e
    
#     return store

# # --------------------------------------------------------------------------------
# # 3. PDF Processing & Storage
# # --------------------------------------------------------------------------------
# class EnhancedPDFProcessor:
#     """
#     Handles loading and splitting PDF files into text chunks
#     with metadata and improved chunking.
#     """

#     def __init__(self) -> None:
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         """Load a PDF file using PDFPlumberLoader."""
#         loader = PDFPlumberLoader(file_path)
#         return loader.load()
    
#     def process_pdf(self, file, vector_store: Chroma) -> bool:
#         """
#         Process a single PDF file:
#           - Save locally to PDFS_DIRECTORY.
#           - Load & split text into chunks.
#           - Add metadata (filename, chunk index, etc.) to chunks.
#           - Store results in the vector store with retries.
#         """
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
            
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
            
#             for i, chunk in enumerate(chunks):
#                 chunk.metadata.update({
#                     'source': file.name,
#                     'chunk_index': i,
#                     'total_chunks': len(chunks),
#                     'chars_count': len(chunk.page_content),
#                     'words_count': len(chunk.page_content.split())
#                 })
            
#             max_retries = 3
#             for attempt in range(max_retries):
#                 try:
#                     vector_store.add_documents(chunks)
#                     vector_store.persist()  # Persist data
#                     break
#                 except Exception:
#                     if attempt == max_retries - 1:
#                         raise
#                     time.sleep(1)
            
#             return True
        
#         except Exception as e:
#             st.error(f"Error processing '{file.name}': {str(e)}")
#             return False

# def get_stored_pdfs(vector_store: Chroma) -> List[str]:
#     """
#     Fetch a unique list of PDF filenames that have been processed and stored.
#     """
#     try:
#         result = vector_store._collection.get(include=['metadatas'])
#         if result and 'metadatas' in result:
#             sources = [
#                 meta['source']
#                 for meta in result['metadatas'] if meta and 'source' in meta
#             ]
#             return list(set(sources))
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# # --------------------------------------------------------------------------------
# # 4. Retrieval & Context Building
# # --------------------------------------------------------------------------------
# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store: Chroma) -> List[Any]:
#     """
#     Enhanced retrieval system: 
#       - Dynamically set k based on query length & PDF selection.
#       - Filter documents by selected PDFs.
#       - Perform similarity search with a relevance threshold.
#       - Sort chunks by their original order.
#     """
#     if not selected_pdfs:
#         return []
    
#     query_words = len(query.split())
#     base_k = min(4 * len(selected_pdfs), 10)
#     k = min(base_k + (query_words // 10), 15)  # adapt if needed
    
#     filter_dict = {"source": {"$in": selected_pdfs}}
#     results = vector_store.similarity_search_with_relevance_scores(
#         query,
#         k=k,
#         filter=filter_dict
#     )
    
#     threshold = 0.3  # Filter out low-relevance chunks
#     filtered_results = [doc for doc, score in results if score > threshold]
    
#     filtered_results.sort(key=lambda x: (
#         x.metadata['source'],
#         x.metadata.get('chunk_index', 0)
#     ))
    
#     return filtered_results

# def build_context(docs: List[Any]) -> str:
#     """
#     Build a coherent context string grouped by source PDF.
#     """
#     if not docs:
#         return ""
    
#     doc_groups = {}
#     for doc in docs:
#         src = doc.metadata['source']
#         doc_groups.setdefault(src, []).append(doc)
    
#     context_parts = []
#     for source, chunks in doc_groups.items():
#         chunks.sort(key=lambda x: x.metadata.get('chunk_index', 0))
#         source_text = "\n".join(c.page_content for c in chunks)
#         context_parts.append(f"From {source}:\n{source_text}")
    
#     return "\n\n".join(context_parts)

# # --------------------------------------------------------------------------------
# # 5. LLM Interaction
# # --------------------------------------------------------------------------------
# def get_last_two_history() -> List[Dict[str, str]]:
#     """Retrieve the last two messages from session history."""
#     history = st.session_state.get("history", [])
#     return history[-2:] if len(history) >= 2 else history

# def generate_answer(question: str, context: str) -> str:
#     """
#     Generate answer using OpenRouter (LLM).
#     Includes:
#       - A system role message with instructions.
#       - The last two conversation turns.
#       - A prompt with PDF-based context and user question.
#     """
#     last_two = get_last_two_history()
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a helpful assistant that answers questions based on the provided PDF "
#                 "context and recent chat history. Provide accurate, well-structured responses "
#                 "and cite the specific PDF sources when possible."
#             )
#         }
#     ]
    
#     # Add the last two history messages
#     for msg in last_two:
#         messages.append({"role": msg["role"], "content": msg["text"]})
    
#     # Combine context + question into a user role message
#     prompt = (
#         f"Context from relevant PDF sections:\n\n{context}\n\n"
#         f"Question: {question}\n\n"
#         "Please provide a comprehensive answer based on the context above. "
#         "If the context doesn't contain enough information to fully answer the question, "
#         "please indicate that clearly."
#     )
    
#     messages.append({"role": "user", "content": prompt})
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",  # or adjust to your desired model
#         "messages": messages,
#         "temperature": 0.6,
#         "max_tokens": 66000
#     }
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
    
#     try:
#         response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload))
#         data = response.json()
#         if "choices" not in data:
#             st.error(f"API response error: {data}")
#             return f"Error generating answer: 'choices' missing. Full response: {data}"
        
#         return data["choices"][0]["message"]["content"]
    
#     except Exception as e:
#         return f"Error generating answer: {str(e)}"

# # --------------------------------------------------------------------------------
# # 6. Main Streamlit App
# # --------------------------------------------------------------------------------
# def main() -> None:
#     """
#     Multi-PDF Chat App:
#       - Upload & process PDFs to store in vector DB (Chroma).
#       - Select which PDFs to reference.
#       - Ask a question that references these PDFs.
#       - Get context-aware answers from an LLM.
#     """
#     st.set_page_config(
#         page_title="Multi-PDF Chat",
#         layout="wide",
#         page_icon="🤖"
#     )
    
#     # Initialize or retrieve cached vector store
#     vector_store = get_vector_store()
#     pdf_processor = EnhancedPDFProcessor()
    
#     # Initialize conversation history
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     # -----------------------------
#     # Sidebar
#     # -----------------------------
#     with st.sidebar:
#         st.header("PDF Manager")
#         st.info(
#             "1. Upload your PDFs below.\n"
#             "2. They will be processed automatically.\n"
#             "3. Select which PDFs to use for context.\n"
#             "4. Ask your questions in the main chat area!"
#         )
        
#         stored_pdfs = get_stored_pdfs(vector_store)
        
#         # File uploader
#         uploaded_files = st.file_uploader(
#             "Upload PDFs (Max 200MB each)",
#             type="pdf",
#             accept_multiple_files=True
#         )
        
#         # Process newly uploaded PDFs
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 new_pdfs = [f for f in uploaded_files if f.name not in stored_pdfs]
#                 if new_pdfs:
#                     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                         tasks = [
#                             executor.submit(pdf_processor.process_pdf, file, vector_store)
#                             for file in new_pdfs
#                         ]
#                         concurrent.futures.wait(tasks)
                
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
        
#         # Let user select which PDFs to include in the search
#         if stored_pdfs:
#             selected_pdfs = st.multiselect(
#                 "Select PDFs for context:",
#                 options=stored_pdfs,
#                 default=stored_pdfs
#             )
#         else:
#             selected_pdfs = []
#             st.warning("No PDFs found. Please upload some!")
        
#         if st.button("Clear Chat History"):
#             st.session_state.history = []
        
#         st.markdown("---")
#         st.header("About")
#         st.markdown(
#             "This **Multi-PDF Chat App** allows you to upload multiple PDFs and ask questions. "
#             "A Large Language Model references the content of these PDFs to provide context-aware answers."
#         )
    
#     # -----------------------------
#     # Main Chat Interface
#     # -----------------------------
#     st.title("Chat with Your PDFs 📄🔍")
    
#     if selected_pdfs:
#         st.subheader(f"Active PDFs: {', '.join(selected_pdfs)}")
#     else:
#         st.subheader("No PDFs selected. Please choose/upload PDFs in the sidebar.")
    
#     # Display the conversation history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     # Chat input
#     user_input = st.chat_input("Ask a question about your selected PDFs...")
    
#     # If user asked a question AND we have selected PDFs
#     if user_input and selected_pdfs:
#         # 1) Log user message
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # 2) Retrieve context
#         with st.spinner("Retrieving context from PDFs..."):
#             docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#             context = build_context(docs)
        
#         # 3) Generate LLM answer
#         with st.chat_message("assistant"):
#             response_placeholder = st.empty()
#             with st.spinner("Generating answer..."):
#                 answer = generate_answer(user_input, context)
            
#             # Optional: Typewriter effect
#             full_response = ""
#             for chunk in answer.split('\n'):
#                 full_response += chunk + "\n"
#                 response_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.01)
#             response_placeholder.markdown(full_response)
            
#             # 4) Add assistant message to history
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()

# ============================================================================================
# import os
# import time
# import json
# import requests
# import concurrent.futures
# from typing import List, Dict, Any

# import streamlit as st
# import torch
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma

# # --------------------------------------------------------------------------------
# # 1. Constants & Configuration
# # --------------------------------------------------------------------------------
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# MAX_WORKERS = 4  # Could use os.cpu_count() or a similar approach.

# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"
# SITE_NAME = "Multi-PDF Chat App"

# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # Minimizing Torch memory usage if no GPU is used.
# # torch.classes.__path__ = []

# # --------------------------------------------------------------------------------
# # 2. Caching & Initialization
# # --------------------------------------------------------------------------------
# @st.cache_resource(show_spinner=False)
# def get_embeddings() -> HuggingFaceEmbeddings:
#     """
#     Initialize and cache the HuggingFace embeddings model.
#     Using 'cpu' by default. If you have a GPU, set device='cuda'.
#     """
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={'device': 'cpu'}  # Change to 'cuda' if your environment supports GPU.
#     )

# @st.cache_resource(show_spinner=False)
# def get_vector_store() -> Chroma:
#     """
#     Initialize and cache the Chroma vector store.
#     Chroma automatically persists data in DB_DIRECTORY.
#     """
#     return Chroma(
#         collection_name="multi_pdf_store",
#         embedding_function=get_embeddings(),
#         persist_directory=DB_DIRECTORY
#     )

# # --------------------------------------------------------------------------------
# # 3. PDF Processing
# # --------------------------------------------------------------------------------
# class PDFProcessor:
#     """
#     Handles loading and splitting PDF files into text chunks.
#     Utilizes parallel processing for multiple PDFs.
#     """
#     def __init__(self) -> None:
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         """Load a PDF file using PDFPlumber."""
#         loader = PDFPlumberLoader(file_path)
#         return loader.load()
    
#     def process_pdf(self, file, vector_store: Chroma) -> bool:
#         """
#         Process a single PDF file:
#           - Save locally to PDFS_DIRECTORY.
#           - Load & split text into chunks.
#           - Add metadata (filename) to chunks.
#           - Store results in the vector store.
#         """
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
            
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
            
#             # Assign each chunk's source metadata to the filename
#             for chunk in chunks:
#                 chunk.metadata['source'] = file.name
            
#             # Store chunks in Chroma
#             vector_store.add_documents(chunks)
#             return True
        
#         except Exception as e:
#             st.error(f"Error processing '{file.name}': {str(e)}")
#             return False

# def get_stored_pdfs(vector_store: Chroma) -> List[str]:
#     """
#     Fetch a unique list of PDF filenames that have been processed and stored.
#     """
#     try:
#         result = vector_store._collection.get(include=['metadatas'])
#         if result and 'metadatas' in result:
#             return list({meta['source'] for meta in result['metadatas'] if meta and 'source' in meta})
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# # --------------------------------------------------------------------------------
# # 4. LLM Interaction & Chat Helpers
# # --------------------------------------------------------------------------------
# def get_last_two_history() -> List[Dict[str, str]]:
#     """Retrieve the last two messages from session history."""
#     history = st.session_state.get("history", [])
#     return history[-2:] if len(history) >= 2 else history

# def generate_answer(question: str, context: str) -> str:
#     """
#     Interact with the LLM via OpenRouter.
#     The prompt includes:
#       - A system message with instructions.
#       - The last two conversation turns.
#       - The user's new question plus PDF-based context.
#     """
#     last_two = get_last_two_history()
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You are a helpful assistant that answers questions based on both the provided PDF "
#                 "context and the recent chat history. Provide concise, accurate responses."
#             )
#         }
#     ]
    
#     # Add the last two messages from the conversation history
#     for msg in last_two:
#         messages.append({"role": msg["role"], "content": msg["text"]})
    
#     # Include current user query plus any retrieved PDF context
#     messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"})
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",
#         "messages": messages,
#         "temperature": 0.6,
#         "max_tokens": 66000
#     }
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
    
#     try:
#         response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload))
#         data = response.json()
#         if "choices" not in data:
#             st.error(f"API response error: {data}")
#             return f"Error generating answer: response missing 'choices'. Full response: {data}"
        
#         return data["choices"][0]["message"]["content"]
    
#     except Exception as e:
#         return f"Error generating answer: {str(e)}"

# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store: Chroma) -> List[Any]:
#     """
#     Return the top-k relevant text chunks from the selected PDFs.
#     Uses similarity search in the Chroma vector store.
#     """
#     if not selected_pdfs:
#         return []
    
#     filter_dict = {"source": {"$in": selected_pdfs}}
#     k = min(4 * len(selected_pdfs), 10)
#     return vector_store.similarity_search(query, k=k, filter=filter_dict)

# # --------------------------------------------------------------------------------
# # 5. Main Streamlit App
# # --------------------------------------------------------------------------------
# def main() -> None:
#     """Entry point: Multi-PDF Chat App."""
#     st.set_page_config(
#         page_title="Multi-PDF Chat",
#         layout="wide",
#         page_icon="🤖"  # Page icon can be changed or removed if desired
#     )
    
#     # Initialize or retrieve cached vector store
#     vector_store = get_vector_store()
#     pdf_processor = PDFProcessor()
    
#     # Initialize conversation history
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     # -----------------------------
#     # Sidebar
#     # -----------------------------
#     with st.sidebar:
#         st.header("PDF Manager")
#         st.info(
#             "1. Upload your PDFs below.\n"
#             "2. They will be processed automatically.\n"
#             "3. Select which PDFs to use for context.\n"
#             "4. Ask your questions in the main chat area!"
#         )
        
#         stored_pdfs = get_stored_pdfs(vector_store)
        
#         # File uploader
#         uploaded_files = st.file_uploader(
#             "Upload PDFs (Max 200MB each)",
#             type="pdf",
#             accept_multiple_files=True
#         )
        
#         # Process newly uploaded PDFs
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 new_pdfs = [f for f in uploaded_files if f.name not in stored_pdfs]
#                 if new_pdfs:
#                     with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                         futures = [
#                             executor.submit(pdf_processor.process_pdf, file, vector_store)
#                             for file in new_pdfs
#                         ]
#                         concurrent.futures.wait(futures)
                
#             # Refresh stored PDF list after processing
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
        
#         # Let user select which PDFs to include in the search
#         if stored_pdfs:
#             selected_pdfs = st.multiselect(
#                 "Select PDFs for context:",
#                 options=stored_pdfs,
#                 default=stored_pdfs
#             )
#         else:
#             selected_pdfs = []
#             st.warning("No PDFs found. Please upload some!")
        
#         if st.button("Clear Chat History"):
#             st.session_state.history = []
        
#         st.markdown("---")
#         st.header("About")
#         st.markdown(
#             "This **Multi-PDF Chat App** allows you to upload multiple PDFs and ask questions about them. "
#             "It uses a Large Language Model that references the content in these PDFs to provide context-aware answers."
#         )
    
#     # -----------------------------
#     # Main Chat Interface
#     # -----------------------------
#     st.title("Chat with Your PDFs 📄🔍")
    
#     if selected_pdfs:
#         st.subheader(f"Active PDFs: {', '.join(selected_pdfs)}")
#     else:
#         st.subheader("No PDFs selected. Please choose/upload PDFs in the sidebar.")
    
#     # Display the conversation history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     # Chat input text field
#     user_input = st.chat_input("Ask a question about your selected PDFs...")
    
#     # If user typed a question and PDFs are selected
#     if user_input and selected_pdfs:
#         # Add the user's message to history
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # Assistant response area
#         with st.chat_message("assistant"):
#             response_placeholder = st.empty()
            
#             # Retrieve PDF context
#             with st.spinner("Retrieving context from PDFs..."):
#                 docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#                 context = "\n\n".join(doc.page_content for doc in docs)
            
#             # Generate LLM answer
#             with st.spinner("Generating answer..."):
#                 answer = generate_answer(user_input, context)
            
#             # Stream the response with a typewriter effect
#             full_response = ""
#             for chunk in answer.split('\n'):
#                 full_response += chunk + "\n"
#                 response_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.01)  # Adjust speed to your preference
#             response_placeholder.markdown(full_response)
            
#             # Save the assistant's answer in history
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()
#============================================================================================================================
#so far best onee
# import os
# import time
# import json
# import requests
# import concurrent.futures
# from typing import List, Dict, Any

# import streamlit as st
# import torch
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# # Use the updated Chroma package from langchain_chroma
# from langchain_chroma import Chroma

# # Performance optimization: Disable torch CUDA if not needed
# torch.classes.__path__ = []

# # --- Constants & Configuration ---
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# MAX_WORKERS = 4  # Adjust based on your CPU cores

# # Ensure necessary directories exist
# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # OpenRouter Configuration (replace with your actual API key)
# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"
# SITE_NAME = "Multi-PDF Chat App"

# # --- Helper Functions & Caching ---
# @st.cache_resource(show_spinner=False)
# def get_embeddings() -> HuggingFaceEmbeddings:
#     """
#     Initialize and cache the HuggingFace embeddings model.
#     Using a well-supported sentence-transformers model to avoid warnings.
#     """
#     model_name = "sentence-transformers/all-mpnet-base-v2"
#     return HuggingFaceEmbeddings(
#         model_name=model_name,
#         model_kwargs={'device': 'cpu'}
#     )

# @st.cache_resource(show_spinner=False)
# def get_vector_store() -> Chroma:
#     """
#     Initialize and cache the Chroma vector store.
#     Persistence is now handled automatically.
#     """
#     return Chroma(
#         collection_name="multi_pdf_store",
#         embedding_function=get_embeddings(),
#         persist_directory=DB_DIRECTORY
#     )

# # --- PDF Processing ---
# class PDFProcessor:
#     """Process PDF files and convert them into vectorized text chunks."""
#     def __init__(self) -> None:
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         """Load a PDF file using PDFPlumberLoader."""
#         loader = PDFPlumberLoader(file_path)
#         return loader.load()
    
#     def process_pdf(self, file, vector_store: Chroma) -> bool:
#         """
#         Process a PDF file:
#           - Save the file locally.
#           - Extract and split text.
#           - Add source metadata.
#           - Update the vector store.
#         """
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
            
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
            
#             for chunk in chunks:
#                 chunk.metadata['source'] = file.name
            
#             vector_store.add_documents(chunks)
#             return True
#         except Exception as e:
#             st.error(f"Error processing '{file.name}': {str(e)}")
#             return False

# def get_stored_pdfs(vector_store: Chroma) -> List[str]:
#     """
#     Retrieve a unique list of PDF filenames that have been processed
#     and stored in the vector store.
#     """
#     try:
#         result = vector_store._collection.get(include=['metadatas'])
#         if result and 'metadatas' in result:
#             return list({meta['source'] for meta in result['metadatas'] if meta and 'source' in meta})
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# # --- Chat History & LLM Communication ---
# def get_last_two_history() -> List[Dict[str, str]]:
#     """Retrieve the last two chat messages from session history."""
#     history = st.session_state.get("history", [])
#     return history[-2:] if len(history) >= 2 else history

# def generate_answer(question: str, context: str) -> str:
#     """
#     Generate an answer from the LLM.
#     Constructs a payload that includes:
#       - A system message.
#       - The last two conversation messages.
#       - The current question with PDF context.
#     Includes additional error checking for missing 'choices'.
#     """
#     last_two = get_last_two_history()
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context and conversation history."},
#     ]
#     for msg in last_two:
#         messages.append({"role": msg["role"], "content": msg["text"]})
#     messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"})
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",
#         "messages": messages,
#         "temperature": 0.7,
#         "max_tokens": 66000
#     }
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
    
#     try:
#         response = requests.post(
#             url=f"{BASE_URL}/chat/completions",
#             headers=headers,
#             data=json.dumps(payload)
#         )
#         data = response.json()
#         if "choices" not in data:
#             st.error(f"API response error: {data}")
#             return f"Error generating answer: response does not contain 'choices'. Full response: {data}"
#         return data["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Error generating answer: {str(e)}"

# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store: Chroma) -> List[Any]:
#     """
#     Retrieve documents relevant to the query from the selected PDFs.
#     The number of documents returned adapts based on the number of PDFs selected.
#     """
#     if not selected_pdfs:
#         return []
    
#     filter_dict = {"source": {"$in": selected_pdfs}}
#     k = min(4 * len(selected_pdfs), 10)
#     return vector_store.similarity_search(query, k=k, filter=filter_dict)

# # --- Main Application ---
# def main() -> None:
#     """Main function to run the Multi-PDF Chat application."""
#     st.set_page_config(page_title="Multi-PDF Chat", layout="wide")
    
#     vector_store = get_vector_store()
#     pdf_processor = PDFProcessor()
    
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     # --- Sidebar: PDF Manager & App Info ---
#     with st.sidebar:
#         st.header("PDF Manager")
#         st.info(
#             "Upload your PDFs below. After processing, select the PDFs you want to chat about. "
#             "The assistant will use these PDFs as context to answer your questions."
#         )
        
#         stored_pdfs = get_stored_pdfs(vector_store)
        
#         uploaded_files = st.file_uploader("Upload PDFs (Max 200MB per file)", type="pdf", accept_multiple_files=True)
        
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                     futures = [executor.submit(pdf_processor.process_pdf, file, vector_store)
#                                for file in uploaded_files if file.name not in stored_pdfs]
#                     concurrent.futures.wait(futures)
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
        
#         if stored_pdfs:
#             selected_pdfs = st.multiselect("Select PDFs to include in the chat:", options=stored_pdfs, default=[stored_pdfs[0]])
#         else:
#             selected_pdfs = []
#             st.warning("No PDFs available. Please upload PDF files.")
        
#         if st.button("Clear Chat History"):
#             st.session_state.history = []
        
#         st.markdown("---")
#         st.header("About")
#         st.markdown(
#             "Multi-PDF Chat App allows you to interact with multiple PDFs using advanced LLM capabilities. "
#             "Upload PDFs, select them, and ask questions based on their content."
#         )
    
#     # --- Main Chat Interface ---
#     st.title("Chat with Your PDFs")
    
#     if selected_pdfs:
#         st.subheader(f"Active PDFs: {', '.join(selected_pdfs)}")
#     else:
#         st.subheader("No PDFs selected. Please select/upload PDFs in the sidebar.")
    
#     # Display chat history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     # Chat input
#     user_input = st.chat_input("Ask a question related to the selected PDFs...")
    
#     if user_input and selected_pdfs:
#         # Add user message to history
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # Retrieve context and generate answer
#         with st.chat_message("assistant"):
#             response_placeholder = st.empty()
#             with st.spinner("Searching relevant documents..."):
#                 docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#                 context = "\n\n".join(doc.page_content for doc in docs)
            
#             with st.spinner("Generating answer..."):
#                 answer = generate_answer(user_input, context)
            
#             # Simulate streaming the response with a typewriter effect
#             full_response = ""
#             for chunk in answer.split('\n'):
#                 full_response += chunk + "\n"
#                 response_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.02)
#             response_placeholder.markdown(full_response)
            
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()











#================================================================================================================================
    #workinggggggg with history
# import os
# import time
# import requests
# import json
# import streamlit as st
# from typing import List, Dict, Any
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import torch
# import concurrent.futures

# # Performance optimization: Disable torch CUDA if not needed
# torch.classes.__path__ = []

# # --- Constants & Config ---
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# MAX_WORKERS = 4  # Adjust based on your CPU cores

# # Ensure directories exist
# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # OpenRouter Configuration
# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"
# SITE_NAME = "Multi-PDF Chat App"

# # Initialize embeddings with caching
# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="allenai/specter",
#         model_kwargs={'device': 'cpu'}
#     )

# # Initialize vector store with caching
# @st.cache_resource
# def get_vector_store():
#     return Chroma(
#         persist_directory=DB_DIRECTORY,
#         embedding_function=get_embeddings(),
#         collection_name="multi_pdf_store"
#     )

# # Optimized document processing
# class PDFProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         loader = PDFPlumberLoader(file_path)
#         return loader.load()
    
#     def process_pdf(self, file, vector_store):
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
            
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
            
#             # Add source metadata
#             for chunk in chunks:
#                 chunk.metadata['source'] = file.name
            
#             vector_store.add_documents(chunks)
#             vector_store.persist()
#             return True
#         except Exception as e:
#             st.error(f"Error processing {file.name}: {str(e)}")
#             return False

# def get_stored_pdfs(vector_store) -> List[str]:
#     """Efficiently retrieve list of stored PDFs"""
#     try:
#         result = vector_store._collection.get(
#             include=['metadatas']
#         )
#         if result and 'metadatas' in result:
#             return list(set(
#                 meta['source'] for meta in result['metadatas'] 
#                 if meta and 'source' in meta
#             ))
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# def get_last_two_history() -> List[Dict[str, str]]:
#     """Retrieve the last two chat messages from session history."""
#     history = st.session_state.get("history", [])
#     return history[-2:] if len(history) >= 2 else history

# def generate_answer(question: str, context: str) -> str:
#     """
#     Generate answer with the last two chat history messages included.
    
#     Note: We are not using caching here since the session history is dynamic.
#     """
#     # Get the last two messages from session history
#     last_two = get_last_two_history()
    
#     messages = [
#         {"role": "system", "content": "You are a helpful profesional assistant that answers indepth  questions based on the provided context ,and your own knowledge and conversation history if needed."},
#     ]
    
#     # Append the last two messages from the conversation history
#     for msg in last_two:
#         messages.append({"role": msg["role"], "content": msg["text"]})
    
#     # Append the current question and the context extracted from PDFs
#     messages.append({"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"})
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",
#         "messages": messages,
#         "temperature": 1,
#         "max_tokens": 66000
#     }
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
    
#     try:
#         response = requests.post(
#             url=f"{BASE_URL}/chat/completions",
#             headers=headers,
#             data=json.dumps(payload)
#         )
#         data = response.json()
#         return data["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Error Thinking: {str(e)}"

# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store) -> List[Any]:
#     """Retrieve relevant documents from selected PDFs"""
#     if not selected_pdfs:
#         return []
    
#     filter_dict = {"source": {"$in": selected_pdfs}}
#     return vector_store.similarity_search(
#         query,
#         k=min(4 * len(selected_pdfs), 10),  # Adaptive k based on PDF count
#         filter=filter_dict
#     )

# def main():
#     st.set_page_config(page_title="Multi-PDF Chat", layout="wide")
    
#     # Initialize components
#     vector_store = get_vector_store()
#     pdf_processor = PDFProcessor()
    
#     # Session state initialization
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     with st.sidebar:
#         st.title("PDF Manager")
        
#         # Get and display stored PDFs
#         stored_pdfs = get_stored_pdfs(vector_store)
        
#         # File uploader
#         uploaded_files = st.file_uploader(
#             "Upload PDFs (200MB max per file)",
#             type="pdf",
#             accept_multiple_files=True
#         )
        
#         # Process new uploads
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                     futures = [
#                         executor.submit(pdf_processor.process_pdf, file, vector_store)
#                         for file in uploaded_files
#                         if file.name not in stored_pdfs
#                     ]
#                     concurrent.futures.wait(futures)
            
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
        
#         # Multi-select for PDFs
#         if stored_pdfs:
#             selected_pdfs = st.multiselect(
#                 "Select PDFs to chat with",
#                 options=stored_pdfs,
#                 default=stored_pdfs[0] if stored_pdfs else None
#             )
#         else:
#             selected_pdfs = []
#             st.warning("No PDFs available. Please upload PDF files.")
        
#         st.button("Clear Chat", on_click=lambda: st.session_state.update(history=[]))
    
#     # Main chat interface
#     st.title("Chat with Multiple PDFs")
    
#     if selected_pdfs:
#         st.write(f"Currently chatting with: {', '.join(selected_pdfs)}")
    
#     # Display chat history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     # Chat input
#     user_input = st.chat_input("Ask a question about your selected PDFs...")
    
#     if user_input and selected_pdfs:
#         # Add user message to history
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # Get relevant documents and generate response
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
            
#             with st.spinner("Searching documents..."):
#                 docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#                 context = "\n\n".join(doc.page_content for doc in docs)
            
#             with st.spinner("Thinking..."):
#                 answer = generate_answer(user_input, context)
            
#             # Stream the response
#             full_response = ""
#             for chunk in answer.split('\n'):
#                 full_response += chunk + "\n"
#                 message_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.02)
#             message_placeholder.markdown(full_response)
            
#             # Add assistant response to history
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()


# import os
# import time
# import requests
# import json
# import streamlit as st
# from typing import List, Dict, Any
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# import torch
# from functools import lru_cache
# import concurrent.futures

# # Performance optimization: Disable torch CUDA if not needed
# torch.classes.__path__ = []

# # --- Constants & Config ---
# PDFS_DIRECTORY = 'pdfs'
# DB_DIRECTORY = 'db'
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# MAX_WORKERS = 4  # Adjust based on your CPU cores

# # Ensure directories exist
# os.makedirs(PDFS_DIRECTORY, exist_ok=True)
# os.makedirs(DB_DIRECTORY, exist_ok=True)

# # OpenRouter Configuration
# OPENROUTER_API_KEY = "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"
# BASE_URL = "https://openrouter.ai/api/v1"
# SITE_URL = "http://localhost:8501"
# SITE_NAME = "Multi-PDF Chat App"

# # Initialize embeddings with caching
# @st.cache_resource
# def get_embeddings():
#     return HuggingFaceEmbeddings(
#         model_name="allenai/specter",
#         model_kwargs={'device': 'cpu'}
#     )


# # Initialize vector store with caching
# @st.cache_resource
# def get_vector_store():
#     return Chroma(
#         persist_directory=DB_DIRECTORY,
#         embedding_function=get_embeddings(),
#         collection_name="multi_pdf_store"
#     )

# # Optimized document processing
# class PDFProcessor:
#     def __init__(self):
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP,
#             add_start_index=True
#         )
    
#     @staticmethod
#     def load_pdf(file_path: str) -> List[Any]:
#         loader = PDFPlumberLoader(file_path)
#         return loader.load()
    
#     def process_pdf(self, file, vector_store):
#         try:
#             file_path = os.path.join(PDFS_DIRECTORY, file.name)
#             with open(file_path, "wb") as f:
#                 f.write(file.getbuffer())
            
#             documents = self.load_pdf(file_path)
#             chunks = self.text_splitter.split_documents(documents)
            
#             # Add source metadata
#             for chunk in chunks:
#                 chunk.metadata['source'] = file.name
            
#             vector_store.add_documents(chunks)
#             vector_store.persist()
#             return True
#         except Exception as e:
#             st.error(f"Error processing {file.name}: {str(e)}")
#             return False

# def get_stored_pdfs(vector_store) -> List[str]:
#     """Efficiently retrieve list of stored PDFs"""
#     try:
#         result = vector_store._collection.get(
#             include=['metadatas']
#         )
#         if result and 'metadatas' in result:
#             return list(set(
#                 meta['source'] for meta in result['metadatas'] 
#                 if meta and 'source' in meta
#             ))
#     except Exception as e:
#         st.error(f"Error retrieving stored PDFs: {str(e)}")
#     return []

# @lru_cache(maxsize=100)
# def generate_answer(question: str, context: str) -> str:
#     """Generate answer with caching for repeated questions"""
#     messages = [
#         {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
#         {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
#     ]
    
#     payload = {
#         "model": "google/gemini-2.0-flash-thinking-exp:free",
#         "messages": messages,
#         "temperature": 0.7,
#         "max_tokens": 66000
#     }
    
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#         "HTTP-Referer": SITE_URL,
#         "X-Title": SITE_NAME
#     }
    
#     try:
#         response = requests.post(
#             url=f"{BASE_URL}/chat/completions",
#             headers=headers,
#             data=json.dumps(payload)
#         )
#         data = response.json()
#         return data["choices"][0]["message"]["content"]
#     except Exception as e:
#         return f"Error Thinking: {str(e)}"

# def retrieve_relevant_docs(query: str, selected_pdfs: List[str], vector_store) -> List[Any]:
#     """Retrieve relevant documents from selected PDFs"""
#     if not selected_pdfs:
#         return []
    
#     filter_dict = {"source": {"$in": selected_pdfs}}
#     return vector_store.similarity_search(
#         query,
#         k=min(4 * len(selected_pdfs), 10),  # Adaptive k based on PDF count
#         filter=filter_dict
#     )

# def main():
#     st.set_page_config(page_title="Multi-PDF Chat", layout="wide")
    
#     # Initialize components
#     vector_store = get_vector_store()
#     pdf_processor = PDFProcessor()
    
#     # Session state initialization
#     if 'history' not in st.session_state:
#         st.session_state.history = []
    
#     with st.sidebar:
#         st.title("PDF Manager")
        
#         # Get and display stored PDFs
#         stored_pdfs = get_stored_pdfs(vector_store)
        
#         # File uploader
#         uploaded_files = st.file_uploader(
#             "Upload PDFs (200MB max per file)",
#             type="pdf",
#             accept_multiple_files=True
#         )
        
#         # Process new uploads
#         if uploaded_files:
#             with st.spinner("Processing PDFs..."):
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#                     futures = [
#                         executor.submit(pdf_processor.process_pdf, file, vector_store)
#                         for file in uploaded_files
#                         if file.name not in stored_pdfs
#                     ]
#                     concurrent.futures.wait(futures)
            
#             stored_pdfs = get_stored_pdfs(vector_store)
#             st.success("PDFs processed successfully!")
        
#         # Multi-select for PDFs
#         if stored_pdfs:
#             selected_pdfs = st.multiselect(
#                 "Select PDFs to chat with",
#                 options=stored_pdfs,
#                 default=stored_pdfs[0] if stored_pdfs else None
#             )
#         else:
#             selected_pdfs = []
#             st.warning("No PDFs available. Please upload PDF files.")
        
#         st.button("Clear Chat", on_click=lambda: st.session_state.update(history=[]))
    
#     # Main chat interface
#     st.title("Chat with Multiple PDFs")
    
#     if selected_pdfs:
#         st.write(f"Currently chatting with: {', '.join(selected_pdfs)}")
    
#     # Display chat history
#     for msg in st.session_state.history:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["text"])
    
#     # Chat input
#     user_input = st.chat_input("Ask a question about your selected PDFs...")
    
#     if user_input and selected_pdfs:
#         # Add user message to history
#         st.session_state.history.append({"role": "user", "text": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)
        
#         # Get relevant documents and generate response
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
            
#             with st.spinner("Searching documents..."):
#                 docs = retrieve_relevant_docs(user_input, selected_pdfs, vector_store)
#                 context = "\n\n".join(doc.page_content for doc in docs)
            
#             with st.spinner("Thinking..."):
#                 answer = generate_answer(user_input, context)
            
#             # Stream the response
#             full_response = ""
#             for chunk in answer.split('\n'):
#                 full_response += chunk + "\n"
#                 message_placeholder.markdown(full_response + "▌")
#                 time.sleep(0.02)
#             message_placeholder.markdown(full_response)
            
#             # Add assistant response to history
#             st.session_state.history.append({"role": "assistant", "text": answer})

# if __name__ == "__main__":
#     main()
