import streamlit as st
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_session_state():
    state_vars = {
        "text": None,
        "chunks": None,
        "embeddings": None,
        "model": None,
        "reduced_embeddings": None,
        "cluster_labels": None,
        "clusterer": None,
        "cluster_summaries": None,
    }
    for var, default in state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def process_document(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = [c for c in text_splitter.split_text(text) if len(c.strip()) > 20]
    model = load_embedding_model()
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(batch)
        all_embeddings.append(batch_embeddings)
    embeddings = np.vstack(all_embeddings)
    return chunks, embeddings, model

def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    try:
        return SentenceTransformer('all-mpnet-base-v2')
    except:
        return SentenceTransformer('all-MiniLM-L6-v2')
