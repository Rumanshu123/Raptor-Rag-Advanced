# Raptor-Rag-Advanced

# RAPTOR RAG PDF Cluster Summarizer

A powerful, interactive app to **analyze, cluster, summarize, and query any PDF document** using state-of-the-art Retrieval-Augmented Generation (RAG) techniques, inspired by the RAPTOR approach.

---

## Table of Contents

- [Introduction](#introduction)
- [How It Works](#how-it-works)
- [Project Architecture](#project-architecture)
- [Key Concepts Explained](#key-concepts-explained)
- [Models and Algorithms Used](#models-and-algorithms-used)
- [Benefits](#benefits)
- [Limitations & Cons](#limitations--cons)
- [How to Run](#how-to-run)
- [Example Use Cases](#example-use-cases)
- [References](#references)

---

## Introduction

This project enables you to upload any PDF (e.g., books, reports, research papers), and interactively:
- **Extract text** from the PDF
- **Split the text into meaningful chunks**
- **Embed and cluster** similar chunks using vector embeddings
- **Summarize each cluster** at multiple levels (high, mid, detailed)
- **Visualize document structure** as clusters
- **Ask questions** and get context-aware answers using Retrieval-Augmented Generation (RAG) with LLMs

It is inspired by the RAPTOR approach, which builds a hierarchical tree of summaries and clusters to enable efficient, multi-level retrieval and reasoning over large documents[[2]].

---

## How It Works

1. **PDF Upload & Extraction:**  
   Extracts all text from your PDF using `pypdf`.

2. **Text Chunking:**  
   Splits the text into manageable chunks (e.g., paragraphs or sections) for better processing.

3. **Embedding:**  
   Converts each chunk into a vector using a Sentence Transformer model (`all-MiniLM-L6-v2` for speed and efficiency).

4. **Clustering:**  
   Groups similar chunks using HDBSCAN or KMeans, revealing thematic structure in the document.

5. **Summarization:**  
   For each cluster, generates summaries at various levels (high, mid, detailed) using an LLM (e.g., OpenAI GPT-3.5/4).

6. **Visualization:**  
   Shows clusters on a 2D interactive scatter plot for easy exploration.

7. **Querying (RAG):**  
   When you ask a question, the app retrieves the most relevant clusters and chunks, assembles a context, and generates an answer using the LLM—grounded in your document.

---

## Project Architecture


- **Streamlit**: User interface and workflow orchestration
- **pypdf**: PDF text extraction
- **LangChain**: Text chunking and processing
- **Sentence Transformers**: Embedding model for semantic understanding
- **HDBSCAN/KMeans**: Clustering algorithm for grouping similar content
- **OpenAI GPT-3.5/4**: Summarization and question answering
- **Plotly**: Visualization of clusters

---

## Key Concepts Explained

### **Retrieval-Augmented Generation (RAG)**
Combines document retrieval with LLM-based answer generation. Instead of relying only on the LLM's memory, it fetches relevant document chunks and uses them as context for more accurate, grounded answers.

### **RAPTOR Approach**
A recursive, hierarchical RAG method:
- **Clusters and summarizes** chunks at multiple levels, forming a tree structure.
- Enables retrieval at different abstraction levels, improving both broad and specific queries[[2]][[6]].

### **Chunking**
Breaks long document text into smaller, overlapping pieces to make processing and retrieval more efficient.

### **Embedding**
Transforms text into high-dimensional vectors so that similar content is close together in vector space.

### **Clustering**
Groups similar chunks, revealing topics/themes in the document.

### **Summarization**
Uses an LLM to generate concise summaries for each cluster, at different levels of detail.

### **Visualization**
Displays clusters/topics as colored dots on a 2D plot, helping users explore the document's structure.

### **Querying**
Lets users ask questions; the app retrieves the most relevant clusters/chunks and generates a context-aware answer.

---

## Models and Algorithms Used

- **Embedding Model:**  
  `all-MiniLM-L6-v2` (Sentence Transformers)  
  *Chosen for its balance of speed and accuracy, making it suitable for interactive apps and large documents.*

- **Clustering Algorithm:**  
  `HDBSCAN` (preferred) or `KMeans`  
  *HDBSCAN can find clusters of varying sizes and ignore noise, ideal for real-world documents.*

- **Summarization & QA Model:**  
  `OpenAI GPT-3.5` or `GPT-4`  
  *Used for generating summaries and answering questions based on retrieved context.*

- **Visualization:**  
  `Plotly` for interactive 2D scatter plots.

---

## Benefits

- **Handles Large, Complex Documents:**  
  Efficiently processes books, reports, and research papers.

- **Multi-level Summarization:**  
  Provides both high-level overviews and detailed breakdowns of document topics.

- **Context-aware Q&A:**  
  Answers are grounded in your document, not just the LLM's training data.

- **Interactive Visualization:**  
  Explore document structure and topics visually.

- **Extensible & Modular:**  
  Swap in different models or algorithms as needed.

---

## Limitations & Cons

- **Requires API Access:**  
  Needs an OpenAI API key for summarization and Q&A.

- **Processing Time:**  
  Large PDFs may take time to process, especially on CPU and with many clusters/summaries.

- **Cost:**  
  Each summarization or Q&A call to OpenAI may incur API costs.

- **Model Limitations:**  
  Summarization and answers are only as good as the underlying LLM and embeddings.

- **No Image/Table Extraction:**  
  Current version focuses on text; images/tables are not processed.

---


