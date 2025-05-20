import streamlit as st
from utils.extraction import extract_text_from_pdf
from utils.processing import process_document, load_embedding_model
from utils.clustering import dimensionality_reduction, perform_clustering, visualize_clusters
from utils.summarization import generate_cluster_summaries
from utils.query import process_query
from utils.processing import init_session_state

# Initialize session state
init_session_state()

st.title("PDF Cluster Summarizer")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
        st.session_state['text'] = text
        st.success("Text extracted!")

if st.session_state['text']:
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            chunks, embeddings, model = process_document(st.session_state['text'])
            st.session_state['chunks'] = chunks
            st.session_state['embeddings'] = embeddings
            st.session_state['model'] = model
            st.success("Document processed!")

if st.session_state['embeddings'] is not None:
    if st.button("Cluster and Visualize"):
        with st.spinner("Clustering..."):
            reduced = dimensionality_reduction(st.session_state['embeddings'])
            clusterer, labels = perform_clustering(st.session_state['embeddings'])
            st.session_state['reduced_embeddings'] = reduced
            st.session_state['cluster_labels'] = labels
            fig = visualize_clusters(reduced, labels, st.session_state['chunks'])
            st.plotly_chart(fig)
            st.success("Clusters visualized!")

if st.session_state['cluster_labels'] is not None:
    if st.button("Summarize Clusters"):
        with st.spinner("Summarizing..."):
            summaries = generate_cluster_summaries(
                st.session_state['chunks'],
                st.session_state['cluster_labels']
            )
            st.session_state['cluster_summaries'] = summaries
            st.success("Summaries generated!")
            for cid, summary in summaries.items():
                st.subheader(f"Cluster {cid}")
                st.write(summary['high'])
                with st.expander("Mid-level summary"):
                    st.write(summary['mid'])
                with st.expander("Detailed summary"):
                    st.write(summary['detailed'])

if st.session_state['cluster_summaries'] is not None:
    query = st.text_input("Ask a question about your document")
    if query:
        context, details = process_query(
            query,
            st.session_state['model'],
            st.session_state['cluster_summaries'],
            st.session_state['cluster_labels'],
            st.session_state['chunks'],
            st.session_state['embeddings']
        )
        st.write("**Relevant Context:**")
        st.write(context)
