import numpy as np

def process_query(query, model, cluster_summaries, cluster_labels, chunks, embeddings, top_k=10, summary_level="mid"):
    """
    Process a user query to find relevant context for RAG.
    """
    # Embed the query
    query_embedding = model.encode([query])[0]

    # Compute cosine similarity with all chunks
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top-k most similar chunk indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Find clusters for these chunks
    relevant_clusters = {}
    for idx in top_indices:
        cluster_id = cluster_labels[idx]
        if cluster_id == -1:  # Skip noise
            continue
        if cluster_id not in relevant_clusters:
            relevant_clusters[cluster_id] = []
        relevant_clusters[cluster_id].append((idx, similarities[idx]))

    # If no clusters, use the top chunks directly
    if not relevant_clusters:
        context = "\n\n".join([f"CHUNK {i+1}: {chunks[idx]}" for i, idx in enumerate(top_indices)])
        return context, []

    # Collect summaries and key chunks from relevant clusters
    context_parts = []
    used_summaries = []

    # Sort clusters by max similarity in top_indices
    sorted_clusters = sorted(
        relevant_clusters.items(),
        key=lambda x: max(sim for _, sim in x[1]),
        reverse=True
    )

    for cluster_id, cluster_info in sorted_clusters:
        if cluster_id not in cluster_summaries:
            continue
        # Add the chosen level summary
        summary = cluster_summaries[cluster_id][summary_level]
        context_parts.append(f"CLUSTER {cluster_id} {summary_level.upper()} SUMMARY: {summary}")
        used_summaries.append({"cluster": cluster_id, "summary": summary})

        # Add top 2 chunks from this cluster
        cluster_chunk_indices = [idx for idx, _ in cluster_info]
        cluster_similarities = [similarities[idx] for idx in cluster_chunk_indices]
        sorted_indices = np.argsort(cluster_similarities)[-2:][::-1]
        top_cluster_chunks = [chunks[cluster_chunk_indices[i]] for i in sorted_indices]

        for chunk in top_cluster_chunks:
            context_parts.append(f"DETAIL FROM CLUSTER {cluster_id}: {chunk}")

    context = "\n\n".join(context_parts)
    return context, used_summaries

def generate_answer(query, context, api_key, model="gpt-4-turbo"):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are a helpful assistant that answers questions based on the provided context.

    CONTEXT:
    {context}

    QUESTION: {query}

    Answer the question based ONLY on the provided context. If the context doesn't contain 
    enough information to answer the question fully, say so. Provide a comprehensive and 
    accurate answer.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a document question-answering assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content
