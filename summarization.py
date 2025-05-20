import streamlit as st
import time

def summarize_cluster(cluster_texts, level="detailed", model_name="gpt-3.5-turbo"):
    from openai import OpenAI
    client = OpenAI(api_key=st.session_state.get("OPENAI_API_KEY", ""))
    combined_text = "\n\n".join(cluster_texts)[:12000]
    prompt = f"Summarize the following text ({level}):\n{combined_text}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500 if level == "detailed" else 250
    )
    return response.choices[0].message.content

def generate_cluster_summaries(chunks, cluster_labels, summary_model="gpt-3.5-turbo"):
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(chunks[i].strip())
    all_summaries = {}
    for cid, texts in clusters.items():
        if len(texts) < 2:
            continue
        all_summaries[cid] = {
            "high": summarize_cluster(texts, "high", summary_model),
            "mid": summarize_cluster(texts, "mid", summary_model),
            "detailed": summarize_cluster(texts, "detailed", summary_model),
            "raw_chunks": texts
        }
    return all_summaries
