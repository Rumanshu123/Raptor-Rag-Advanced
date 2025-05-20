import numpy as np
import pandas as pd
import plotly.express as px

def dimensionality_reduction(embeddings):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)

def perform_clustering(embeddings, min_cluster_size=5):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    return clusterer, labels

def visualize_clusters(reduced_embeddings, cluster_labels, chunks):
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'cluster': cluster_labels,
        'text': [chunk[:50] + "..." for chunk in chunks]
    })
    df['cluster'] = df['cluster'].astype(str)
    fig = px.scatter(
        df, x='x', y='y', color='cluster',
        hover_data=['text'],
        title='Document Clusters Visualization'
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
