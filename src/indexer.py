import faiss
import numpy as np

class CatalogIndexer:
    def __init__(self, embedding_dim=512):
        # IndexFlatIP maps directly to Cosine Similarity when vectors are normalized
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.product_ids = []

    def add_to_index(self, product_id, embedding):
        vector = np.array([embedding]).astype('float32')
        self.index.add(vector)
        self.product_ids.append(product_id)

    def search(self, query_embedding, top_k=5):
        vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1: # Valid match
                results.append({
                    "product_id": self.product_ids[idx],
                    "score": float(dist)
                })
        return results