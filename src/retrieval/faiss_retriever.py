import pickle
from pathlib import Path

import faiss
import numpy as np

from models.retrieval_result import RetrievalResult


class FAISSRetriever:

    def __init__(
        self,
        index_path="data/faiss.index",
        documents_path="data/documents.pkl"
    ):
        self.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as f:
            self.documents = pickle.load(f)

    def retrieve(self, query_embedding, k=20):
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            doc_data = self.documents[idx]
            if isinstance(doc_data, dict):
                from models.document import Document
                doc = Document(
                    id=doc_data.get('id', idx),
                    title=doc_data.get('title', ''),
                    body=doc_data.get('body', ''),
                    label=doc_data.get('label', 0),
                    text=doc_data.get('text', '')
                )
            else:
                doc = doc_data

            results.append(
                RetrievalResult(
                    document=doc,
                    score=float(score)
                )
            )

        return results