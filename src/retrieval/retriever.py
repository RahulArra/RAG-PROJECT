import pickle
import numpy as np

from embedding.embedder import Embedder
from retrieval.faiss_index import FaissIndex


class Retriever:

    def __init__(self):

        self.embedder = Embedder()

        self.index = FaissIndex.load(
            "data/ivf.index"
        )

        with open(
            "data/documents.pkl",
            "rb"
        ) as f:

            self.documents = pickle.load(f)

    def retrieve(
        self,
        query,
        k=20
    ):

        query_embedding = self.embedder.embed_query(
            query
        )

        scores, ids = self.index.search(
            query_embedding,
            k
        )

        docs = []

        for i in ids[0]:

            docs.append(
                self.documents[i]
            )

        return docs