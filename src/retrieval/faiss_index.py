import faiss
import numpy as np


class FaissIndex:

    def __init__(self, index):

        self.index = index

    @classmethod
    def load(
        cls,
        path
    ):

        index = faiss.read_index(path)

        return cls(index)

    def search(
        self,
        query_embedding,
        k
    ):

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        faiss.normalize_L2(
            query_embedding
        )

        return self.index.search(
            query_embedding,
            k
        )