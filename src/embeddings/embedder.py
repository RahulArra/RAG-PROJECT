from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def embed_query(
        self,
        query
    ):

        return self.model.encode(
            query
        )

    def embed_documents(
        self,
        docs
    ):

        return self.model.encode(
            docs,
            show_progress_bar=True
        )