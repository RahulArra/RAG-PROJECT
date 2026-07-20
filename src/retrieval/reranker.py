from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(self, query, retrieval_results):

        pairs = [
            (query, res.document.text)
            for res in retrieval_results
        ]

        scores = self.model.predict(pairs)

        for res, score in zip(retrieval_results, scores):
            res.score = float(score)

        ranked = sorted(
            retrieval_results,
            key=lambda x: x.score,
            reverse=True
        )

        return ranked