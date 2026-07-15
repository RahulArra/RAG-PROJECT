from retrieval.retriever import Retriever
from retrieval.reranker import Reranker


retriever = Retriever()

reranker = Reranker()


query = input("Query : ")


docs = retriever.retrieve(
    query,
    k=20
)


print("=" * 60)
print("FAISS RESULTS")
print("=" * 60)

for i, doc in enumerate(docs):

    print(i + 1)

    print(doc[:200])

    print("-" * 60)


reranked = reranker.rerank(
    query,
    docs
)


print()

print("=" * 60)
print("RERANKED RESULTS")
print("=" * 60)


for score, doc in reranked:

    print(score)

    print(doc[:200])

    print("-" * 60)