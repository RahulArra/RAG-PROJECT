import pickle
import time

import numpy as np
from sentence_transformers import SentenceTransformer


with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

embeddings = np.load("data/embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")


query = input("Enter your query: ")

query_embedding = model.encode(query)


start = time.perf_counter()

scores = np.dot(embeddings, query_embedding) / (
    np.linalg.norm(embeddings, axis=1) *
    np.linalg.norm(query_embedding)
)

top_indices = np.argsort(scores)[::-1][:5]

end = time.perf_counter()


print("\nTOP 5 RESULTS:\n")

for i in top_indices:
    print("Score:", scores[i])
    print("Document:", documents[i][:200])
    print("-" * 50)


print(f"\nSearch time: {(end - start) * 1000:.4f} ms")