import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

embeddings = np.load("data/embeddings.npy").astype("float32")

model = SentenceTransformer("all-MiniLM-L6-v2")


# Normalize document vectors
faiss.normalize_L2(embeddings)

# Dimension = 384
dimension = embeddings.shape[1]

# Exact search using inner product
index = faiss.IndexFlatIP(dimension)

# Add all vectors to FAISS
index.add(embeddings)

print("Vectors in index:", index.ntotal)


query = input("Enter your query: ")

query_embedding = model.encode([query]).astype("float32")

# Normalize query vector
faiss.normalize_L2(query_embedding)


start = time.perf_counter()

scores, indices = index.search(query_embedding, 5)

end = time.perf_counter()


print("\nTOP 5 RESULTS:\n")

for score, i in zip(scores[0], indices[0]):
    print("Score:", score)
    print("Document:", documents[i][:200])
    print("-" * 50)


print(f"\nSearch time: {(end - start) * 1000:.4f} ms")