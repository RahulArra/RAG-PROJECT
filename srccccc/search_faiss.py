import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ----------------------------
# Load documents
# ----------------------------

with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)


# ----------------------------
# Load FAISS index
# ----------------------------

index = faiss.read_index("data/faiss.index")


# ----------------------------
# Load embedding model
# ----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------
# User Query
# ----------------------------

query = input("Enter your query: ")


# ----------------------------
# Create Query Embedding
# ----------------------------

query_embedding = model.encode([query]).astype(np.float32)

faiss.normalize_L2(query_embedding)


# ----------------------------
# Search
# ----------------------------

start = time.perf_counter()

scores, indices = index.search(query_embedding, 5)

end = time.perf_counter()


# ----------------------------
# Display Results
# ----------------------------

print("\nTOP 5 RESULTS\n")

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):

    doc = documents[idx]

    print(f"Rank : {rank}")
    print(f"Score: {score:.4f}")
    print(f"Title: {doc['title']}")
    print(f"Body : {doc['body'][:300]}...")
    print("-" * 80)


print(f"\nSearch Time: {(end-start)*1000:.4f} ms")