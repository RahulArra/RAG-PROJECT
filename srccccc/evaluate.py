import faiss
import numpy as np
import pickle
import time

from sentence_transformers import SentenceTransformer


with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

embeddings = np.load("data/embeddings.npy").astype("float32")

model = SentenceTransformer("all-MiniLM-L6-v2")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]


# -------------------------
# Exact Index
# -------------------------

exact_index = faiss.IndexFlatIP(dimension)
exact_index.add(embeddings)


# -------------------------
# Load IVF Index
# -------------------------

ivf_index = faiss.read_index("data/ivf.index")


query = input("Enter query: ")

query_embedding = model.encode([query]).astype("float32")

faiss.normalize_L2(query_embedding)


# -------------------------
# Exact Search
# -------------------------

exact_scores, exact_indices = exact_index.search(
    query_embedding,
    5
)


# -------------------------
# IVF Search
# -------------------------

ivf_index.nprobe = 10

ivf_scores, ivf_indices = ivf_index.search(
    query_embedding,
    5
)


print("\nEXACT RESULTS")

print(exact_indices[0])


print("\nIVF RESULTS")

print(ivf_indices[0])


# -------------------------
# Calculate Recall@5
# -------------------------

exact_set = set(exact_indices[0])

ivf_set = set(ivf_indices[0])

matches = len(exact_set.intersection(ivf_set))

recall = matches / 5


print("\nMatches:", matches)

print("Recall@5:", recall)