import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from generation.prompt_builder import PromptBuilder
from retrieval.reranker import Reranker


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
reranker = Reranker()
prompt_builder = PromptBuilder()


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

scores, indices = index.search(query_embedding, 20)

candidate_docs = []

for idx in indices[0]:
    candidate_docs.append(documents[idx])

end = time.perf_counter()


# ----------------------------
# Display FAISS Results
# ----------------------------

print("=" * 80)
print("FAISS RESULTS")
print("=" * 80)

for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):

    doc = documents[idx]

    print(f"Rank : {rank}")
    print(f"Score: {score:.4f}")
    print(f"Title: {doc.title}")
    print("-" * 80)


# ----------------------------
# Rerank
# ----------------------------

reranked = reranker.rerank(query, candidate_docs)

print()
print("=" * 80)
print("RERANKED RESULTS")
print("=" * 80)

for rank, (score, doc) in enumerate(reranked[:5], start=1):

    print(f"Rank : {rank}")
    print(f"Score: {score:.4f}")
    print(f"Title: {doc.title}")
    print("-" * 80)

print()
print("=" * 80)
print("PROMPT")
print("=" * 80)

prompt = prompt_builder.build(
    query,
    [doc for _, doc in reranked[:3]]
)

print(prompt)

print(f"\nSearch Time: {(end-start)*1000:.4f} ms")