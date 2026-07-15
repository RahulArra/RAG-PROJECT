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

dimensions = embeddings.shape[1]

nlist = 100

quantizer = faiss.IndexFlatIP(dimensions)

index = faiss.IndexIVFFlat(
    quantizer,
    dimensions,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

print("TRAINING IVF INDEX")

index.train(embeddings)

print("adding vectors..")

index.add(embeddings)
faiss.write_index(index, "data/ivf.index")
index.nprobe=10

query = input("enter your query")

query_embedding = model.encode([query]).astype("float32")

faiss.normalize_L2(query_embedding)

start = time.perf_counter()

scores,indices = index.search(query_embedding,5)

end = time.perf_counter()
print("\nTOP 5 RESULTS:\n")

for score, i in zip(scores[0], indices[0]):

    print("Score:", score)
    print("Document:", documents[i][:200])
    print("-" * 50)


print(f"\nSearch time: {(end - start) * 1000:.4f} ms")