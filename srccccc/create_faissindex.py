import faiss
import numpy as np

# Load embeddings
embeddings = np.load("data/embeddings.npy").astype("float32")

# Normalize
faiss.normalize_L2(embeddings)

# Create index
index = faiss.IndexFlatIP(embeddings.shape[1])

# Add vectors
index.add(embeddings)

# Save index
faiss.write_index(index, "data/faiss.index")

print("FAISS index created successfully!")
print("Vectors:", index.ntotal)