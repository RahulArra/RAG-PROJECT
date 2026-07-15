from datasets import load_dataset
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


ds = load_dataset(
    "pacovaldez/stackoverflow-questions",
    split="train"
)

ds = ds.select(range(100000))


def clean_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(" ", strip=True)


documents = []

for row in ds:
    title = row["title"]
    body = clean_html(row["body"])

    document = title + "\n" + body
    documents.append(document)


print("Number of documents:", len(documents))


model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    documents,
    show_progress_bar=True
)

print("Embeddings shape:", embeddings.shape)

print("First 10 values of first embedding:")
print(embeddings[0][:10])



import os
import pickle
import numpy as np

os.makedirs("data", exist_ok=True)

np.save("data/embeddings.npy", embeddings)

with open("data/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("Documents and embeddings saved successfully!")