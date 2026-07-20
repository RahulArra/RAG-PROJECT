import os
import pickle
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.append(str(Path(__file__).resolve().parent))

from embedding.embedder import Embedder
from models.document import Document
from utils.html_cleaner import clean_html


class DataPreparer:

    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedder = Embedder(embedding_model)

    def prepare(self, limit):
        ds = load_dataset(
            "pacovaldez/stackoverflow-questions",
            split="train"
        )
        ds = ds.select(range(limit))

        documents = []
        texts = []

        for i, row in enumerate(ds):
            title = row["title"]
            body = clean_html(row["body"])

            document = Document(
                id=i,
                title=title,
                body=body,
                label=row["label"],
                text=title + "\n" + body
            )

            documents.append(document)
            texts.append(document.text)

        print(f"Loaded {len(documents)} documents")

        embeddings = self.embedder.encode(texts)
        embeddings = embeddings.astype(np.float32)

        return documents, embeddings

    def save(self, limit):
        documents, embeddings = self.prepare(limit)

        os.makedirs("data", exist_ok=True)

        np.save("data/embeddings.npy", embeddings)

        with open("data/documents.pkl", "wb") as f:
            pickle.dump(documents, f)

        print("Saved successfully.")


if __name__ == "__main__":
    preparer = DataPreparer()
    preparer.save(100000)
