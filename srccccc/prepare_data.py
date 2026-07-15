from datasets import load_dataset
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import os


class DataPreparer:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def clean_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(" ", strip=True)

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

            body = self.clean_html(row["body"])

            document = {
                "id": i,
                "title": title,
                "body": body,
                "label": row["label"],      # <-- keep original metadata
                "text": title + "\n" + body
            }

            documents.append(document)
            texts.append(document["text"])

        print(f"Loaded {len(documents)} documents")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        embeddings = embeddings.astype(np.float32)

        return documents, embeddings


if __name__ == "__main__":

    preparer = DataPreparer()

    documents, embeddings = preparer.prepare(100000)

    os.makedirs("data", exist_ok=True)

    np.save("data/embeddings.npy", embeddings)

    with open("data/documents.pkl", "wb") as f:
        pickle.dump(documents, f)

    print("Saved successfully.")