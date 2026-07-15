from datasets import load_dataset

class Loader:

    def load(self, n):

        ds = load_dataset(
            "pacovaldez/stackoverflow-questions",
            split="train"
        )

        return ds.select(range(n))