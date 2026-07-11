from datasets import load_dataset

ds = load_dataset(
    "pacovaldez/stackoverflow-questions",
    split="train"
)

print(ds)
print(ds.column_names)
print(ds[0])