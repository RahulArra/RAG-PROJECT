import pickle

with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

print(type(documents))
print(len(documents))

print("\nFirst document:\n")
print(documents[0])