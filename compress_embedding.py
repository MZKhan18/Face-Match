import pickle
import gzip

# Load existing embeddings
with open("embedding.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Save compressed version
with gzip.open("embedding.pkl.gz", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ embedding.pkl.gz created successfully")
