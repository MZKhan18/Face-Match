import os
import gzip
import pickle
import numpy as np
import gdown
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

EMBEDDING_FILE = "embedding.pkl.gz"
EMBEDDING_URL = "https://drive.google.com/uc?id=1ZGcKdmKYDhgjPQM9LGkirh0PgZ5uPqbq"

if not os.path.exists(EMBEDDING_FILE):
    print("Downloading face embedding database...")
    gdown.download(EMBEDDING_URL, EMBEDDING_FILE, quiet=False)

with gzip.open(EMBEDDING_FILE, "rb") as f:
    feature_list = np.array(pickle.load(f))

filenames = pickle.load(open("filenames.pkl", "rb"))

img_path = "sample/ranbir.jpg"

query_embedding = DeepFace.represent(
    img_path=img_path,
    model_name="VGG-Face",
    detector_backend="opencv",
    enforce_detection=False
)[0]["embedding"]

query_embedding = np.array(query_embedding).reshape(1, -1)

similarities = cosine_similarity(query_embedding, feature_list)[0]
best_index = np.argmax(similarities)
best_score = similarities[best_index]

print(f"Best similarity score: {best_score:.4f}")
print("Matched file:", filenames[best_index])

matched_img = Image.open(filenames[best_index])
matched_img.show()
