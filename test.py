import pickle
import numpy as np
import cv2
import gzip
import gdown
import os
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_FILE = "embedding.pkl.gz"
EMBEDDING_URL = "https://drive.google.com/uc?id=1ZGcKdmKYDhgjPQM9LGkirh0PgZ5uPqbq"

if not os.path.exists(EMBEDDING_FILE):
    print("Downloading face embedding database...")
    gdown.download(EMBEDDING_URL, EMBEDDING_FILE, quiet=False)

with gzip.open("embedding.pkl.gz", "rb") as f:
    feature_list = np.array(pickle.load(f))

filenames = pickle.load(open("filenames.pkl", "rb"))

img_path = "sample/virat young.jpg"

query_embedding = DeepFace.represent(
    img_path=img_path,
    model_name="VGG-Face",
    enforce_detection=False
)[0]["embedding"]

query_embedding = np.array(query_embedding).reshape(1, -1)

similarities = cosine_similarity(query_embedding, feature_list)[0]

# Get best match
best_match_index = np.argmax(similarities)
best_score = similarities[best_match_index]

print("Best similarity score:", best_score)

matched_img = cv2.imread(filenames[best_match_index])

cv2.imshow("Matched Face", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
