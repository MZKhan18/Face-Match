import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import re
import pickle
import numpy as np
import gzip
import gdown
import streamlit as st
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

st.set_page_config(
    page_title="Face Match System",
    layout="wide"
)

st.title("🔐 Face Match – Identity Similarity System")
st.write(
    "This application extracts facial embeddings using a deep learning model "
    "and matches them against a reference database. "
    "Such systems can be extended for **identity verification, cyber security, "
    "and criminal face matching** use-cases."
)
EMBEDDING_FILE = "embedding.pkl.gz"
EMBEDDING_URL = "https://drive.google.com/uc?id=1ZGcKdmKYDhgjPQM9LGkirh0PgZ5uPqbq"

if not os.path.exists(EMBEDDING_FILE):
    with st.spinner("Downloading face embedding database..."):
        gdown.download(EMBEDDING_URL, EMBEDDING_FILE, quiet=False)


ACTORS_DIR = "Actors"
ACTORS_ZIP = "Actors.zip"
ACTORS_URL = "https://drive.google.com/uc?id=1PNmNQ4cQ1_kn6p-9XXdVBZba_wRDVj77"

@st.cache_resource
def prepare_actor_images():
    if not os.path.exists(ACTORS_DIR):
        with st.spinner("Downloading reference image database..."):
            gdown.download(ACTORS_URL, ACTORS_ZIP, quiet=False)

        import zipfile
        with zipfile.ZipFile(ACTORS_ZIP, "r") as zip_ref:
            zip_ref.extractall()

        os.remove(ACTORS_ZIP)

prepare_actor_images()

def load_actor_image(actor_dir):
    try:
        for file in os.listdir(actor_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                return Image.open(os.path.join(actor_dir, file))
    except Exception:
        pass
    return None


@st.cache_resource
def load_resources():
    with gzip.open("embedding.pkl.gz", "rb") as f:
        feature_list = np.array(pickle.load(f))

    filenames = pickle.load(open("filenames.pkl", "rb"))
    return feature_list, filenames



feature_list, filenames = load_resources()

uploaded_image = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

def extract_embedding(img_path):
    embedding = DeepFace.represent(
        img_path=img_path,
        model_name="VGG-Face",
        detector_backend="opencv",
        enforce_detection=False
    )[0]["embedding"]

    return np.array(embedding).reshape(1, -1)


def find_best_match(query_embedding, feature_list):
    similarities = cosine_similarity(query_embedding, feature_list)[0]
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    return best_index, best_score

def extract_embedding_from_pil(pil_img):
    embedding = DeepFace.represent(
        img_path=np.array(pil_img),
        model_name="VGG-Face",
        detector_backend="opencv",
        enforce_detection=False
    )[0]["embedding"]

    return np.array(embedding).reshape(1, -1)


if uploaded_image is not None:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        display_img = Image.open(uploaded_image)
        st.image(display_img, width=250)

    with st.spinner("Extracting facial features and matching..."):
        query_embedding = extract_embedding_from_pil(display_img)
        best_index, best_score = find_best_match(query_embedding, feature_list)



    matched_path = filenames[best_index]

# Actor directory & name
    actor_dir = os.path.dirname(matched_path)
    actor_name = os.path.basename(actor_dir)

    with col2:
        st.subheader("Matched Identity")
        st.markdown(f"### 🧑 {actor_name}")

        if os.path.exists(actor_dir):
            actor_img = load_actor_image(actor_dir)
            if actor_img is not None:
                st.image(actor_img, width=250)
            else:
                st.warning("⚠️ No image found in actor folder.")
        else:
            st.warning("⚠️ Actor folder not found.")
            
    st.markdown("---")
    st.subheader("🔎 Match Confidence")

    match_percent = int(best_score * 100)

    st.progress(match_percent)
    st.markdown(f"### {match_percent}% similarity")




