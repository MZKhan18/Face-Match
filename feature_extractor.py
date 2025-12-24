# import os
# import pickle
#
# actors = os.listdir('Actors')
#
# filenames = []
#
# for actor in actors:
#     for file in os.listdir(os.path.join('Actors',actor)):
#         filenames.append(os.path.join('Actors',actor,file))
#
# pickle.dump(filenames,open('filenames.pkl','wb'))

from deepface import DeepFace
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

features = []

for file in tqdm(filenames):
    embedding = DeepFace.represent(
        img_path=file,
        model_name="VGG-Face",
        enforce_detection=False
    )[0]["embedding"]
    features.append(embedding)

pickle.dump(features, open("embedding.pkl", "wb"))
