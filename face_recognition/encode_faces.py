import os
import face_recognition
import pickle

known_encodings = []
known_names = []

base_dir = "dataset"
for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if not os.path.isdir(person_dir):
        continue
    for file in os.listdir(person_dir):
        path = os.path.join(person_dir, file)
        img = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

os.makedirs("models", exist_ok=True)
with open("models/encodings.pickle", "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("Encodings saved to models/encodings.pickle")