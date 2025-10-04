# encode_faces.py
import face_recognition
import pickle
import cv2
import os

dataset_dir = "faces_dataset"
encodings_file = os.path.join(dataset_dir, "encodings.pickle")

knownEncodings = []
knownNames = []

print("[INFO] Starting encoding process...")

# loop over each person in dataset
for person_name in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] Processing {person_name}...")

    # loop over images of the person
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect faces
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(person_name)

# save encodings
data = {"encodings": knownEncodings, "names": knownNames}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encoding complete. Encodings saved to {encodings_file}")