import cv2
import os
import numpy as np
import pickle

# --- OPTIMIZATION: Radius=1, Neighbors=8 (Better for webcam) ---
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

image_paths = []
face_samples = []
ids = []
name_map = {} 

current_id = 0
dataset_path = "faces_dataset"

print("[INFO] Training started...")

for name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, name)
    if not os.path.isdir(person_path): continue
    
    name_map[current_id] = name
    print(f"[INFO] Processing {name} (ID: {current_id})")
    
    for filename in os.listdir(person_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(person_path, filename)
            
            # 1. Read as Gray
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 2. OPTIMIZATION: Fix Lighting
            img = cv2.equalizeHist(img) 

            # Detect face
            faces = detector.detectMultiScale(img)
            
            for (x,y,w,h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(current_id)
                
    current_id += 1

print(f"[INFO] Training on {len(face_samples)} faces...")
recognizer.train(face_samples, np.array(ids))
recognizer.save("trainer.yml")

with open("names.pkl", "wb") as f:
    pickle.dump(name_map, f)

print("[INFO] Done. Now run 'streamlit run app.py'")