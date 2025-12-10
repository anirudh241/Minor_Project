import streamlit as st
import cv2
import pickle

st.title("Campus Gate (LBPH)")
run = st.checkbox("Open Camera")
FRAME_WINDOW = st.image([])

# Load Model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load Names
with open("names.pkl", "rb") as f:
    names = pickle.load(f)

if run:
    cap = cv2.VideoCapture(0)
while run:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optimize detection: scaleFactor 1.2 -> 1.1 (More accurate)
        gray = cv2.equalizeHist(gray)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        
        for (x,y,w,h) in faces:
            # Recognize
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # ROUND the confidence to make it readable
            score = round(confidence)

            # --- TUNING ZONE ---
            # LBPH "Confidence" is actually DISTANCE. Lower is better.
            # 0 = Exact Clone. 
            # < 50 = Very Good Match.
            # < 80 = Acceptable Match (Try changing this first!)
            if confidence < 75: 
                name = names[id_]
                color = (0, 255, 0) # Green
            else:
                name = "Outsider"
                color = (0, 0, 255) # Red
                
            # Draw Box
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            
            # Show Name AND Score (Critical for debugging)
            text = f"{name} ({score})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)