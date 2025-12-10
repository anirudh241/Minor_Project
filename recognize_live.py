import cv2
import face_recognition
import pickle
import numpy as np
from datetime import datetime
import csv
import os

# Paths
ENCODINGS_PATH = "faces_dataset/encodings.pickle"
ATTENDANCE_CSV = "attendance.csv"

# --- NEW: Frame skipping configuration ---
PROCESS_EVERY_N_FRAMES = 10  # Process only every 5th frame to save resources
frame_counter = 0

# Load encodings
print("[INFO] Loading encodings...")
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

# Initialize webcam
print("[INFO] Starting webcam...")
video_capture = cv2.VideoCapture(1)

# --- NEW: Variables to store the last known locations and names ---
known_face_locations = []
known_face_names = []

# (Your mark_attendance function and CSV check code remains the same here)
# ...
if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    with open(ATTENDANCE_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, date_str, time_str])

previous_frame_names = set()
# ...

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_counter += 1

    # --- CHANGE 1: Only run face recognition on designated frames ---
    if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
        # Reset the lists for the new processing frame
        known_face_locations = []
        known_face_names = []
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces and encodings
        boxes = face_recognition.face_locations(rgb_small_frame)
        encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

        current_frame_names = set()
        for encoding, box in zip(encodings, boxes):
            face_distances = face_recognition.face_distance(data["encodings"], encoding)

            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.5:
                    name = data["names"][best_match_index]
            
            # Store the results for this processed frame
            known_face_locations.append(box)
            known_face_names.append(name)
            
            if name != "Unknown":
                current_frame_names.add(name)
        
        # Handle attendance logging for newly appeared people
        newly_appeared_names = current_frame_names - previous_frame_names
        for name in newly_appeared_names:
            mark_attendance(name)
            print(f"[ATTENDANCE] {name} entered the frame and has been marked present.")
        previous_frame_names = current_frame_names

    # --- CHANGE 2: Draw the results from the LAST processed frame on EVERY frame ---
    for (top, right, bottom, left), name in zip(known_face_locations, known_face_names):
        # Scale back box locations
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()