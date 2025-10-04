import cv2
from deepface import DeepFace
import os
import logging
import time

# Disable DeepFace's info and debug logs for cleaner output
logging.getLogger("deepface").setLevel(logging.ERROR)

def run_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables to manage the display time
    recognized_person = None
    last_recognition_time = 0
    display_duration = 3  # Display name for 3 seconds
    
    # Counter for the number of recognitions
    recognition_count = 0
    max_recognitions = 1000

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        # Display the live video feed
        cv2.imshow('Live Webcam Feed', frame)

        current_time = time.time()

        # Only run recognition if a person is not currently being displayed
        # or if the display duration has passed
        if recognized_person is None or (current_time - last_recognition_time) > display_duration:
            try:
                results = DeepFace.find(img_path=frame, db_path="my_database", enforce_detection=False)
                
                if isinstance(results, list) and results:
                    for result in results:
                        if not result.empty:
                            # Found a new face, update the recognized_person and time
                            identity = os.path.basename(os.path.dirname(result.iloc[0]['identity']))
                            recognized_person = identity
                            last_recognition_time = current_time
                            print(f"Person 1: Name({recognized_person})")
                            
                            # Increment the recognition count
                            recognition_count += 1
                            if recognition_count >= max_recognitions:
                                print(f"Reached {max_recognitions} recognitions. Stopping.")
                                break
                        else:
                            recognized_person = None
                            
                else:
                    recognized_person = None

            except Exception as e:
                recognized_person = None

        if recognition_count >= max_recognitions or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()