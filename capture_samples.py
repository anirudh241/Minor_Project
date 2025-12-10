import cv2
import os
import time

# --- CONFIGURATION ---
name = "Anurag"   # <--- CHANGE THIS for each person
target_images = 20 # Captures 20 images
capture_delay = 1  # 1 second delay between snaps

# Setup paths
dataset_dir = "faces_dataset"
person_dir = os.path.join(dataset_dir, name)

if not os.path.exists(person_dir):
    os.makedirs(person_dir)
    print(f"[INFO] Created folder: {person_dir}")

# Initialize Camera
cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
last_capture_time = time.time()

print(f"[INFO] Starting capture for: {name}")
print("[INFO] Please rotate your head slowly (Left, Right, Tilt, Smile)...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not responding.")
        break
    
    # 1. Detect Face (for visual feedback)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 2. Draw 'Target' Box
    # If a face is found, we draw a Green box. If waiting, it's Yellow.
    for (x,y,w,h) in faces:
        
        # Check if enough time has passed to take a photo
        if time.time() - last_capture_time >= capture_delay:
            count += 1
            filename = f"{person_dir}/img_{int(time.time())}.jpg"
            cv2.imwrite(filename, gray[y:y+h, x:x+w])
            
            print(f"📸 Captured {count}/{target_images}")
            last_capture_time = time.time() # Reset timer
            
            # Visual Flash Effect (White Box)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 255, 255), 4)
        else:
            # Just Waiting (Green Box)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

    # 3. Display Status on Screen
    cv2.putText(frame, f"Captured: {count}/{target_images}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Capture - Rotate Head Slowly", frame)

    # Stop condition
    if count >= target_images:
        print("[INFO] Collection Complete!")
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()