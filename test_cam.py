from ultralytics import YOLO
import cv2

def run_webcam_detection():
    # Use the smallest YOLO model available
    model = YOLO("yolov8n.pt") 
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set a very low resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Skip a high number of frames
    frame_skip = 5  # Process only every 5th frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
        
        frame_count += 1
        if frame_count % frame_skip == 0:
            # Run YOLO prediction
            results = model.predict(source=frame, show=True, conf=0.5)

            # Print detection details for each individual
            if results and results[0].boxes:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    
                    # Check if the detected object is a 'person' (class ID 0)
                    if model.names[class_id] == 'person':
                        print(f"Detected a person with confidence: {confidence:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_detection()