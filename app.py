import streamlit as st
import cv2
import pickle
import csv
import pandas as pd
import os
import time
import numpy as np
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
CSV_FILE = "attendance.csv"
DATASET_DIR = "faces_dataset"
CONFIDENCE_THRESHOLD = 70 
COOLDOWN_SECONDS = 60
CONFIRMATION_SECONDS = 3.0 
FRAME_SKIP = 3 

st.set_page_config(page_title="Campus Gate System", page_icon="🎓", layout="wide")

# --- BACKEND FUNCTIONS ---

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Role", "Timestamp"])

def log_entry(name, role):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, role, now_str])

def fetch_logs():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["Name", "Role", "Timestamp"])

def get_next_image_number(person_path):
    if not os.path.exists(person_path): return 1
    existing_files = [f for f in os.listdir(person_path) if f.startswith("img_") and f.endswith(".jpg")]
    if not existing_files: return 1
    numbers = []
    for f in existing_files:
        try:
            num = int(f.split("_")[1].split(".")[0])
            numbers.append(num)
        except:
            pass
    return max(numbers) + 1 if numbers else 1

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    ids = []
    names = {}
    curr_id = 0

    if not os.path.exists(DATASET_DIR): return False

    for name in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, name)
        if not os.path.isdir(path): continue
        
        names[curr_id] = name
        
        for file in os.listdir(path):
            if file.endswith(('.jpg', '.png')):
                img_path = os.path.join(path, file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.equalizeHist(img)
                    face_rects = detector.detectMultiScale(img, 1.1, 5)
                    for (x,y,w,h) in face_rects:
                        faces.append(img[y:y+h, x:x+w])
                        ids.append(curr_id)
                except:
                    continue
        curr_id += 1
    
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer.yml")
        with open("names.pkl", "wb") as f:
            pickle.dump(names, f)
        return True
    return False

# --- FRONTEND ---
st.title("🎓 Campus Gate System")

col1, col2 = st.columns([2, 1])

init_csv()

st.sidebar.title("System Mode")
app_mode = st.sidebar.selectbox("Select Operation", ["Gate Monitor", "Register Personnel"])

# --- MODE 1: REGISTRATION (FIXED) ---
if app_mode == "Register Personnel":
    
    with col1:
        st.subheader("👤 Registration Studio")
        
        # FIX: Use Radio instead of Tabs to prevent variable overwriting
        reg_mode = st.radio("Operation Type", ["Create New Member", "Update Existing Member"], horizontal=True)
        
        final_name = ""
        
        if reg_mode == "Create New Member":
            new_name_input = st.text_input("Enter Full Name", placeholder="e.g. John Doe")
            if new_name_input:
                final_name = new_name_input.strip()

        else: # Update Existing
            if os.path.exists(DATASET_DIR):
                existing_users = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
                if existing_users:
                    final_name = st.selectbox("Select Member", existing_users)
                else:
                    st.warning("No existing members found.")
            else:
                st.warning("Dataset directory missing.")

        FRAME_WINDOW = st.image([])
        
        start_btn = False
        if final_name:
            st.success(f"Selected Target: **{final_name}**")
            start_btn = st.button(f"📸 Start Capture")
        else:
            st.info("Waiting for input...")

    # Capture Logic
    if start_btn and final_name:
        if not os.path.exists(DATASET_DIR): os.makedirs(DATASET_DIR)
        person_path = os.path.join(DATASET_DIR, final_name)
        if not os.path.exists(person_path): os.makedirs(person_path)
        
        start_index = get_next_image_number(person_path)
        end_index = start_index + 20
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        count = 0
        current_img_num = start_index
        
        with col2:
            st.subheader("Status")
            prog_bar = st.progress(0)
            status_txt = st.empty()
        
        while current_img_num < end_index:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                
                if int(time.time() * 10) % 5 == 0: 
                    file_name = f"{person_path}/img_{current_img_num}.jpg"
                    cv2.imwrite(file_name, gray[y:y+h, x:x+w])
                    
                    current_img_num += 1
                    count += 1
                    progress = count / 20
                    prog_bar.progress(progress)
                    status_txt.success(f"Saved: {final_name}/img_{current_img_num-1}.jpg")
                    time.sleep(0.1) 
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
        
        cap.release()
        
        with st.status("Processing...", expanded=True) as status:
            st.write("📂 Files Saved.")
            time.sleep(0.5)
            st.write("🧠 Retraining AI...")
            if train_model():
                status.update(label=f"✅ {final_name} Registered Successfully!", state="complete", expanded=False)
                time.sleep(1)
                st.rerun()
            else:
                status.update(label="❌ Training Failed", state="error")

# --- MODE 2: MONITOR ---
elif app_mode == "Gate Monitor":
    st.sidebar.divider()
    run = st.sidebar.checkbox("Active Monitoring", value=False)
    
    if st.sidebar.button("Clear History"):
        if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
        init_csv()
        if 'logged_people' in st.session_state: st.session_state.logged_people = {}
        st.toast("History Cleared")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        with open("names.pkl", "rb") as f: names = pickle.load(f)
    except:
        st.warning("⚠️ System empty. Please register someone first.")
        st.stop()

    if 'logged_people' not in st.session_state: st.session_state.logged_people = {}
    if 'entry_timers' not in st.session_state: st.session_state.entry_timers = {}
    if 'prediction_buffer' not in st.session_state: st.session_state.prediction_buffer = deque(maxlen=8)

    with col1:
        st.subheader("Live Gate Feed")
        FRAME_WINDOW = st.image([])
    with col2:
        st.subheader("Entry Logs")
        table_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_count = 0
        cached_results = [] 

        while run:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            
            if frame_count % FRAME_SKIP == 0:
                cached_results = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                present_names = set()

                if len(faces) == 0: st.session_state.prediction_buffer.clear()
                
                for (x,y,w,h) in faces:
                    id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    predicted_name = names.get(id_, "Unknown") if confidence < CONFIDENCE_THRESHOLD else "Outsider"
                    
                    st.session_state.prediction_buffer.append(predicted_name)
                    final_name = max(set(st.session_state.prediction_buffer), key=st.session_state.prediction_buffer.count)

                    color = (0, 0, 255)
                    status_text = ""

                    if final_name != "Outsider":
                        present_names.add(final_name)
                        color = (0, 255, 0)
                        
                        current_time = time.time()
                        if final_name not in st.session_state.entry_timers:
                            st.session_state.entry_timers[final_name] = current_time
                        
                        elapsed = current_time - st.session_state.entry_timers[final_name]
                        if elapsed < CONFIRMATION_SECONDS:
                            status_text = f"Verifying... {round(CONFIRMATION_SECONDS - elapsed, 1)}s"
                            color = (0, 255, 255)
                        else:
                            last_log = st.session_state.logged_people.get(final_name, 0)
                            if (current_time - last_log) > COOLDOWN_SECONDS:
                                log_entry(final_name, "Authorized")
                                st.toast(f"✅ Verified: {final_name}")
                                st.session_state.logged_people[final_name] = current_time
                            else:
                                status_text = "Authorized"

                    cached_results.append((x, y, w, h, final_name, color, status_text))

                for name in list(st.session_state.entry_timers.keys()):
                    if name not in present_names: del st.session_state.entry_timers[name]
            
            for (x, y, w, h, name, color, status) in cached_results:
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if status: cv2.putText(frame, status, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)
            
            if frame_count % FRAME_SKIP == 0:
                df = fetch_logs()
                if not df.empty: table_placeholder.dataframe(df.iloc[::-1], height=400, hide_index=True)

        cap.release()
    else:
        df = fetch_logs()
        if not df.empty: table_placeholder.dataframe(df.iloc[::-1], height=400, hide_index=True)