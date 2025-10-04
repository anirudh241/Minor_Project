# evaluate_lfw.py (Final Version with Corrected Nested Path)
import face_recognition
import os
import time
from tqdm import tqdm

# --- Configuration (Path corrected for the extra nested folder) ---
LFW_DATASET_DIR = "lfw_dataset"
LFW_IMAGES_DIR = os.path.join(LFW_DATASET_DIR, "lfw-deepfunneled", "lfw-deepfunneled") # <-- THE FIX IS HERE
LFW_PAIRS_FILE = os.path.join(LFW_DATASET_DIR, "pairs.csv")
RECOGNITION_THRESHOLD = 0.5

# --- Helper function to get image paths ---
def get_image_path(person, img_num):
    """Constructs the full path for a given LFW image."""
    img_num_str = str(img_num).zfill(4)
    img_filename = f"{person}_{img_num_str}.jpg"
    return os.path.join(LFW_IMAGES_DIR, person, img_filename)

# --- Main Logic ---
print("[INFO] Starting LFW evaluation...")

# Step 1: Parse the pairs file into a temporary list
all_pairs = []
with open(LFW_PAIRS_FILE, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        parts = [part for part in line.strip().split(',') if part]
        all_pairs.append(parts)

# Step 2: Pre-flight check to find only valid pairs that exist on disk
print(f"[INFO] Found {len(all_pairs)} pairs in csv. Verifying which image files exist...")
valid_image_pairs = []
is_same_person_list = []

for parts in tqdm(all_pairs, desc="Verifying file paths"):
    path1, path2 = None, None
    is_same = False

    if len(parts) == 3:  # Same person pair
        person, img1_num, img2_num = parts[0], int(parts[1]), int(parts[2])
        path1 = get_image_path(person, img1_num)
        path2 = get_image_path(person, img2_num)
        is_same = True
    elif len(parts) == 4:  # Different person pair
        person1, img1_num, person2, img2_num = parts[0], int(parts[1]), parts[2], int(parts[3])
        path1 = get_image_path(person1, img1_num)
        path2 = get_image_path(person2, img2_num)
        is_same = False

    if path1 and path2 and os.path.exists(path1) and os.path.exists(path2):
        valid_image_pairs.append((path1, path2))
        is_same_person_list.append(is_same)

print(f"[INFO] Found {len(valid_image_pairs)} valid pairs to evaluate.")

# Step 3: Perform the evaluation only on the valid pairs
correct_predictions = 0
start_time = time.time()

for i in tqdm(range(len(valid_image_pairs)), desc="Evaluating Valid Pairs"):
    path1, path2 = valid_image_pairs[i]
    is_same = is_same_person_list[i]

    try:
        img1 = face_recognition.load_image_file(path1)
        encodings1 = face_recognition.face_encodings(img1)

        img2 = face_recognition.load_image_file(path2)
        encodings2 = face_recognition.face_encodings(img2)

        if len(encodings1) > 0 and len(encodings2) > 0:
            encoding1, encoding2 = encodings1[0], encodings2[0]
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            prediction_is_same = distance < RECOGNITION_THRESHOLD

            if prediction_is_same == is_same:
                correct_predictions += 1
    except Exception as e:
        print(f"\n[ERROR] An error occurred processing pair {path1}, {path2}: {e}")

end_time = time.time()

# --- Calculate and print results ---
total_pairs_processed = len(valid_image_pairs)
accuracy = (correct_predictions / total_pairs_processed) * 100 if total_pairs_processed > 0 else 0
duration = end_time - start_time

print("\n" + "="*30)
print("      EVALUATION COMPLETE")
print("="*30)
print(f"Total pairs evaluated: {total_pairs_processed}")
print(f"Correct predictions:   {correct_predictions}")
print(f"Accuracy:              {accuracy:.2f}%")
print(f"Total time taken:      {duration:.2f} seconds")
print("="*30)