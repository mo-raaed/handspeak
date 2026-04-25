import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)

DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "asl_alphabet_train", "asl_alphabet_train")
OUTPUT_CSV = os.path.join(_PROJECT_ROOT, "data", "landmarks.csv")
MP_TASK_PATH = os.path.join(_PROJECT_ROOT, "models", "hand_landmarker.task")

def extract_landmarks():
    print("Initializing MediaPipe Hand Landmarker...")
    base_options = python.BaseOptions(model_asset_path=MP_TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    # Prepare CSV header
    header = ["label"]
    for i in range(21):
        header.extend([f"x_{i}", f"y_{i}", f"z_{i}"])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    
    total_processed = 0
    total_saved = 0

    print(f"Starting extraction to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_name in tqdm(classes, desc="Processing Classes"):
            class_dir = os.path.join(DATASET_PATH, class_name)
            images = [img for img in os.listdir(class_dir) if img.endswith(".jpg") or img.endswith(".png")]
            
            for img_name in tqdm(images, desc=class_name, leave=False):
                img_path = os.path.join(class_dir, img_name)
                
                try:
                    mp_image = mp.Image.create_from_file(img_path)
                    detection_result = detector.detect(mp_image)
                    
                    if detection_result.hand_landmarks:
                        # Only take the first hand detected
                        hand = detection_result.hand_landmarks[0]
                        row = [class_name]
                        for lm in hand:
                            row.extend([lm.x, lm.y, lm.z])
                        writer.writerow(row)
                        total_saved += 1
                        
                    total_processed += 1
                except Exception as e:
                    # Some images might be corrupt or unreadable
                    pass

    print(f"\nExtraction complete!")
    print(f"Total images processed: {total_processed}")
    print(f"Total landmarks saved: {total_saved} ({(total_saved/total_processed)*100:.1f}%)")

if __name__ == "__main__":
    extract_landmarks()
