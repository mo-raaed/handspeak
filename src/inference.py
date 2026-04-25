import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import pickle
from gesture_map import get_word

# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "landmark_model.pth")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the tiny Feed-Forward Neural Network to match training
class LandmarkNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def get_bounding_box(hand_landmarks, width, height):
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
        
    pad = 20
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(width, x_max + pad)
    y_max = min(height, y_max + pad)
    return x_min, y_min, x_max, y_max

def main():
    print("Loading label encoder...")
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    print("Loading Landmark Neural Network...")
    model = LandmarkNN(input_size=63, num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    print("Loading MediaPipe tasks...")
    mp_task_path = os.path.join(MODELS_DIR, "hand_landmarker.task")
    if not os.path.exists(mp_task_path):
        print("Downloading hand_landmarker.task...")
        os.makedirs(os.path.dirname(mp_task_path), exist_ok=True)
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", mp_task_path)

    base_options = python.BaseOptions(model_asset_path=mp_task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    # Utility for drawing the skeleton
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror
        h, w, c = frame.shape
        
        # Convert BGR to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                # Extract the 63 coordinates
                coords = []
                for lm in hand_landmarks:
                    coords.extend([lm.x, lm.y, lm.z])
                
                # Convert to tensor and pass to model
                input_tensor = torch.FloatTensor([coords]).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = outputs.max(1)
                    
                    # Decode prediction
                    predicted_class = le.inverse_transform([predicted.item()])[0]
                    
                # Map to word
                word = get_word(predicted_class) if predicted_class not in ['del', 'nothing', 'space'] else ""
                
                # Draw bounding box and results
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                cv2.putText(frame, f"Letter: {predicted_class}", (x_min, max(30, y_min - 40)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                if word:
                    cv2.putText(frame, f"Word: {word}", (x_min, max(60, y_min - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("HandSpeak - ASL Medical Translator (Landmarks AI)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
