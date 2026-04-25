import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
import mediapipe as mp
import os
from gesture_map import get_word
from PIL import Image

# Configuration
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_model(num_classes):
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    return model

def get_bounding_box(hand_landmarks, width, height):
    x_min, y_min = width, height
    x_max, y_max = 0, 0
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * width), int(lm.y * height)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    
    # Add padding
    pad = 40
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(width, x_max + pad)
    y_max = min(height, y_max + pad)
    return x_min, y_min, x_max, y_max

def main():
    print("Loading model...")
    model = build_model(len(CLASS_NAMES))
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror
        h, w, c = frame.shape
        
        # Convert BGR to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box
                x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, w, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                if x_max > x_min and y_max > y_min:
                    # Crop and predict
                    crop_img = rgb_frame[y_min:y_max, x_min:x_max]
                    pil_img = Image.fromarray(crop_img)
                    
                    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        _, predicted = outputs.max(1)
                        predicted_class = CLASS_NAMES[predicted.item()]
                        
                        # Map to word
                        word = get_word(predicted_class) if predicted_class not in ['del', 'nothing', 'space'] else ""
                        
                        # Display
                        cv2.putText(frame, f"Letter: {predicted_class}", (x_min, y_min - 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                        if word:
                            cv2.putText(frame, f"Word: {word}", (x_min, y_min - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("HandSpeak - ASL Medical Translator", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
