# 🤟 HandSpeak — ASL-Based Communication for Non-Verbal Patients

> Real-time American Sign Language (ASL) hand gesture recognition to help non-verbal patients communicate medical needs to healthcare staff.

---

## 📌 Problem Statement

Non-verbal patients in hospitals — whether due to intubation, neurological conditions, or post-surgical recovery — often struggle to communicate basic needs to nurses and doctors. Miscommunication delays care and increases patient distress.

**HandSpeak** bridges this gap. A nurse provides the patient with a printed reference plate showing 26 ASL hand gestures, each mapped to a common medical word. The patient performs gestures in front of a webcam, and the system detects them in real-time, building sentences word-by-word on screen.

---

## 👥 Team

| Name | Role |
|------|------|
| **Mohammed Raaed Azeez** | Project Lead |
| **Mohammed Mahdi Qasim** | Data Engineer |
| **Alhasan Mudher Yaseen** | Systems Engineer |

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Deep Learning | TensorFlow / Keras |
| Hand Detection | MediaPipe Hands |
| Computer Vision | OpenCV |
| Dataset | [ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87,000 images, 26 classes) |
| Dataset Download | `kagglehub` |

---

## 🔄 Pipeline Overview

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│  Webcam  │───▶│ MediaPipe │───▶│   CNN    │───▶│ Word Mapping │───▶│ UI Display │
│  (Feed)  │    │ (Landmarks│    │ (Predict │    │ (gesture_map │    │ (Sentence  │
│          │    │  Extract) │    │  Letter) │    │   .py)       │    │  Builder)  │
└──────────┘    └───────────┘    └──────────┘    └──────────────┘    └────────────┘
```

1. **Camera** — OpenCV captures live webcam frames.
2. **MediaPipe** — Detects hand landmarks (21 key points per hand).
3. **CNN** — A trained convolutional neural network classifies the gesture as a letter (A–Z).
4. **Word Mapping** — Each letter maps to a predefined medical word.
5. **UI Display** — The recognized word appears on screen, building a sentence over time.

---

## 🗂 Folder Structure

```
handspeak/
├── data/              # Dataset (not committed to git)
├── models/            # Saved model weights (not committed to git)
├── notebooks/         # Jupyter notebooks for exploration
├── src/
│   ├── gesture_map.py # Letter → medical word mapping
│   ├── preprocess.py  # Image loading, augmentation, splitting
│   ├── train.py       # Model architecture & training loop
│   ├── inference.py   # Real-time prediction from webcam
│   └── ui.py          # On-screen sentence display overlay
├── results/           # Evaluation outputs, plots, confusion matrices
├── .gitignore
├── CONTRIBUTING.md
├── requirements.txt
└── README.md
```

---

## 🔤 Gesture-to-Word Mapping

| Letter | Word | Letter | Word |
|--------|------|--------|------|
| A | Help | N | No |
| B | Pain | O | Okay |
| C | Water | P | Please |
| D | Food | Q | Quiet |
| E | Medicine | R | Rest |
| F | Nurse | S | Sleep |
| G | Doctor | T | Toilet |
| H | Hot | U | Uncomfortable |
| I | I | V | Very |
| J | Problem | W | Want |
| K | Cold | X | Anxious |
| L | More | Y | Yes |
| M | Stop | Z | Dizzy |

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/mo-raaed/handspeak.git
cd handspeak
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

```python
import kagglehub
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset downloaded to:", path)
```

Move or symlink the downloaded dataset into the `data/` directory.

### 5. Train the Model

```bash
python src/train.py
```

### 6. Run Real-Time Inference

```bash
python src/inference.py
```

---

## 📊 Results & Demo

> _Coming soon — accuracy metrics, confusion matrix, and a demo video will be added here after model training is complete._

---

## 📝 Changelog

* **May 1, 2026:**
  * **Desktop UI Application:** Created `src/ui.py`, a full-featured Tkinter desktop application that embeds the webcam feed, displays real-time gesture detection, and includes a sentence builder with hold-to-confirm input, text-to-speech output, undo/clear controls, and an emergency button for non-verbal patient communication.

* **April 25, 2026:**
  * **Landmarks Inference Update:** Rewrote `src/inference.py` to use the new `LandmarkNN` model. The webcam now extracts the 63 hand coordinates via MediaPipe and feeds them into the tiny Neural Network, achieving flawless real-world background immunity.
  * **Advanced AI Pipeline (Landmarks Training):** Created `src/train_landmarks.py` to train a lightning-fast, lightweight Multi-Layer Perceptron (MLP) on the extracted skeletal coordinates, ignoring raw pixels to completely eliminate background bias.
  * **Advanced AI Pipeline (Landmarks Extraction):** Created `src/extract_landmarks.py` to batch process all 87,000 dataset images using MediaPipe. This script extracts the 21 `(x, y, z)` skeletal coordinates from the hands and saves them to a CSV, allowing us to train a highly robust, background-immune neural network instead of relying on raw pixels.
  * **Aspect Ratio Fix:** Updated `src/inference.py` to enforce square bounding boxes around detected hands. This prevents aspect-ratio stretching when resizing crops to 64x64, significantly improving prediction accuracy.
  * **Real-time Inference:** Created `src/inference.py` using MediaPipe Tasks API for robust hand landmark detection and OpenCV for real-time visualization.
  * **Training Script:** Added `src/train.py` utilizing a customized ResNet18 model to train on the ASL Alphabet dataset, achieving over 99% validation accuracy.

---

## 📄 License

This project is developed for **ENGR 422 — Computer Vision** at the American University of Iraq, Sulaimani (AUIS), Spring 2026.
