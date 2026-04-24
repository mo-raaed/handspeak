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

## 📄 License

This project is developed for **ENGR 422 — Computer Vision** at the American University of Iraq, Sulaimani (AUIS), Spring 2026.
