"""
ui.py
HandSpeak - ASL Medical Translator
Full desktop UI with webcam feed, sentence builder, and text-to-speech.
Built with Tkinter + OpenCV + MediaPipe + PyTorch.
"""

import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import sys
import pickle
import time
import threading
import collections
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

# Add src directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from gesture_map import get_word, GESTURE_MAP

# ─── Configuration ───────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "landmark_model.pth")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
MP_TASK_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timing: how long a gesture must be held before it is accepted (seconds)
HOLD_THRESHOLD = 2.0
# Cooldown after a word is added before accepting the next one (seconds)
COOLDOWN = 1.0
# Number of recent frames to use for smoothing predictions
SMOOTH_WINDOW = 15

# ─── Control Signals ────────────────────────────────────────────────────────
SIGNAL_START = "V"   # Show V to start translating
SIGNAL_STOP  = "A"   # Show A to stop translating
SIGNAL_CLEAR = "L"   # Show L to clear the sentence

# ─── Colors ──────────────────────────────────────────────────────────────────
BG_DARK    = "#0f1117"
BG_PANEL   = "#1a1d27"
BG_CARD    = "#232735"
ACCENT     = "#6c5ce7"
ACCENT_ALT = "#a29bfe"
GREEN      = "#00b894"
RED        = "#e17055"
ORANGE     = "#fdcb6e"
TEXT_WHITE = "#f0f0f0"
TEXT_DIM   = "#8a8ea0"

# ─── Neural Network (must match training) ────────────────────────────────────
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

# ─── Main Application ────────────────────────────────────────────────────────
class HandSpeakApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HandSpeak — ASL Medical Translator")
        self.root.configure(bg=BG_DARK)
        self.root.state("zoomed")  # Start maximized on Windows
        self.root.minsize(1100, 700)

        # ── State ────────────────────────────────────────────────────────
        self.sentence_words = []
        self.current_letter = ""
        self.current_word = ""
        self.stable_letter = ""       # The smoothed, stable letter
        self.hold_start = 0.0
        self.last_added_time = 0.0
        self.hold_progress = 0.0
        self.running = True
        self.translating = False      # Starts paused — show V to begin
        self.prediction_buffer = collections.deque(maxlen=SMOOTH_WINDOW)

        # ── Load AI Models ───────────────────────────────────────────────
        self._load_models()

        # ── Build UI ─────────────────────────────────────────────────────
        self._build_ui()

        # ── Start webcam loop in a thread ────────────────────────────────
        self.cap = cv2.VideoCapture(0)
        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

        # ── Periodic UI refresh ──────────────────────────────────────────
        self._update_ui()

        # ── Graceful shutdown ────────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═════════════════════════════════════════════════════════════════════
    #  Model Loading
    # ═════════════════════════════════════════════════════════════════════
    def _load_models(self):
        # Label encoder
        with open(ENCODER_PATH, "rb") as f:
            self.le = pickle.load(f)

        # PyTorch landmark model
        num_classes = len(self.le.classes_)
        self.model = LandmarkNN(input_size=63, num_classes=num_classes)
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        )
        self.model.to(DEVICE)
        self.model.eval()

        # MediaPipe hand landmarker
        if not os.path.exists(MP_TASK_PATH):
            os.makedirs(os.path.dirname(MP_TASK_PATH), exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                MP_TASK_PATH,
            )
        base_opts = python.BaseOptions(model_asset_path=MP_TASK_PATH)
        opts = vision.HandLandmarkerOptions(base_options=base_opts, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(opts)

    # ═════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ═════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Fonts ────────────────────────────────────────────────────────
        self.font_title  = tkfont.Font(family="Segoe UI", size=22, weight="bold")
        self.font_large  = tkfont.Font(family="Segoe UI", size=18, weight="bold")
        self.font_medium = tkfont.Font(family="Segoe UI", size=14)
        self.font_small  = tkfont.Font(family="Segoe UI", size=11)
        self.font_sentence = tkfont.Font(family="Segoe UI", size=20, weight="bold")
        self.font_btn    = tkfont.Font(family="Segoe UI", size=12, weight="bold")

        # ── Title Bar ────────────────────────────────────────────────────
        title_bar = tk.Frame(self.root, bg=BG_PANEL, height=56)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar, text="🤟  HandSpeak", font=self.font_title,
            bg=BG_PANEL, fg=ACCENT_ALT
        ).pack(side="left", padx=20)
        tk.Label(
            title_bar, text="ASL → Medical Word Translator",
            font=self.font_small, bg=BG_PANEL, fg=TEXT_DIM
        ).pack(side="left", padx=10)

        # Status indicator (right side of title bar)
        self.status_dot = tk.Label(
            title_bar, text="●", font=self.font_medium, bg=BG_PANEL, fg=RED
        )
        self.status_dot.pack(side="right", padx=5)
        self.status_text = tk.Label(
            title_bar, text="Paused — Show ✌ (V) to start", font=self.font_small,
            bg=BG_PANEL, fg=TEXT_DIM
        )
        self.status_text.pack(side="right", padx=5)

        # ── Bottom: Sentence Bar (pack BEFORE content so it claims space) ─
        sent_bar = tk.Frame(self.root, bg=BG_PANEL)
        sent_bar.pack(fill="x", side="bottom")

        # Sentence label
        sent_top = tk.Frame(sent_bar, bg=BG_PANEL)
        sent_top.pack(fill="x", padx=20, pady=(12, 0))

        tk.Label(
            sent_top, text="📝 SENTENCE BUILDER", font=self.font_small,
            bg=BG_PANEL, fg=TEXT_DIM
        ).pack(side="left")

        self.lbl_sentence = tk.Label(
            sent_bar, text="Make a sign to start building a sentence...",
            font=self.font_sentence, bg=BG_PANEL, fg=TEXT_DIM,
            anchor="w", wraplength=1200
        )
        self.lbl_sentence.pack(fill="x", padx=20, pady=(6, 0))

        # Buttons row
        btn_row = tk.Frame(sent_bar, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=20, pady=(8, 14))

        self._make_btn(btn_row, "🔊  Speak", ACCENT, self._speak_sentence).pack(side="left", padx=(0, 8))
        self._make_btn(btn_row, "⬅  Undo", ORANGE, self._undo_word).pack(side="left", padx=(0, 8))
        self._make_btn(btn_row, "🗑  Clear", RED, self._clear_sentence).pack(side="left", padx=(0, 8))
        self._make_btn(btn_row, "🚨  Emergency", "#d63031", self._emergency).pack(side="right")

        # ── Main Content Area ────────────────────────────────────────────
        content = tk.Frame(self.root, bg=BG_DARK)
        content.pack(fill="both", expand=True, padx=16, pady=10)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        # ── Left: Camera Feed ────────────────────────────────────────────
        cam_frame = tk.Frame(content, bg=BG_CARD, bd=0, highlightthickness=2,
                             highlightbackground=BG_CARD)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.cam_label = tk.Label(cam_frame, bg=BG_CARD)
        self.cam_label.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Right: Side Panel ────────────────────────────────────────────
        side = tk.Frame(content, bg=BG_DARK)
        side.grid(row=0, column=1, sticky="nsew")

        # -- Detection Card --
        det_card = tk.Frame(side, bg=BG_CARD, bd=0)
        det_card.pack(fill="x", pady=(0, 10))

        tk.Label(
            det_card, text="DETECTED SIGN", font=self.font_small,
            bg=BG_CARD, fg=TEXT_DIM
        ).pack(anchor="w", padx=16, pady=(14, 0))

        self.lbl_letter = tk.Label(
            det_card, text="—", font=tkfont.Font(family="Segoe UI", size=48, weight="bold"),
            bg=BG_CARD, fg=ACCENT_ALT
        )
        self.lbl_letter.pack(pady=(0, 4))

        self.lbl_word = tk.Label(
            det_card, text="Waiting for gesture...", font=self.font_medium,
            bg=BG_CARD, fg=TEXT_WHITE
        )
        self.lbl_word.pack(pady=(0, 6))

        # Hold progress bar
        prog_frame = tk.Frame(det_card, bg=BG_CARD)
        prog_frame.pack(fill="x", padx=16, pady=(0, 14))

        tk.Label(
            prog_frame, text="Hold:", font=self.font_small,
            bg=BG_CARD, fg=TEXT_DIM
        ).pack(side="left")

        self.progress_canvas = tk.Canvas(
            prog_frame, height=14, bg=BG_DARK, bd=0, highlightthickness=0
        )
        self.progress_canvas.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # -- Last Added Feedback --
        self.lbl_last_added = tk.Label(
            det_card, text="", font=self.font_small,
            bg=BG_CARD, fg=GREEN
        )
        self.lbl_last_added.pack(pady=(0, 10))

        # -- Gesture Map Reference Card --
        ref_card = tk.Frame(side, bg=BG_CARD, bd=0)
        ref_card.pack(fill="both", expand=True, pady=(0, 0))

        tk.Label(
            ref_card, text="GESTURE MAP", font=self.font_small,
            bg=BG_CARD, fg=TEXT_DIM
        ).pack(anchor="w", padx=16, pady=(14, 6))

        ref_scroll = tk.Frame(ref_card, bg=BG_CARD)
        ref_scroll.pack(fill="x", padx=16, pady=(0, 14))

        # Show a compact grid of letter → word mappings
        cols = 2
        for idx, (letter, word) in enumerate(GESTURE_MAP.items()):
            r, c = divmod(idx, cols)
            lbl = tk.Label(
                ref_scroll,
                text=f"{letter} → {word}",
                font=self.font_small,
                bg=BG_CARD, fg=TEXT_DIM, anchor="w"
            )
            lbl.grid(row=r, column=c, sticky="w", padx=(0, 20), pady=1)

    def _make_btn(self, parent, text, color, command):
        btn = tk.Button(
            parent, text=text, font=self.font_btn,
            bg=color, fg="white", activebackground=color,
            activeforeground="white", bd=0, padx=18, pady=6,
            cursor="hand2", command=command
        )
        return btn

    # ═════════════════════════════════════════════════════════════════════
    #  Camera Loop (runs in a background thread)
    # ═════════════════════════════════════════════════════════════════════
    def _camera_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)

            raw_letter = ""

            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    # Extract & normalize coordinates
                    raw = []
                    for lm in hand_lms:
                        raw.extend([lm.x, lm.y, lm.z])

                    wx, wy, wz = raw[0], raw[1], raw[2]
                    rel = []
                    for i in range(0, len(raw), 3):
                        rel.append(raw[i] - wx)
                        rel.append(raw[i+1] - wy)
                        rel.append(raw[i+2] - wz)

                    mx = max(abs(v) for v in rel)
                    if mx > 0:
                        rel = [v / mx for v in rel]

                    tensor = torch.FloatTensor([rel]).to(DEVICE)
                    with torch.no_grad():
                        out = self.model(tensor)
                        _, pred = out.max(1)
                        raw_letter = self.le.inverse_transform([pred.item()])[0]

                    # Draw bounding box
                    xs = [int(lm.x * w) for lm in hand_lms]
                    ys = [int(lm.y * h) for lm in hand_lms]
                    pad = 20
                    x1 = max(0, min(xs) - pad)
                    y1 = max(0, min(ys) - pad)
                    x2 = min(w, max(xs) + pad)
                    y2 = min(h, max(ys) + pad)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (108, 92, 231), 2)

                    # Draw landmark dots on the hand
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (162, 155, 254), -1)

            # ── Smoothing: use majority vote over last N frames ──────────
            self.prediction_buffer.append(raw_letter)
            if self.prediction_buffer:
                counter = collections.Counter(self.prediction_buffer)
                smoothed_letter, count = counter.most_common(1)[0]
                # Only accept if at least 60% of recent frames agree
                if count >= len(self.prediction_buffer) * 0.6:
                    detected_letter = smoothed_letter
                else:
                    detected_letter = self.stable_letter  # keep previous
            else:
                detected_letter = ""

            # ── Control signal handling ──────────────────────────────────
            now = time.time()

            # START signal: V
            if detected_letter == SIGNAL_START and not self.translating:
                if detected_letter == self.stable_letter:
                    elapsed = now - self.hold_start
                    self.hold_progress = min(elapsed / HOLD_THRESHOLD, 1.0)
                    if elapsed >= HOLD_THRESHOLD:
                        self.translating = True
                        self.hold_start = now
                        self.hold_progress = 0.0
                        self.prediction_buffer.clear()
                else:
                    self.stable_letter = detected_letter
                    self.hold_start = now
                    self.hold_progress = 0.0
                self.current_letter = detected_letter
                self.current_word = "▶ START"
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._latest_frame = rgb_display
                time.sleep(0.01)
                continue

            # STOP signal: A
            if detected_letter == SIGNAL_STOP and self.translating:
                if detected_letter == self.stable_letter:
                    elapsed = now - self.hold_start
                    self.hold_progress = min(elapsed / HOLD_THRESHOLD, 1.0)
                    if elapsed >= HOLD_THRESHOLD:
                        self.translating = False
                        self.hold_start = now
                        self.hold_progress = 0.0
                        self.prediction_buffer.clear()
                else:
                    self.stable_letter = detected_letter
                    self.hold_start = now
                    self.hold_progress = 0.0
                self.current_letter = detected_letter
                self.current_word = "⏹ STOP"
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._latest_frame = rgb_display
                time.sleep(0.01)
                continue

            # CLEAR signal: L
            if detected_letter == SIGNAL_CLEAR and self.translating:
                if detected_letter == self.stable_letter:
                    elapsed = now - self.hold_start
                    self.hold_progress = min(elapsed / HOLD_THRESHOLD, 1.0)
                    if elapsed >= HOLD_THRESHOLD:
                        self.sentence_words.clear()
                        self.hold_start = now
                        self.hold_progress = 0.0
                        self.prediction_buffer.clear()
                else:
                    self.stable_letter = detected_letter
                    self.hold_start = now
                    self.hold_progress = 0.0
                self.current_letter = detected_letter
                self.current_word = "🗑 CLEAR"
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._latest_frame = rgb_display
                time.sleep(0.01)
                continue

            # ── If not translating, skip word detection ───────────────────
            if not self.translating:
                self.current_letter = ""
                self.current_word = ""
                self.hold_progress = 0.0
                rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._latest_frame = rgb_display
                time.sleep(0.01)
                continue

            # ── Normal word detection (only when translating) ────────────
            if detected_letter and detected_letter not in ("del", "nothing", "space", SIGNAL_START, SIGNAL_STOP, SIGNAL_CLEAR):
                detected_word = get_word(detected_letter)
            else:
                detected_word = ""

            # ── Hold-to-confirm logic ────────────────────────────────────
            now = time.time()
            if detected_word and detected_letter == self.stable_letter:
                # Same letter as before — accumulate hold time
                elapsed = now - self.hold_start
                self.hold_progress = min(elapsed / HOLD_THRESHOLD, 1.0)
                if elapsed >= HOLD_THRESHOLD and (now - self.last_added_time) > COOLDOWN:
                    self.sentence_words.append(detected_word)
                    self.last_added_time = now
                    # Reset hold but keep stable_letter so holding longer adds again
                    self.hold_start = now
                    self.hold_progress = 0.0
            elif detected_letter != self.stable_letter:
                # Letter changed — reset hold timer
                self.stable_letter = detected_letter
                self.hold_start = now
                self.hold_progress = 0.0

            self.current_letter = detected_letter
            self.current_word = detected_word

            # Convert frame for Tkinter display
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._latest_frame = rgb_display

            time.sleep(0.01)

    # ═════════════════════════════════════════════════════════════════════
    #  UI Update (runs on main thread via `after`)
    # ═════════════════════════════════════════════════════════════════════
    def _update_ui(self):
        if not self.running:
            return

        # Update camera image
        if hasattr(self, "_latest_frame"):
            img = Image.fromarray(self._latest_frame)
            # Scale to fit the label
            lw = self.cam_label.winfo_width()
            lh = self.cam_label.winfo_height()
            if lw > 10 and lh > 10:
                img = img.resize((lw, lh), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.configure(image=imgtk)
            self.cam_label._imgtk = imgtk  # prevent GC

        # Update detected letter / word
        if self.current_letter and self.current_letter not in ("del", "nothing", "space"):
            self.lbl_letter.configure(text=self.current_letter, fg=ACCENT_ALT)
            self.lbl_word.configure(text=self.current_word if self.current_word else "—", fg=TEXT_WHITE)
        else:
            self.lbl_letter.configure(text="—", fg=TEXT_DIM)
            self.lbl_word.configure(text="Waiting for gesture...", fg=TEXT_DIM)

        # Update status indicator
        if self.translating:
            self.status_dot.configure(fg=GREEN)
            self.status_text.configure(text="Translating — Show ✊ (A) to stop")
        else:
            self.status_dot.configure(fg=RED)
            self.status_text.configure(text="Paused — Show ✌ (V) to start")

        # Update hold progress bar
        self.progress_canvas.delete("all")
        cw = self.progress_canvas.winfo_width()
        if cw > 0:
            fill_w = int(cw * self.hold_progress)
            color = GREEN if self.hold_progress >= 1.0 else ACCENT
            self.progress_canvas.create_rectangle(0, 0, fill_w, 14, fill=color, outline="")

        # Update last-added feedback
        if self.sentence_words:
            self.lbl_last_added.configure(
                text=f"✓ Added: {self.sentence_words[-1]}",
                fg=GREEN
            )

        # Update sentence
        if self.sentence_words:
            self.lbl_sentence.configure(
                text=" ".join(self.sentence_words),
                fg=TEXT_WHITE
            )
        else:
            self.lbl_sentence.configure(
                text="Make a sign to start building a sentence...",
                fg=TEXT_DIM
            )

        self.root.after(33, self._update_ui)  # ~30 FPS

    # ═════════════════════════════════════════════════════════════════════
    #  Button Actions
    # ═════════════════════════════════════════════════════════════════════
    def _speak_sentence(self):
        if not self.sentence_words:
            return
        sentence = " ".join(self.sentence_words)
        # Use Windows SAPI for text-to-speech (runs in background)
        def _speak():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(sentence)
                engine.runAndWait()
            except ImportError:
                # Fallback: use Windows built-in PowerShell speech
                import subprocess
                subprocess.run(
                    ["powershell", "-Command",
                     f"Add-Type -AssemblyName System.Speech; "
                     f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{sentence}')"],
                    shell=True
                )
        threading.Thread(target=_speak, daemon=True).start()

    def _undo_word(self):
        if self.sentence_words:
            self.sentence_words.pop()
            if not self.sentence_words:
                self.lbl_last_added.configure(text="", fg=GREEN)

    def _clear_sentence(self):
        self.sentence_words.clear()
        self.lbl_last_added.configure(text="", fg=GREEN)

    def _emergency(self):
        self.sentence_words = ["EMERGENCY", "—", "I", "NEED", "HELP", "NOW"]
        self._speak_sentence()

    # ═════════════════════════════════════════════════════════════════════
    #  Shutdown
    # ═════════════════════════════════════════════════════════════════════
    def _on_close(self):
        self.running = False
        time.sleep(0.1)
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = HandSpeakApp(root)
    root.mainloop()
