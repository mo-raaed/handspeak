# Contributing to HandSpeak

Thanks for contributing! Follow these guidelines to keep the repo clean and the team in sync.

---

## 🔀 Branch Naming

Create a new branch for every feature or fix. Use this format:

```
your-name/short-description
```

**Examples:**

```
raaed/add-training-script
mahdi/preprocess-pipeline
alhasan/webcam-ui-overlay
```

---

## 🚫 Never Push Directly to `main`

The `main` branch is protected. All changes go through pull requests.

1. Create your feature branch off `main`.
2. Work on your branch locally.
3. Push your branch to GitHub.
4. Open a Pull Request (PR) targeting `main`.

---

## 📬 Opening a Pull Request

1. Give your PR a clear title (e.g., *"Add CNN training script"*).
2. In the description, explain **what** you changed and **why**.
3. Tag at least one teammate for review.
4. Wait for approval before merging.

---

## ✅ Before You Push

Always run the project locally to make sure nothing is broken:

```bash
# Activate your virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt

# Run a quick test (adjust as needed)
python src/gesture_map.py
```

If your changes involve training or inference, verify those scripts run without errors before pushing.

---

## 💬 Questions?

Open an issue or message the team on our group chat. Don't sit on a blocker — ask early.
