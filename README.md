
# Lexara — Sentiment Analysis Engine

> *Lexara reads every emotion.* A full in-browser NLP pipeline wrapped in a luxury editorial UI — TF-IDF, Logistic Regression, and Naïve Bayes built from scratch, with a live 3D animated globe, parallax effects, and 3D tilt cards.

---

## Overview

**Lexara** is the final version of the Project 2 Sentiment Analysis implementation. It loads labeled review data, preprocesses and vectorizes text using TF-IDF, trains two classifiers, evaluates them, and exposes a polished full-page interface for real-time sentiment prediction — all running entirely in the browser with zero ML libraries.

The project is named **Lexara** (from *lexis* — the vocabulary of a language) and follows a dark luxury editorial design language: deep charcoal backgrounds, gold typography, Cormorant Garamond serif display font, and a rotating 3D wireframe globe that reacts to your prediction.

---
Live Dashboard 



https://madhurimauppathil-pixel.github.io/sentiment-scope/




## Project Requirements Checklist

| Requirement | Implementation |
|---|---|
| Load labeled text data (tweets / reviews) | 70 hand-labeled samples — positive, negative, neutral |
| Preprocess text (clean, tokenize) | Lowercase → strip punctuation → tokenize → stopword removal |
| Convert text to numeric features | Custom **TF-IDF Vectorizer** with L2 normalization |
| Train classifier — Naïve Bayes | **Gaussian Naïve Bayes** with log-space likelihood |
| Train classifier — Logistic Regression | **Softmax Logistic Regression** with gradient descent |
| Evaluate (accuracy, F1) | Accuracy · Macro F1 · Per-class Precision / Recall / F1 |
| CLI to input text + show predicted sentiment | Full interactive UI with confidence score, probability breakdown, token view |

---

## Features

### ML Pipeline (all from scratch — no libraries)
- **TF-IDF Vectorizer** — fits a vocabulary from the corpus, computes smoothed IDF weights, transforms any text into an L2-normalized dense vector
- **Logistic Regression** — softmax multiclass, trained with gradient descent over 200 epochs (lr = 0.1), random weight initialization
- **Gaussian Naïve Bayes** — Gaussian class-conditional likelihood, computed in log-space to prevent underflow, Laplace-smoothed variances
- **Evaluation** — accuracy, per-class precision/recall/F1, macro-averaged F1 on a held-out 20% test split

### UI / Visual
- **3D Wireframe Globe** — rendered on Canvas 2D using perspective projection, latitude/longitude rings, and floating orbiting data points; colour reacts live to the predicted sentiment
- **Parallax hero** — globe and background orbs drift with mouse position via `mousemove` tracking
- **3D Tilt Cards** — every card has a `perspective(800px)` CSS 3D tilt effect driven by mouse position
- **Animated Counters** — stats count up from zero on load using `setInterval`
- **Boot screen** — staged loading messages ("Fitting vectorizer… → Training classifiers… → Evaluating metrics…") with a glowing gold progress bar
- **Sticky navigation** — smooth-scroll to Analyse / Metrics sections
- **Scan-line animation** — a sweeping gold line overlays the result card for a terminal aesthetic
- **History panel** — last 5 predictions, clickable to restore
- **Model switcher** — toggle between Logistic Regression and Naïve Bayes globally

---

## File Structure

```
sentiment_3d.jsx
│
├── DATASET                    # 70 labeled text samples (positive/negative/neutral)
├── SW                         # Stopword set (~80 common English words)
│
├── clean(text)                # Preprocess: lowercase → strip punct → tokenize → filter stopwords
│
├── class TFIDF                # TF-IDF Vectorizer
│   ├── .fit(docs)             # Build vocab + compute IDF weights
│   └── .tr(tokens)            # Transform tokens → L2-normalised TF-IDF vector
│
├── class LR                   # Logistic Regression (Softmax)
│   ├── .fit(X, y)             # Gradient descent, 200 epochs
│   ├── .prob(x)               # Softmax probabilities per class
│   └── .pred(x)               # Argmax class label
│
├── class NB                   # Gaussian Naïve Bayes
│   ├── .fit(X, y)             # Estimate priors, means, variances
│   ├── .ll(x, c)              # Log-likelihood for class c
│   └── .pred(x)               # Argmax class label
│
├── calcMetrics(yT, yP, C)     # Accuracy · per-class P/R/F1 · macro F1
├── buildPipeline()            # Orchestrate: vectorize → 80/20 split → train both models
│
├── Scene3D({ sentiment })     # Canvas 2D animated 3D wireframe globe
├── TiltCard({ children })     # CSS perspective 3D mouse-tilt wrapper component
├── Counter({ to })            # Animated number count-up component
│
└── App()                      # Main: boot screen → hero → analyse → metrics → pipeline → footer
```

---

## How the ML Pipeline Works

### Step 1 — Preprocessing

```
"I LOVE this product!! It's GREAT."
  → lowercase          "i love this product it s great"
  → strip punctuation  "i love this product it s great"
  → split & filter     ["love", "product", "great"]
  (stopwords like "i", "this", "it", "s" are removed)
```

### Step 2 — TF-IDF Vectorization

```
TF(t, d)  = count(t in d) / |d|
IDF(t)    = log((N+1) / (df(t)+1)) + 1     ← smoothed to avoid zero division
TF-IDF(t) = TF(t, d) × IDF(t)
vector    = TF-IDF values → L2 normalized  ← unit vector for cosine similarity
```

### Step 3 — Training (80 / 20 split)

**Logistic Regression** — gradient descent on softmax cross-entropy:
```
scores  = W·x + b           ← one weight vector per class
probs   = softmax(scores)
error   = prob[c] - (1 if true class else 0)
W[c]   -= lr × error × x
b[c]   -= lr × error
```

**Naïve Bayes** — Gaussian class-conditional distributions:
```
P(x | c) = Gaussian(x_j ; μ_cj, σ²_cj)     ← per feature, per class
P(c | x) ∝ P(c) × ∏ P(x_j | c)             ← computed in log-space
```

### Step 4 — Evaluation

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (P × R) / (P + R)
Macro F1  = mean(F1_positive, F1_negative, F1_neutral)
```

### Step 5 — Prediction

```
tokens  = clean(input text)
vector  = tfidf.tr(tokens)
proba   = model.prob(vector)       ← softmax or normalised Bayes probabilities
label   = argmax(proba)            ← "positive" | "negative" | "neutral"
```

---

## Dataset

| Class | Count | Description |
|---|---|---|
| `positive` | 43 | Strong satisfaction, praise, recommendation |
| `negative` | 15 | Complaints, defects, regret, warnings |
| `neutral`  | 10 | Average, acceptable, neither good nor bad |

**Train / Test split:** 80% train / 20% test (stratified by order)

---

## UI Sections

| Section | Description |
|---|---|
| **Hero** | Full-screen landing with animated 3D globe, parallax mouse orbs, headline, and animated stat strip |
| **§ 01 — Analyse** | Textarea input, model selector (LR / NB), example chips, result card with confidence + probability breakdown + token display, history sidebar |
| **§ 02 — Performance Metrics** | 4-stat accuracy grid, per-class Precision / Recall / F1 cards with glow bars, model toggle |
| **§ 03 — Pipeline Architecture** | 6-card step-by-step breakdown of the NLP pipeline |
| **Footer** | Lexara branding |

---

## Running the Project

### Option A — Claude Artifact (instant)
Paste the contents of `sentiment_3d.jsx` directly into Claude.ai as a React Artifact. It runs immediately — no setup needed.

### Option B — Vite (local dev)

```bash
npm create vite@latest lexara -- --template react
cd lexara
# Replace src/App.jsx with the contents of sentiment_3d.jsx
npm install
npm run dev
```

Then open `http://localhost:5173`.

### Option C — CodeSandbox / StackBlitz
Create a new React sandbox → paste the file into `App.jsx` → runs instantly in the browser.

---

## Design System

| Token | Value | Usage |
|---|---|---|
| `bg` | `#0a0804` | Page background — warm near-black |
| `cream` | `#f5f0e8` | Primary body text |
| `gold` | `#c9a84c` | Accent, borders, highlights |
| `goldLight` | `#e8c96a` | Bright gold — headings and stat values |
| Positive | `#4ade80` | Green — positive sentiment glow |
| Negative | `#fb7185` | Rose — negative sentiment glow |
| Neutral | `#fbbf24` | Amber — neutral sentiment glow |

**Fonts:** Cormorant Garamond (display serif) · Rajdhani (labels / all-caps) · JetBrains Mono (data / numbers)

---

## Tech Stack

| Layer | Detail |
|---|---|
| Framework | React 18 — `useState`, `useEffect`, `useRef`, `useCallback` |
| Styling | Inline CSS + injected `<style>` block + Google Fonts |
| 3D Graphics | Canvas 2D API with manual perspective projection (no Three.js) |
| ML | 100% custom JavaScript — no scikit-learn, no TensorFlow, no ML libraries |
| Runtime | Browser only — zero server, zero backend, zero dependencies beyond React |

---

## Author

Built as **Project 2 — Sentiment Analysis**.  
All ML algorithms implemented from scratch in JavaScript.  
Final version: **Lexara** — luxury editorial redesign with 3D Canvas globe and interactive pipeline.
