# 📱 SMS Spam Classifier

A web-based SMS spam detection app powered by a **Naive Bayes** machine learning model trained on the UCI SMS Spam Collection dataset. Enter any message and get an instant **Spam / Not Spam** prediction with a confidence score.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)

---

## ✨ Features

- **Real-time prediction** — results appear instantly via AJAX (no page reload)
- **Confidence score** — animated bar showing model certainty
- **Sample messages** — six click-to-test examples (spam & ham)
- **Clean light UI** — centered card layout, smooth animations, fully responsive
- **Modular code** — training and inference are cleanly separated

---

## 🗂️ Project Structure

```
SMS Spam Classifier/
├── data/
│   └── spam.csv              # Auto-downloaded UCI SMS dataset
├── models/
│   ├── model.pkl             # Trained MultinomialNB model
│   └── vectorizer.pkl        # Fitted TF-IDF vectorizer
├── static/
│   ├── css/
│   │   └── style.css         # Light-theme stylesheet
│   └── js/
│       └── main.js           # AJAX prediction + UI animations
├── templates/
│   └── index.html            # Jinja2 HTML template
├── preprocess.py             # Shared text cleaning utilities
├── train.py                  # Dataset loading + model training script
├── app.py                    # Flask web application
├── spam.csv                  # (Optional) Kaggle SMS Spam CSV to merge in
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project

```bash
cd "SMS Spam Classifier"
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train.py
```

This will:
- Auto-download the UCI SMS Spam Collection dataset (if not already present)
- Merge with `spam.csv` in the root directory if available (Kaggle format)
- Deduplicate and preprocess the combined dataset
- Train a **MultinomialNB** classifier with **TF-IDF** features
- Print accuracy, classification report, and confusion matrix
- Save `models/model.pkl` and `models/vectorizer.pkl`

### 4. Start the web app

```bash
python app.py
```

Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 🤖 Model Details

| Component | Details |
|---|---|
| **Algorithm** | Multinomial Naive Bayes |
| **Features** | TF-IDF (top 5,000 unigrams + bigrams) |
| **Dataset** | UCI SMS Spam Collection (5,572 messages) |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Accuracy** | ~98% on test set |
| **Inference Time** | < 50ms |

### Preprocessing Pipeline

1. Lowercase the text
2. Remove URLs
3. Strip punctuation and non-alphabetic characters
4. Remove English stopwords (NLTK)

---

## 🌐 API Reference

### `GET /`
Returns the main UI page.

### `POST /predict`
Classifies a message as spam or ham.

**Request body (JSON):**
```json
{ "message": "Congratulations! You've won a free iPhone." }
```

**Response (JSON):**
```json
{
  "label": "Spam 🚫",
  "is_spam": true,
  "confidence": 99.1
}
```

**Error responses:**
| Status | Reason |
|---|---|
| `400` | Empty message provided |
| `503` | Model not loaded — run `python train.py` first |

---

## 📦 Dependencies

```
flask>=2.3.0
scikit-learn>=1.3.0
pandas>=2.0.0
nltk>=3.8.0
joblib>=1.3.0
requests>=2.31.0
```

---

## 📊 Dataset

The model is trained on the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — a public dataset of 5,572 tagged SMS messages (747 spam, 4,825 ham).

You can also drop in the [Kaggle version](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (`spam.csv`) into the project root and re-run `python train.py`. The trainer will automatically merge and deduplicate both sources.

---

## 🚀 Usage Tips

- **Test with samples** — click any of the six sample chips on the homepage
- **Retrain anytime** — just run `python train.py` again after adding new data
- **Production deployment** — replace `app.run(debug=True)` with a WSGI server like Gunicorn

---

## 📄 License

This project is for educational purposes. The SMS Spam Collection dataset is provided under the [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) license.
