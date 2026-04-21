import pathlib
import joblib
from flask import Flask, render_template, request, jsonify
from preprocess import clean_text
#  App Setup
app = Flask(__name__)
BASE_DIR = pathlib.Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
def load_artifacts():
    """Load model and vectorizer at startup — fail fast with a clear message."""
    model_path = MODEL_DIR / "model.pkl"
    vec_path = MODEL_DIR / "vectorizer.pkl"

    if not model_path.exists() or not vec_path.exists():
        raise FileNotFoundError(
            "Model files not found. Please run  python train.py  first."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


try:
    MODEL, VECTORIZER = load_artifacts()
    print("[OK] Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    MODEL, VECTORIZER = None, None
    print(f"[!] Warning: {e}")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Guard: model not loaded
    if MODEL is None or VECTORIZER is None:
        return jsonify({"error": "Model not loaded. Run python train.py first."}), 503
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message to classify."}), 400

    # Preprocess → vectorize → predict
    cleaned = clean_text(message)
    features = VECTORIZER.transform([cleaned])
    prediction = int(MODEL.predict(features)[0])
    probabilities = MODEL.predict_proba(features)[0]
    confidence = float(probabilities[prediction])

    label = "Spam 🚫" if prediction == 1 else "Not Spam ✅"

    return jsonify(
        {
            "label": label,
            "is_spam": bool(prediction),
            "confidence": round(confidence * 100, 1),
        }
    )
# Entry Point 
if __name__ == "__main__":
    app.run(debug=True, port=5000)
