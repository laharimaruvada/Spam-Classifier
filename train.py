import io
import urllib.request
import zipfile
import pathlib
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import clean_text
# Paths
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
UCI_CSV_PATH = DATA_DIR / "spam.csv"
# UCI SMS Spam Collection 
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00228/smsspamcollection.zip"
)
# UCI Dataset Download 
def download_uci_dataset():
    """Download and extract the UCI SMS Spam Collection if not already present."""
    if UCI_CSV_PATH.exists():
        print(f"[OK] UCI dataset already exists at {UCI_CSV_PATH}")
        return

    print("[>>] Downloading UCI SMS Spam Collection ...")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
            zip_bytes = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            raw_bytes = zf.read("SMSSpamCollection")

        lines = raw_bytes.decode("utf-8", errors="replace").splitlines()
        rows = [line.split("\t", 1) for line in lines if line.strip()]
        df = pd.DataFrame(rows, columns=["label", "message"])
        df.to_csv(UCI_CSV_PATH, index=False)
        print(f"[OK] Saved {len(df):,} rows to {UCI_CSV_PATH}")

    except Exception as exc:
        print(f"[!!] UCI download failed ({exc}). Will rely on root spam.csv only.")
# Dataset Loading 
def load_dataset():
    frames = []

    # Source 1: root-level spam.csv (Kaggle version)
    root_csv = BASE_DIR / "spam.csv"
    if root_csv.exists():
        df_root = pd.read_csv(root_csv, encoding="latin-1", usecols=[0, 1])
        df_root.columns = ["label", "message"]
        frames.append(df_root)
        print(f"[OK] Loaded {len(df_root):,} rows from root spam.csv (Kaggle)")
    else:
        print("[!!] Root spam.csv not found - skipping Kaggle source.")
    # Source 2: data/spam.csv (downloaded UCI version)
    download_uci_dataset()
    if UCI_CSV_PATH.exists():
        df_uci = pd.read_csv(UCI_CSV_PATH, encoding="latin-1")
        df_uci = df_uci.iloc[:, :2]
        df_uci.columns = ["label", "message"]
        frames.append(df_uci)
        print(f"[OK] Loaded {len(df_uci):,} rows from data/spam.csv (UCI)")
    if not frames:
        raise RuntimeError("No dataset found. Cannot train.")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined.drop_duplicates(subset=["message"], inplace=True)
    after = len(combined)
    if before != after:
        print(f"[-] Removed {before - after:,} duplicate messages ({after:,} unique kept)")

    combined.dropna(subset=["label", "message"], inplace=True)
    return combined
def train():
    df = load_dataset()
    # Encode labels
    df["label_enc"] = df["label"].str.strip().map({"ham": 0, "spam": 1})
    df.dropna(subset=["label_enc"], inplace=True)
    df["label_enc"] = df["label_enc"].astype(int)
    spam_count = int(df["label_enc"].sum())
    ham_count = int((df["label_enc"] == 0).sum())
    print(f"[-] Final dataset: {len(df):,} rows  |  Spam: {spam_count:,}  |  Ham: {ham_count:,}")
    # Clean text
    print("[-] Preprocessing text ...")
    df["clean_message"] = df["message"].apply(clean_text)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"],
        df["label_enc"],
        test_size=0.20,
        random_state=42,
        stratify=df["label_enc"],
    )

    # TF-IDF Vectorizer
    print("[-] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train MultinomialNB
    print("[-] Training MultinomialNB ...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    sep = "-" * 50
    print("\n" + sep)
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(sep)
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(sep + "\n")

    # Save artefacts
    model_path = MODEL_DIR / "model.pkl"
    vec_path = MODEL_DIR / "vectorizer.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"[OK] Model saved      -> {model_path}")
    print(f"[OK] Vectorizer saved -> {vec_path}")
if __name__ == "__main__":
    train()
