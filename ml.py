"""Expense categoriser: TF-IDF + Logistic Regression.

SECURITY: joblib files can execute code when loaded. Only load model files
that this app created itself.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "category_model.joblib"

SEED = {
    "Food": ["grocery store weekly shop", "pizza dinner", "starbucks coffee", "restaurant lunch",
             "swiggy food order", "supermarket vegetables", "burger and fries", "bakery bread"],
    "Transport": ["uber ride", "metro card recharge", "petrol fuel station", "bus ticket",
                  "taxi to airport", "train pass", "parking fee", "auto rickshaw fare"],
    "Housing": ["monthly rent", "apartment rent payment", "home repair plumber", "furniture for room",
                "maintenance society fee", "hostel fees", "mortgage payment", "landlord deposit"],
    "Utilities": ["electricity bill", "water bill", "internet broadband bill", "mobile recharge",
                  "gas cylinder", "phone postpaid bill", "wifi subscription", "power bill"],
    "Entertainment": ["netflix subscription", "movie tickets", "spotify premium", "concert tickets",
                      "video game purchase", "bowling night", "amusement park", "streaming service"],
    "Shopping": ["amazon order", "new shoes", "clothes shopping", "headphones purchase",
                 "online shopping gadgets", "gift for friend", "backpack", "watch purchase"],
    "Health": ["pharmacy medicines", "doctor consultation", "dental checkup", "gym membership",
               "hospital bill", "health insurance premium", "eye test glasses", "vitamins"],
    "Education": ["tuition fee", "online course", "textbooks", "exam registration",
                  "udemy class", "stationery notebooks", "coaching classes", "library membership"],
    "Salary": ["monthly salary credited", "paycheck", "salary from employer", "stipend received",
               "freelance payment", "bonus payout", "internship stipend", "wages"],
    "Other": ["miscellaneous", "unknown payment", "atm withdrawal", "bank charges",
              "donation", "cash given", "fine penalty", "other expense"],
}


def _seed_frame() -> pd.DataFrame:
    return pd.DataFrame([(t, c) for c, texts in SEED.items() for t in texts],
                        columns=["description", "category"])


def _build():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
                         LogisticRegression(max_iter=1000))


def train_model(user_df: pd.DataFrame = None):
    """Train on seed data plus (optionally) the user's labelled transactions.
    Returns (model, accuracy on a held-out split)."""
    data = _seed_frame()
    if user_df is not None and not user_df.empty:
        data = pd.concat([data, user_df[["description", "category"]]], ignore_index=True)
    stratify = data["category"] if data["category"].value_counts().min() >= 2 else None
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=stratify)
    model = _build().fit(train["description"], train["category"])
    accuracy = accuracy_score(test["category"], model.predict(test["description"]))
    model = _build().fit(data["description"], data["category"])  # final model on all data
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model, accuracy


def load_or_train():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)  # trusted: created by train_model
    return train_model()[0]


def predict_category(description: str, model=None):
    if not str(description).strip():
        raise ValueError("Description cannot be blank.")
    model = model or load_or_train()
    probs = model.predict_proba([description])[0]
    i = probs.argmax()
    return model.classes_[i], float(probs[i])
