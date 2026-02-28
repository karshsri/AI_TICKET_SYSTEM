# train_model.py

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Training data
texts = [
    "I forgot my password",
    "How to reset password",
    "Password is incorrect",
    "Cannot login to my account",
    "How to see leave balance",
    "Check my leave days",
    "How many leaves do I have",
]

labels = [
    "authentication",
    "authentication",
    "authentication",
    "authentication",
    "leave_query",
    "leave_query",
    "leave_query",
]

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Train model
model.fit(texts, labels)

# Save model
joblib.dump(model, "ticket_model.pkl")

print("Model trained and saved!")