import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download movie_reviews if not already
nltk.download("movie_reviews")
nltk.download("punkt")

# -------------------------------
# 1. Load dataset
# -------------------------------
reviews = []
labels = []

for category in movie_reviews.categories():  # pos / neg
    for fileid in movie_reviews.fileids(category):
        review_text = movie_reviews.raw(fileid)
        reviews.append(review_text)
        labels.append(1 if category == "pos" else 0)

df = pd.DataFrame({"review": reviews, "sentiment": labels})

# -------------------------------
# 2. Preprocessing (Negation Handling)
# -------------------------------
def handle_negations(text):
    # Turn "not good" -> "not_good"
    return re.sub(r"\bnot\s+(\w+)", r"not_\1", text.lower())

df["review"] = df["review"].apply(handle_negations)

# -------------------------------
# 3. Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)

# -------------------------------
# 4. Vectorize
# -------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 5. Train Model
# -------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# -------------------------------
# 6. Evaluate
# -------------------------------
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# -------------------------------
# 7. Save Artifacts
# -------------------------------
joblib.dump(model, "best_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("💾 Model and vectorizer saved as 'best_model.pkl' and 'vectorizer.pkl'")

# -------------------------------
# 8. Quick Test
# -------------------------------
test_text = "This movie was not good! but plot was nice"
test_text = handle_negations(test_text)
X_new = vectorizer.transform([test_text])
prediction = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0]

print("\nQuick Test Review:", test_text)
print("Prediction:", "Positive 😀" if prediction == 1 else "Negative 😞")
print("Confidence:", f"{max(proba)*100:.2f}%")
