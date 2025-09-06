# sentiment_analysis_svm.py

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 0. Setup
# -----------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# 1. Map Ratings → Sentiment
# -----------------------------

# load your dataset (make sure df has Ratings & Reviews)
df = pd.read_csv("APPLE_iPhone_SE.csv")

# -----------------------------
# 2. Map Ratings → Sentiment
# -----------------------------
def map_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df['sentiment'] = df['Ratings'].apply(map_sentiment)

# -----------------------------
# 3. Text Cleaning Function
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):  
        return ""
    text = text.replace("READ MORE", "")  
    text = re.sub(r"[^a-zA-Z]", " ", text)  
    tokens = text.lower().split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_reviews'] = df['Reviews'].apply(clean_text)

# -----------------------------
# 4. Feature Extraction (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_reviews']).toarray()
y = df['sentiment']


# -----------------------------
# 5. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Train SVM Model
# -----------------------------
svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)

# -----------------------------
# 7. Predictions
# -----------------------------
y_pred = svm_clf.predict(X_test)

# -----------------------------
# 8. Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Format precision, recall, f1-score to 2 decimals; support to int
for col in ["precision", "recall", "f1-score"]:
    if col in report_df.columns:
        report_df[col] = report_df[col].apply(lambda x: f"{x:.2f}")
if "support" in report_df.columns:
    report_df["support"] = report_df["support"].astype(int)

# Split class rows vs summary rows
class_rows = report_df.loc[["negative", "neutral", "positive"]]
summary_rows = report_df.drop(["negative", "neutral", "positive"], errors="ignore")

# Print with spacing
print("\nClassification Report:")
print(class_rows)
print("\n")
print(summary_rows)


# -----------------------------
# 9. Confusion Matrix Visualization
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

# -----------------------------
# 10. Top Important Words
# -----------------------------
feature_names = vectorizer.get_feature_names_out()
coefs = svm_clf.coef_

# For multi-class, we get coef per class
for idx, label in enumerate(svm_clf.classes_):
    top_features = sorted(zip(coefs[idx], feature_names))[-20:]
    plt.figure(figsize=(6, 6))
    plt.barh([f for _, f in top_features], [c for c, _ in top_features])
    plt.title(f"Top 20 Important Words for {label}")
    #plt.show()


# Save the trained model and vectorizer
joblib.dump(svm_clf, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("Model and vectorizer saved!")
