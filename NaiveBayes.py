import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("APPLE_iPhone_SE.csv")

print(df['Ratings'].value_counts())

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
# 5. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Naïve Bayes Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 7. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 8. Confusion Matrix Visualization
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["negative", "neutral", "positive"],
            yticklabels=["negative", "neutral", "positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()