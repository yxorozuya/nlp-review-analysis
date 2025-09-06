from flask import Flask, render_template, request
import joblib

# Load trained model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        review = request.form["review"]
        X = vectorizer.transform([review]).toarray()  
        result = model.predict(X)[0] 
        prediction = result.capitalize()  # "Positive", "Neutral", "Negative"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
