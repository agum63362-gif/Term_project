from flask import Flask, render_template, request
import pandas as pd
import re
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Function to load CSV robustly
def load_dataset(filename):
    texts = []
    languages = []
    
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                if len(row) > 2:
                    text = ','.join(row[:-1])
                    language = row[-1].strip()
                else:
                    text = row[0].strip()
                    language = row[1].strip()
                
                if text and language:
                    texts.append(text)
                    languages.append(language)
    
    return pd.DataFrame({"text": texts, "language": languages})

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    url = "http://127.0.0.1:5000"

    print("Loading dataset...")
    df = load_dataset("language_dataset.csv")
    print(f"✅ Successfully loaded {len(df)} rows - {url}")

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]

    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["language"]

    # Train model
    model = MultinomialNB()
    model.fit(X, y)

    print("✅ Model trained successfully!")

    # Prediction function
    def predict_language(text):
        if not text or not text.strip():
            return "Please enter some text"
        text = clean_text(text)
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)
        return prediction[0]

    @app.route("/", methods=["GET", "POST"])
    def home():
        prediction = ""
        sentence = ""
        
        if request.method == "POST":
            sentence = request.form.get("sentence", "").strip()
            if sentence:
                try:
                    prediction = predict_language(sentence)
                except Exception:
                    prediction = "Error in prediction"

        return render_template("index.html", prediction=prediction, sentence=sentence)

    # Run app without duplicate reload
    app.run(debug=True, use_reloader=False)
    