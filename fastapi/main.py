from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from shared.preprocessing import clean_text


app = FastAPI()

# Load the model and vectorizer
model = joblib.load('logitsic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define request body format
class NewsRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "API is working fine"}

@app.post("/post_news")
def predict_news(news: NewsRequest):
    user_input = news.text

    if not user_input.strip():
        return {"error": "Input text is empty."}

    cleaned_input = clean_text(user_input)
    vect_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vect_input)[0]
    probs = model.predict_proba(vect_input)[0]

    return {
        "input": user_input,
        "cleaned_input": cleaned_input,
        "prediction": "Fake News ⚠️" if prediction == 0 else "Real News ✅",
        "confidence": {
            "fake": round(probs[0], 2),
            "real": round(probs[1], 2)
        }
    }
