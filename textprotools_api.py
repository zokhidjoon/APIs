from fastapi import FastAPI
from pydantic import BaseModel
from langdetect import detect
from textblob import TextBlob
import requests
import os

app = FastAPI(title="TextProTools API (Remote Summarizer)")

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(input: TextInput):
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    payload = {"inputs": input.text}

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
            headers=headers,
            json=payload,
            timeout=60
        )
        result = response.json()

        if isinstance(result, list) and "summary_text" in result[0]:
            return {"summary": result[0]["summary_text"]}
        elif "error" in result:
            return {"error": result["error"]}
        else:
            return {"error": "Unexpected response", "raw": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    blob = TextBlob(input.text)
    polarity = blob.sentiment.polarity
    sentiment = "positive" if polarity > 0 else "negative" if polarity < 0 else "neutral"
    return {"sentiment": sentiment, "polarity": polarity}

@app.post("/langdetect")
def detect_language(input: TextInput):
    lang = detect(input.text)
    return {"language": lang}

@app.post("/wordcount")
def word_count(input: TextInput):
    word_count = len(input.text.split())
    char_count = len(input.text)
    return {"word_count": word_count, "char_count": char_count}

# ✅ Health check endpoint for RapidAPI
@app.get("/health")
def health_check():
    return {"status": "ok"}
