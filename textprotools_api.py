from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langdetect import detect
from textblob import TextBlob

app = FastAPI(title="TextProTools API (Remote Summarizer)")

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(input: TextInput):
    headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
    payload = {"inputs": input.text}
    response = requests.post("https://api-inference.huggingface.co/models/facebook/bart-large-cnn", headers=headers, json=payload)
    result = response.json()
    summary = result[0]["summary_text"] if isinstance(result, list) else result
    return {"summary": summary}

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
