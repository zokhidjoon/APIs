# textprotools_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect
from textblob import TextBlob

app = FastAPI(title="TextProTools API")

summarizer = pipeline("summarization")

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(input: TextInput):
    summary = summarizer(input.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

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
