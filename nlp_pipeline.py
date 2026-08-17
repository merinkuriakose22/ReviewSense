import re
from collections import Counter
import streamlit as st
import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_PATH = (
    "models/reviewsense-roberta"
)


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# --------------------------------------------------
# Rating-based initial sentiment
# --------------------------------------------------

def rating_to_sentiment(rating):

    try:

        rating = float(rating)

    except:

        return "Neutral"

    if rating <= 2:
        return "Negative"

    elif rating == 3:
        return "Neutral"

    else:
        return "Positive"


# --------------------------------------------------
# Conflict detection
# --------------------------------------------------

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "fantastic",
    "happy",
    "satisfied",
    "impressed",
    "easy",
    "fast",
    "perfect",
    "worth",
    "reliable",
    "recommend",
    "love",
    "loved"
}


NEGATIVE_WORDS = {
    "bad",
    "poor",
    "terrible",
    "awful",
    "disappointed",
    "dissatisfied",
    "worst",
    "useless",
    "slow",
    "expensive",
    "problem",
    "problems",
    "issue",
    "issues",
    "unreliable",
    "broken",
    "failure",
    "failed",
    "hate",
    "hated"
}


def detect_text_sentiment(review):

    words = str(review).lower().split()

    positive_count = sum(
        word in POSITIVE_WORDS
        for word in words
    )

    negative_count = sum(
        word in NEGATIVE_WORDS
        for word in words
    )

    if (
        positive_count == 0
        and negative_count == 0
    ):
        return "Unknown"

    if (
        positive_count > negative_count
    ):
        return "Positive"

    if (
        negative_count > positive_count
    ):
        return "Negative"

    return "Mixed"


def detect_conflict(row):

    rating_sentiment = (
        row["initial_sentiment"]
    )

    text_sentiment = (
        row["text_signal"]
    )

    if text_sentiment in [
        "Unknown",
        "Mixed"
    ]:

        return text_sentiment == "Mixed"

    return (
        rating_sentiment
        != text_sentiment
    )


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load_data(
    file_path="data/customer_reviews.csv"
):

    df = pd.read_csv(
        file_path
    )

    # Normalize column names
    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    # Convert column names to expected names
    column_mapping = {}

    for col in df.columns:

        lower = col.lower()

        if lower == "id":
            column_mapping[col] = "ID"

        elif lower == "product":
            column_mapping[col] = "Product"

        elif lower == "brand":
            column_mapping[col] = "Brand"

        elif lower == "category":
            column_mapping[col] = "Category"

        elif lower == "rating":
            column_mapping[col] = "Rating"

        elif lower == "review":
            column_mapping[col] = "Review"

        elif lower == "date":
            column_mapping[col] = "Date"

    df = df.rename(
        columns=column_mapping
    )

    required_columns = [
        "Product",
        "Brand",
        "Category",
        "Rating",
        "Review",
        "Date"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    df["Rating"] = pd.to_numeric(
        df["Rating"],
        errors="coerce"
    )

    df["Review"] = (
        df["Review"]
        .fillna("")
        .astype(str)
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # IMPORTANT:
    # No stopword removal,
    # stemming, lemmatization,
    # punctuation cleaning etc.
    #
    # Your CSV is already cleaned.

    df["initial_sentiment"] = (
        df["Rating"]
        .apply(rating_to_sentiment)
    )

    df["text_signal"] = (
        df["Review"]
        .apply(detect_text_sentiment)
    )

    df["conflict"] = (
        df.apply(
            detect_conflict,
            axis=1
        )
    )

    return df


# --------------------------------------------------
# Load RoBERTa
# --------------------------------------------------

@st.cache_resource
def load_model():

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_PATH
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_PATH
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    model.eval()

    return (
        tokenizer,
        model,
        device
    )


# --------------------------------------------------
# Predict sentiment
# --------------------------------------------------

def predict_sentiment(
    reviews,
    tokenizer,
    model,
    device
):

    results = []

    for review in reviews:

        encoded = tokenizer(
            str(review),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )

        encoded = {
            key: value.to(device)
            for key, value
            in encoded.items()
        }

        with torch.no_grad():

            output = model(
                **encoded
            )

        probabilities = torch.softmax(
            output.logits,
            dim=1
        )

        confidence, prediction = (
            torch.max(
                probabilities,
                dim=1
            )
        )

        prediction = (
            prediction.item()
        )

        confidence = (
            confidence.item()
        )

        results.append({
            "sentiment": LABELS[
                prediction
            ],
            "confidence": confidence
        })

    return results


# --------------------------------------------------
# Keyword extraction
# --------------------------------------------------

def extract_keywords(
    reviews,
    top_n=20
):

    text = " ".join(
        str(review).lower()
        for review in reviews
    )

    words = re.findall(
        r"\b[a-zA-Z]{3,}\b",
        text
    )

    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "was",
        "are",
        "but",
        "not",
        "you",
        "very",
        "have",
        "has",
        "from",
        "its",
        "they",
        "their",
        "product"
    }

    words = [
        word
        for word in words
        if word not in stopwords
    ]

    counter = Counter(words)

    return counter.most_common(
        top_n
    )
