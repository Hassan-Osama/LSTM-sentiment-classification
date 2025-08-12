import re
import os
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

RAW_DATA_PATH = "data/raw/imdb_reviews.csv"
PROCESSED_DATA_PATH = "data/processed/imdb_clean.csv"


def clean_text(text):    
    text = BeautifulSoup(text, "html.parser").get_text()
    
    text = text.lower()
    
    text = re.sub(r"[^a-z\s]", "", text)
    
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_ not in STOPWORDS and token.lemma_.strip() != ""]
    
    return " ".join(tokens)


def preprocess_dataset(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Cleaning texts...")
    df["clean_review"] = df["review"].apply(clean_text)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")


if __name__ == "__main__":
    preprocess_dataset()
