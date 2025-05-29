# ---------- IMPORTS ----------
import nltk
nltk.data.path.insert(0, r"C:\Users\ASUS\AppData\Roaming\nltk_data")

import re
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random

# ---------- FUNCTION: Extract Text ----------
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ---------- FUNCTION: Preprocess Text ----------
def preprocess_text(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    return " ".join(tokens)

# ---------- SAMPLE TRAINING DATA ----------
train_texts = [
    "team player leadership project manager communication public speaker",
    "creative open-minded research writing art music design thinking",
    "organized detail-oriented responsible deadline planning time management",
    "emotional anxious mood stress overthinking worry nervous",
    "friendly helpful cooperative social volunteer group harmony"
]
train_labels = [
    "Extroversion", "Openness", "Conscientiousness", "Neuroticism", "Agreeableness"
]

# ---------- TRAIN MODEL ----------
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train_texts, train_labels)

# ---------- LOAD AND PREDICT ----------
cv_text = extract_text_from_pdf("cv_sample.pdf")
cleaned_text = preprocess_text(cv_text)
prediction = model.predict([cleaned_text])[0]

# ---------- RANDOM VISUAL TRAITS ----------
def random_trait_scores():
    return {
        "Extroversion": random.randint(20, 90),
        "Openness": random.randint(20, 90),
        "Conscientiousness": random.randint(20, 90),
        "Agreeableness": random.randint(20, 90),
        "Neuroticism": random.randint(20, 90)
    }

# ---------- OUTPUT ----------
print("🧠 Predicted Dominant Personality Trait:", prediction)
print("📊 Trait Scores (for display):")
trait_scores = random_trait_scores()
trait_scores[prediction] = 90  # Boost predicted trait

for trait, score in trait_scores.items():
    print(f"- {trait}: {score}%")
