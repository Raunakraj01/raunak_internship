import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab') # Download the 'punkt_tab' resource

def summarize(text, num_sentences=2):
    # Step 1: Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Step 2: Convert sentences to vectors
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Step 3: Calculate similarity matrix
    sim_matrix = cosine_similarity(vectors)

    # Step 4: Score sentences based on similarity
    scores = sim_matrix.sum(axis=1)

    # Step 5: Pick top ranked sentences
    ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:][::-1]]

    # Step 6: Join and return summary
    return " ".join(ranked_sentences)

# === Test Example ===
input_text = """
Artificial Intelligence is a branch of computer science that focuses on building smart machines.
It is widely used in healthcare, finance, education, and many other sectors.
AI helps reduce human effort and error.
However, it also brings concerns about privacy and job loss.
Researchers are working on making AI more ethical and explainable.
"""

summary = summarize(input_text, num_sentences=2)
print("Summary:\n", summary)