import nltk #Natural Language Toolkit → used to split text into sentences
import numpy as np #Used to handle numbers, vectors.
from sklearn.feature_extraction.text import CountVectorizer #Turns sentences into number arrays
from sklearn.metrics.pairwise import cosine_similarity #Measures how similar two sentences are

# Set NLTK data path manually
nltk.data.path.append(r"C:\Users\rajro\AppData\Roaming\nltk_data") # set the path otherwise it will reflect the error
nltk.download('punkt') # This downloads a pre-trained model that breaks a paragraph into sentences.

def summarize(text, num_sentences=2): # this program is for summarization of texy
      # Step 1: Split text into sentences
    sentences = nltk.sent_tokenize(text)
     # Step 2: Create sentence vectors
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()
     # Step 3: Calculate similarity matrix
    sim_matrix = cosine_similarity(vectors)
    # Step 4: Rank sentences using PageRank (or simply score by similarity sum
    scores = sim_matrix.sum(axis=1)
     # Step 5: Select top sentences
    ranked_sentences = [sentences[i] for i in scores.argsort()[-num_sentences:][::-1]]
    # Step 6: Return the summary
    return " ".join(ranked_sentences)

# === Input Text === #in program text to avoid acces problem 
input_text = """
Artificial Intelligence is a branch of computer science that focuses on building smart machines.
It is widely used in healthcare, finance, education, and many other sectors.
AI helps reduce human effort and error.
However, it also brings concerns about privacy and job loss.
Researchers are working on making AI more ethical and explainable.
"""

# === Print Summary ===
summary = summarize(input_text)
print("Summary:\n", summary)

#Note: It may not work on Testing Ppl on CodeTech if Not takking care of installation of libraries using pip