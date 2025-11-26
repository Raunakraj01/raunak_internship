import tkinter as tk
from tkinter import messagebox
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')



def summarize(text, num_sentences=2):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) == 0:
        return "No sentences found in the input."

    if num_sentences > len(sentences):
        num_sentences = len(sentences)

    
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Similarity
    sim_matrix = cosine_similarity(vectors)

    # Score sentences
    scores = sim_matrix.sum(axis=1)

    # Top sentences
    ranked_sentences = [
        sentences[i] for i in scores.argsort()[-num_sentences:][::-1]
    ]

    return " ".join(ranked_sentences)



def generate_summary():
    text = input_box.get("1.0", tk.END).strip()
    num = num_sent_box.get().strip()

    if text == "":
        messagebox.showwarning("No Text", "Please enter some text to summarize.")
        return

    if not num.isdigit():
        messagebox.showwarning("Invalid Input", "Number of sentences must be an integer.")
        return

    num = int(num)

    summary = summarize(text, num)

    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, summary)



root = tk.Tk()
root.title("Text Summarizer")
root.geometry("750x550")
root.resizable(False, False)

title = tk.Label(root, text="Text Summarizer", font=("Arial", 18, "bold"))
title.pack(pady=10)

# Input Label
input_label = tk.Label(root, text="Enter your text:", font=("Arial", 12))
input_label.pack()

# Input Text Box
input_box = tk.Text(root, height=10, width=90, font=("Arial", 10))
input_box.pack(pady=5)

# Number of Sentences Label + Entry
num_frame = tk.Frame(root)
num_frame.pack()

num_label = tk.Label(num_frame, text="Summary Sentence Count:", font=("Arial", 12))
num_label.pack(side=tk.LEFT)

num_sent_box = tk.Entry(num_frame, width=5, font=("Arial", 12))
num_sent_box.pack(side=tk.LEFT, padx=5)

# Summarize Button
summ_btn = tk.Button(root, text="Generate Summary", font=("Arial", 12), command=generate_summary)
summ_btn.pack(pady=10)

# Output Label
output_label = tk.Label(root, text="Summary Output:", font=("Arial", 12))
output_label.pack()

# Output Box
output_box = tk.Text(root, height=10, width=90, font=("Arial", 10))
output_box.pack(pady=5)

root.mainloop()