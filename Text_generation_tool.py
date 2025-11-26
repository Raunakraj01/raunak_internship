import tkinter as tk
from tkinter import messagebox
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


# -------------------- LOAD MODEL --------------------
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# important fix
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

model.eval()


# -------------------- GENERATE TEXT FUNCTION --------------------
def generate_text():
    prompt = input_box.get("1.0", tk.END).strip()

    if prompt == "":
        messagebox.showwarning("No Input", "Please enter a topic or prompt.")
        return

    try:
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, "Generating text...\n\n")

        # Encode input text
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=len(input_ids[0]) + 120,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                top_k=50,
                top_p=0.92,
                temperature=0.9,
                do_sample=True,
                repetition_penalty=1.3,
                early_stopping=True
            )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)

        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, generated)

    except Exception as e:
        messagebox.showerror("Error", str(e))


# -------------------- GUI SETUP --------------------
root = tk.Tk()
root.title("GPT-2 Text Generator")
root.geometry("700x500")
root.resizable(False, False)

title_label = tk.Label(root, text="GPT-2 Text Generator", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

input_label = tk.Label(root, text="Enter your topic/prompt:", font=("Arial", 12))
input_label.pack()

input_box = tk.Text(root, height=5, width=80, font=("Arial", 10))
input_box.pack(pady=5)

generate_btn = tk.Button(root, text="Generate Text", font=("Arial", 12), command=generate_text)
generate_btn.pack(pady=10)

output_label = tk.Label(root, text="Generated Output:", font=("Arial", 12))
output_label.pack()

output_text = tk.Text(root, height=12, width=80, font=("Arial", 10))
output_text.pack()

root.mainloop()
