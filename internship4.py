"""
Task-4: Text Generation using GPT-2
This script takes a user prompt and generates text using the GPT-2 model.
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT2 model and tokenizer
model_name = "gpt2"  # You can also use "distilgpt2" for a smaller version
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

# User input prompt
prompt = input("Enter your topic: ")

# Encode input and generate output
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        early_stopping=True
    )

# Decode and print result
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)