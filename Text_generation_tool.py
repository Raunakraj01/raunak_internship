"""
Task-4: Text Generation using GPT-2
why gpt we are using: It’s pretrained on huge amounts of data, so we don’t need to train from scratch.
This script takes a user prompt and generates text using the GPT-2 model.
why using transformers Lib?, it is python library used by hugging face it gives ready-to-use tools to work with like, NLP,Text generation,translation many more!
this lib provides the pretrained model like Gpt-2,BERT,T5
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer #now GPT2LMHeadModel why this lib ?,LMH stands for Language Modeling Head, it means the model is used to predict the next word/token
# This is the tool that converts your input text into tokens (numbers) that GPT-2 understands., Also used to decode model output back into readable text.
import torch #why importing this ?, Hugging Face’s transformers library is built on top of PyTorch, The model runs on PyTorch (like GPT2LMHeadModel, also tokenizer input must be in the form of Pytorch tensor

# Load pre-trained GPT2 model and tokenizer
model_name = "gpt2" 
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