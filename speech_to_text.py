import tkinter as tk
from tkinter import filedialog, messagebox
import speech_recognition as sr

audio_path = ""



def select_file():
    global audio_path
    audio_path = filedialog.askopenfilename(
        title="Select WAV File",
        filetypes=[("WAV Files", "*.wav")]
    )

    if audio_path:
        file_label.config(text="Selected File: " + audio_path.split("/")[-1])
    else:
        file_label.config(text="No file selected")



def transcribe_audio():
    global audio_path

    if audio_path == "":
        messagebox.showwarning("No File", "Please select a WAV audio file first.")
        return

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_path) as source:
            output_text.delete(1.0, tk.END)
            output_text.insert(tk.END, "Listening to audio...\n")

            audio_data = recognizer.record(source)
            output_text.insert(tk.END, "Converting speech to text...\n")

            text = recognizer.recognize_google(audio_data)
            output_text.insert(tk.END, "\nTranscription:\n" + text)

    except sr.UnknownValueError:
        output_text.insert(tk.END, "\n[ERROR] Speech not clear.")

    except sr.RequestError:
        output_text.insert(tk.END, "\n[ERROR] Network error. Google API unreachable.")

    except Exception as e:
        messagebox.showerror("Error", str(e))



root = tk.Tk()
root.title("Speech Recognition GUI")
root.geometry("650x450")
root.resizable(False, False)

title_label = tk.Label(root, text="Speech-to-Text Converter", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

file_btn = tk.Button(root, text="Select Audio File", font=("Arial", 12), command=select_file)
file_btn.pack(pady=5)

file_label = tk.Label(root, text="No file selected", font=("Arial", 10))
file_label.pack()

transcribe_btn = tk.Button(root, text="Transcribe", font=("Arial", 12), command=transcribe_audio)
transcribe_btn.pack(pady=10)

output_text = tk.Text(root, height=12, width=75, font=("Arial", 10))
output_text.pack(pady=10)


root.mainloop()
