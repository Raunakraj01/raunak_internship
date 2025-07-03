#Description: A small Program implementing already existing Speach recognition to avoid complexity 
import speech_recognition as sr # Using recognition library.

# Initialize recognizer
recognizer = sr.Recognizer() #Creates a recognizer object to store the Speach

# Load the audio file
with sr.AudioFile("sample_audio.wav") as source: #Loads audio file.
    print("Listening to audio...")
    audio_data = recognizer.record(source) #Reads the audio data.
    print("Converting speech to text...")
    try:
        # Use Google's speech recognition API
        text = recognizer.recognize_google(audio_data) # Using Googl's free API to get Text
        print("Transcription:\n", text)
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio.")
    except sr.RequestError:
        print("Could not request results from the service.")

    
# Note: Only Need for one Libraries Which is Google Speech Recogniser Using it's Free API to get TEXT out of Speech 
# Also it is not that advance in order to get Text out of Noisy Voice I am not going that advance we can also use Wav2Vec for noisy voices
"""
I could Also implement this: : but avoided due to the complexity or may not work on testing team system =:
with sr.Microphone() as source:
    print("Speak something:")
    audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio)
    print("You said:", text)

"""