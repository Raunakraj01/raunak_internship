#Description: A small Program implementing already existing Speach recognition to avoid complexity 
import speech_recognition as sr #Imports the speech_recognition libraries but wait why as sr ?, this is just lib name we are deciding when will call it by this name

# Initialize recognizer
recognizer = sr.Recognizer() #we are creating a object from Recognizer class which is present in speech recoginition lib which we are storing in recognizer to use elsewhere, why storing ?

# Load the audio file
with sr.AudioFile("sample_audio.wav") as source: #What this do ?, with is just ensures the closure of file automatically after use, AudioFile is class where this audio file is being opend and being prepared to be converted to text, a variable source which will hold that audio
    print("Listening to audio...")
    audio_data = recognizer.record(source) #record() uses that source (the prepared audio file) to grab the audio into memory for processing
    print("Converting speech to text...")
    try:
        # Use Google's speech recognition API
        text = recognizer.recognize_google(audio_data) # This is the main part: it sends the recorded audio to Google Web Speech API, also returns text if recognised properly
        print("Transcription:\n", text)
    except sr.UnknownValueError: # for error hanlding if speech is not clear!
        print("Speech Recognition could not understand audio.")
    except sr.RequestError: # for error handling if network is not working properly!
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