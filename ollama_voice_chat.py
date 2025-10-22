import ollama
import whisper
import pyttsx3
import pyaudio
import numpy as np
import webrtcvad
import wave
import tempfile
import os
from openwakeword.model import Model

# --- 1. Configuration ---
# --- Corrected Wakeword to match the model ---
WAKEWORD = "hey mycroft"  # The phrase to listen for
WAKEWORD_MODEL_NAME = "hey_mycroft_v0.1" # A pre-trained model from openwakeword

# --- Added Model Configuration ---
OLLAMA_MODEL = "llama3"      # Model to use for Ollama
WHISPER_MODEL = "base.en"    # Model to use for Whisper (e.g., "base.en", "small.en")

# Audio settings (must match for VAD and Whisper)
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                 # Mono
RATE = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS = 30       # 30ms chunks for VAD
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames
VAD_AGGRESSIVENESS = 2       # 0 (least aggressive) to 3 (most aggressive)
SILENCE_CHUNKS = 70          # Number of 30ms silent chunks to stop recording
                             # (70 chunks * 30ms/chunk = 2100ms = 2.1 seconds of silence)

# --- 2. Initialization ---
print("Loading models...")
# Wakeword Model
# --- Corrected model loading to let openwakeword handle downloads ---
oww_model = Model(wakeword_models=[WAKEWORD_MODEL_NAME])

# Whisper Model
# --- Using configured WHISPER_MODEL ---
whisper_model = whisper.load_model(WHISPER_MODEL)

# TTS Engine
tts_engine = pyttsx3.init()

# VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

# --- Corrected print statement to show the actual wakeword ---
print(f"\nReady! Listening for '{WAKEWORD}'...")

# --- 3. Helper Functions ---

def speak(text):
    """Speaks the given text using pyttsx3."""
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def transcribe_audio(file_path):
    """Transcribes audio file using Whisper."""
    try:
        result = whisper_model.transcribe(file_path)
        return result['text']
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""

# --- Improved error handling for Ollama ---
def get_ollama_response(text):
    """Gets a response from Ollama."""
    try:
        # --- Using configured OLLAMA_MODEL ---
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {'role': 'user', 'content': text}
        ])
        return response['message']['content']
    except ollama.ResponseError as e:
        print(f"Ollama Response Error: {e.error}")
        return "I'm sorry, I received an error from Ollama. Please check the console."
    except Exception as e:
        # This often catches connection errors
        print(f"Ollama error: {e}")
        return "I'm sorry, I couldn't connect to Ollama. Is the Ollama server running?"

def record_command():
    """Records audio from the user until silence is detected."""
    print("Listening for command...")
    frames = []
    silent_chunks = 0
    is_speaking = False

    while True:
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
        
        # Use VAD to check if the chunk contains speech
        is_speech = vad.is_speech(data, RATE)

        if is_speaking:
            if not is_speech:
                silent_chunks += 1
                if silent_chunks > SILENCE_CHUNKS:
                    print("Silence detected, processing...")
                    break
            else:
                # Reset silence counter if speech is detected again
                silent_chunks = 0
        elif is_speech:
            # Start counting silence only after speech has begun
            is_speaking = True
            silent_chunks = 0

    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        wf = wave.open(tmp_wav.name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        return tmp_wav.name

# --- 4. Main Loop ---
try:
    while True:
        # --- Wakeword Listening Loop ---
        audio_chunk = stream.read(CHUNK_SIZE)
        
        # Feed audio to openwakeword
        prediction = oww_model.predict(audio_chunk)
        
        # Check if the desired wakeword score is high
        if prediction[WAKEWORD_MODEL_NAME] > 0.5: # 0.5 is the threshold
            print(f"Wakeword '{WAKEWORD}' detected!")
            speak("Yes?")
            
            # --- Command Recording Loop ---
            audio_file_path = record_command()
            
            # --- Process the Command ---
            user_text = transcribe_audio(audio_file_path)
            os.remove(audio_file_path) # Clean up the temp file
            
            if user_text:
                print(f"You: {user_text}")
                
                # --- Corrected exit check for more robust matching ---
                user_prompt = user_text.lower().strip()
                # Check for exact matches, allowing for punctuation
                if user_prompt in ["exit", "exit.", "goodbye", "goodbye."]:
                    speak("Goodbye!")
                    break
                
                # Get and speak the response
                ollama_reply = get_ollama_response(user_text)
                speak(ollama_reply)
            else:
                speak("I'm sorry, I didn't catch that.")
            
            # --- Corrected print statement to show the actual wakeword ---
            print(f"\nReady! Listening for '{WAKEWORD}'...")

except KeyboardInterrupt:
    print("\nStopping assistant...")
finally:
    # --- 5. Cleanup ---
    stream.stop_stream()
    stream.close()
    audio.terminate()
