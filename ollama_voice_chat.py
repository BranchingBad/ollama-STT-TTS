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
WAKEWORD = "hey ollama"  # The phrase to listen for
WAKEWORD_MODEL_NAME = "hey_mycroft_v0.1" # A pre-trained model from openwakeword

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
oww_model = Model(wakeword_models=[f"models/{WAKEWORD_MODEL_NAME}.onnx"])

# Whisper Model
whisper_model = whisper.load_model("base.en")

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

def get_ollama_response(text):
    """Gets a response from Ollama."""
    try:
        response = ollama.chat(model='llama3', messages=[
            {'role': 'user', 'content': text}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I'm sorry, I had trouble processing that."

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
                # Check for exit command
                if "exit" in user_text.lower().strip() or "goodbye" in user_text.lower().strip():
                    speak("Goodbye!")
                    break
                
                # Get and speak the response
                ollama_reply = get_ollama_response(user_text)
                speak(ollama_reply)
            else:
                speak("I'm sorry, I didn't catch that.")
            
            print(f"\nReady! Listening for '{WAKEWORD}'...")

except KeyboardInterrupt:
    print("\nStopping assistant...")
finally:
    # --- 5. Cleanup ---
    stream.stop_stream()
    stream.close()
    audio.terminate()
