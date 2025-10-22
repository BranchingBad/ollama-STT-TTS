import ollama
import whisper
import pyttsx3
import pyaudio
import numpy as np
import webrtcvad
import argparse  # Import argparse for command-line arguments

# --- 1. Configuration ---

# Wakeword settings
WAKEWORD = "hey mycroft"  # The phrase to listen for
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

# --- 2. Command-Line Argument Parsing ---

# Set up argument parser
parser = argparse.ArgumentParser(description="Ollama STT-TTS Voice Assistant")
parser.add_argument('--ollama-model', 
                    type=str, 
                    default='llama3', 
                    help='The Ollama model to use (e.g., "llama3", "mistral", "phi3")')
parser.add_argument('--whisper-model', 
                    type=str, 
                    default='base.en', 
                    help='The Whisper model to use (e.g., "tiny.en", "base.en", "small.en")')
args = parser.parse_args()


# --- 3. Initialization ---
print("Loading models...")

# Wakeword Model
from openwakeword.model import Model
oww_model = Model(wakeword_models=[WAKEWORD_MODEL_NAME])

# Whisper Model
print(f"Loading Whisper model: {args.whisper_model}...")
whisper_model = whisper.load_model(args.whisper_model)

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

# --- 4. Helper Functions ---

def speak(text):
    """Speaks the given text using pyttsx3."""
    print(f"Assistant: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def transcribe_audio(audio_np):
    """Transcribes audio data (NumPy array) using Whisper."""
    try:
        # Transcribe the NumPy array directly
        result = whisper_model.transcribe(audio_np)
        return result['text']
    except Exception as e:
        print(f"Whisper transcription error: {e}")
        return ""

def get_ollama_response(text):
    """Gets a response from Ollama using the configured model."""
    try:
        response = ollama.chat(model=args.ollama_model, messages=[
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

def record_command(first_chunk):
    """
    Records audio from the user until silence is detected.
    Returns audio data as a 32-bit float NumPy array.
    """
    print("Listening for command...")
    frames = [first_chunk]  # Start with the chunk that triggered the wakeword
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

    # Combine all audio chunks
    audio_data = b''.join(frames)
    
    # Convert raw 16-bit audio data to a 32-bit float NumPy array
    # This is the format Whisper expects
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio_np

# --- 5. Main Loop ---
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
            # Pass the triggering chunk to start the recording
            audio_data = record_command(audio_chunk)
            
            # --- Process the Command ---
            user_text = transcribe_audio(audio_data)
            
            if user_text:
                print(f"You: {user_text}")
                
                # Check for exit commands
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
            
            print(f"\nReady! Listening for '{WAKEWORD}'...")

except KeyboardInterrupt:
    print("\nStopping assistant...")
finally:
    # --- 6. Cleanup ---
    stream.stop_stream()
    stream.close()
    audio.terminate()
