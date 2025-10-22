import ollama
import whisper
import pyttsx3
import pyaudio
import numpy as np
import webrtcvad
import argparse
from openwakeword.model import Model

# --- 1. Audio Settings (Constants) ---
# These are generally fixed based on hardware and model requirements
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                 # Mono
RATE = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS = 30       # 30ms chunks for VAD
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames per chunk

# --- 2. Command-Line Argument Parsing ---
parser = argparse.ArgumentParser(description="Ollama STT-TTS Voice Assistant")

# Model settings
parser.add_argument('--ollama-model', 
                    type=str, 
                    default='llama3', 
                    help='The Ollama model to use (e.g., "llama3", "mistral", "phi3")')
parser.add_argument('--whisper-model', 
                    type=str, 
                    default='base.en', 
                    help='The Whisper model to use (e.g., "tiny.en", "base.en", "small.en")')
parser.add_argument('--wakeword-model', 
                    type=str, 
                    default='hey_mycroft_v0.1', 
                    help='The openwakeword model to use (e.g., "hey_mycroft_v0.1")')

# Functionality settings
parser.add_argument('--wakeword', 
                    type=str, 
                    default='hey mycroft', 
                    help='The wakeword phrase to listen for.')
parser.add_argument('--wakeword-threshold', 
                    type=float, 
                    default=0.5, 
                    help='Wakeword detection sensitivity (0.0 to 1.0).')
parser.add_argument('--vad-aggressiveness', 
                    type=int, 
                    default=2, 
                    choices=[0, 1, 2, 3],
                    help='VAD aggressiveness (0=least, 3=most aggressive).')
parser.add_argument('--silence-seconds', 
                    type=float, 
                    default=2.0, 
                    help='Seconds of silence to wait before stopping recording.')

# Global args (will be populated in main)
args = None
SILENCE_CHUNKS = None
PRE_SPEECH_TIMEOUT_CHUNKS = None # Calculated in main

# --- 3. Global Initialization (Models and Services) ---
# These are loaded once and accessed by helper functions
oww_model = None
whisper_model = None
tts_engine = None
vad = None
audio = None
stream = None


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

def record_command(initial_chunk):
    """
    Records audio from the user until silence is detected.
    Starts with the initial chunk that triggered the wakeword.
    Returns audio data as a 32-bit float NumPy array, or None if no speech is detected.
    """
    print("Listening for command...")
    frames = [initial_chunk]  # Start with the chunk that triggered the wakeword
    silent_chunks = 0
    is_speaking = False
    pre_speech_chunks = 0 # Counter for timeout before speech starts

    while True:
        try:
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
            else:
                # --- IMPROVEMENT: Pre-speech timeout ---
                # If no speech has started, count chunks towards a timeout
                pre_speech_chunks += 1
                if pre_speech_chunks > PRE_SPEECH_TIMEOUT_CHUNKS:
                    print("No speech detected, timing out.")
                    return None # Return None to indicate no audio was captured
        except IOError as e:
            # This can happen if the audio stream is interrupted
            print(f"Error reading from audio stream: {e}")
            return None


    # Combine all audio chunks
    audio_data = b''.join(frames)
    
    # Convert raw 16-bit audio data to a 32-bit float NumPy array
    # This is the format Whisper expects
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
    return audio_np

# --- 5. Main Loop ---
def main():
    # Make globals accessible
    global args, SILENCE_CHUNKS, PRE_SPEECH_TIMEOUT_CHUNKS
    global oww_model, whisper_model, tts_engine, vad, audio, stream

    args = parser.parse_args()
    
    # Calculate the number of silent chunks needed based on the duration
    SILENCE_CHUNKS = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
    # Calculate a 5-second "pre-speech" timeout
    PRE_SPEECH_TIMEOUT_CHUNKS = int(5.0 * 1000 / CHUNK_DURATION_MS)


    # --- Initialization ---
    print("Loading models...")

    # Wakeword Model
    print(f"Loading openwakeword model: {args.wakeword_model}...")
    oww_model = Model(wakeword_models=[args.wakeword_model])

    # Whisper Model
    print(f"Loading Whisper model: {args.whisper_model}...")
    whisper_model = whisper.load_model(args.whisper_model)

    # TTS Engine
    tts_engine = pyttsx3.init()

    # VAD
    vad = webrtcvad.Vad(args.vad_aggressiveness)

    # PyAudio
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK_SIZE)
    except IOError as e:
        print(f"FATAL ERROR: Could not open audio stream.")
        print("Please check if a microphone is connected and permissions are correct.")
        print(f"Details: {e}")
        return # Exit the script

    print(f"\nReady! Listening for '{args.wakeword}'...")

    try:
        while True:
            # --- Wakeword Listening Loop ---
            audio_chunk = stream.read(CHUNK_SIZE)
            
            # --- CORRECTION: Convert bytes to NumPy array for openwakeword ---
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

            # Feed audio to openwakeword
            prediction = oww_model.predict(audio_np)
            
            # Check if the desired wakeword score is high
            if prediction[args.wakeword_model] > args.wakeword_threshold:
                print(f"Wakeword '{args.wakeword}' detected!")
                speak("Yes?")
                
                # --- Command Recording Loop ---
                # Record audio *after* the wakeword (and include the trigger chunk)
                audio_data = record_command(audio_chunk)
                
                # --- CORRECTION: Handle timeout from record_command ---
                if audio_data is None:
                    speak("I didn't hear anything.")
                    print(f"\nReady! Listening for '{args.wakeword}'...")
                    continue # Go back to listening for wakeword

                # --- Process the Command ---
                print("Transcribing audio...")
                user_text = transcribe_audio(audio_data)
                
                if user_text:
                    print(f"You: {user_text}")
                    
                    # --- IMPROVEMENT: Robust exit command check ---
                    # Clean up punctuation and check for start of phrase
                    user_prompt = user_text.lower().strip().rstrip(".,!?")
                    if user_prompt.startswith("exit") or user_prompt.startswith("goodbye"):
                        speak("Goodbye!")
                        break
                    
                    # Get and speak the response
                    print(f"Sending to {args.ollama_model}...")
                    ollama_reply = get_ollama_response(user_text)
                    speak(ollama_reply)
                else:
                    speak("I'm sorry, I didn't catch that.")
                
                print(f"\nReady! Listening for '{args.wakeword}'...")

    except KeyboardInterrupt:
        print("\nStopping assistant...")
    finally:
        # --- 6. Cleanup ---
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()

# --- 7. Script Entry Point ---
if __name__ == "__main__":
    main()
