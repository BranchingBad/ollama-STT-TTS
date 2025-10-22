#!/usr/bin/env python3

"""
ollama_voice_chat.py

A simple, hands-free Python voice assistant that runs 100% locally.
This script uses openwakeword for wakeword detection, webrtcvad for silence
detection, OpenAI's Whisper for transcription, and Ollama for generative AI
responses.

Refactored into a class-based structure for improved state management
and to fix critical bugs.
"""

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

class VoiceAssistant:
    """
    Manages all components of the voice assistant:
    - Audio I/O and VAD
    - Wakeword detection (openwakeword)
    - Speech-to-Text (Whisper)
    - Language Model (Ollama)
    - Text-to-Speech (pyttsx3)
    """
    
    def __init__(self, args):
        """
        Initializes the assistant, loads models, and sets up audio.
        """
        self.args = args
        
        # Calculate derived audio settings
        self.silence_chunks = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_ms = 500 # 0.5 seconds pre-buffer
        self.pre_buffer_size_chunks = int(self.pre_buffer_size_ms / CHUNK_DURATION_MS)


        print("Loading models...")

        # Wakeword Model
        print(f"Loading openwakeword model: {args.wakeword_model}...")
        self.oww_model = Model(wakeword_models=[args.wakeword_model])

        # Whisper Model
        print(f"Loading Whisper model: {args.whisper_model}...")
        self.whisper_model = whisper.load_model(args.whisper_model)

        # TTS Engine
        self.tts_engine = pyttsx3.init()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream = None # Stream will be opened in run()
        
        # Conversation history
        self.messages = []

    def speak(self, text):
        """Speaks the given text using pyttsx3."""
        print(f"Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def transcribe_audio(self, audio_np):
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
            # Transcribe the NumPy array directly
            result = self.whisper_model.transcribe(audio_np)
            return result['text']
        except Exception as e:
            print(f"Whisper transcription error: {e}")
            return ""

    def get_ollama_response(self, text):
        """
        Gets a response from Ollama, maintaining conversation history.
        """
        # Append the new user message
        self.messages.append({'role': 'user', 'content': text})
        
        try:
            response = ollama.chat(model=self.args.ollama_model, messages=self.messages)
            
            # Append the assistant's response
            assistant_reply = response['message']['content']
            self.messages.append({'role': 'assistant', 'content': assistant_reply})
            
            return assistant_reply
            
        except ollama.ResponseError as e:
            print(f"Ollama Response Error: {e.error}")
            return "I'm sorry, I received an error from Ollama. Please check the console."
        except Exception as e:
            # This often catches connection errors
            print(f"Ollama error: {e}")
            return "I'm sorry, I couldn't connect to Ollama. Is the Ollama server running?"

    def record_command(self):
        """
        Records audio from the user until silence is detected.
        Uses a pre-buffer to catch the start of speech.
        Returns audio data as a 32-bit float NumPy array, or None if no speech is detected.
        """
        print("Listening for command...")
        frames = []
        silent_chunks = 0
        is_speaking = False
        
        # Store a small buffer of audio *before* speech starts
        pre_buffer = []
        timeout_chunks = self.pre_speech_timeout_chunks

        while True:
            try:
                data = self.stream.read(CHUNK_SIZE)
                is_speech = self.vad.is_speech(data, RATE)

                if is_speaking:
                    # User is actively speaking
                    frames.append(data)
                    if not is_speech:
                        silent_chunks += 1
                        if silent_chunks > self.silence_chunks:
                            print("Silence detected, processing...")
                            break
                    else:
                        silent_chunks = 0 # Reset silence counter
                
                elif is_speech:
                    # Speech has just started
                    print("Speech detected...")
                    is_speaking = True
                    frames.extend(pre_buffer) # Add the pre-speech buffer
                    frames.append(data)
                    pre_buffer.clear()
                
                else: 
                    # User is not speaking, and speech hasn't started
                    # Add to pre-buffer and keep it at size
                    pre_buffer.append(data)
                    if len(pre_buffer) > self.pre_buffer_size_chunks:
                        pre_buffer.pop(0)
                    
                    # Check for timeout
                    timeout_chunks -= 1
                    if timeout_chunks <= 0:
                        print("No speech detected, timing out.")
                        return None
                        
            except IOError as e:
                print(f"Error reading from audio stream: {e}")
                return None


        # Combine all audio chunks
        audio_data = b''.join(frames)
        
        # Convert raw 16-bit audio data to a 32-bit float NumPy array
        # This is the format Whisper expects
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_np

    def run(self):
        """
        The main loop of the assistant. Listens for wakeword, then
        records, transcribes, and responds.
        """
        try:
            # Open stream here
            self.stream = self.audio.open(format=FORMAT,
                                          channels=CHANNELS,
                                          rate=RATE,
                                          input=True,
                                          frames_per_buffer=CHUNK_SIZE)
            
            print(f"\nReady! Listening for '{self.args.wakeword}'...")
            
            while True:
                # --- Wakeword Listening Loop ---
                audio_chunk = self.stream.read(CHUNK_SIZE)
                
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16)

                # Feed audio to openwakeword
                prediction = self.oww_model.predict(audio_np)
                
                # Check if the desired wakeword score is high
                if prediction[self.args.wakeword_model] > self.args.wakeword_threshold:
                    print(f"Wakeword '{self.args.wakeword}' detected!")
                    
                    self.stream.stop_stream()  # Stop listening
                    self.speak("Yes?")         # Speak
                    self.stream.start_stream() # Start listening for command
                    
                    # --- Command Recording Loop ---
                    audio_data = self.record_command()
                    
                    self.stream.stop_stream() # Stop while processing
                    
                    if audio_data is None:
                        self.speak("I didn't hear anything.")
                        print(f"\nReady! Listening for '{self.args.wakeword}'...")
                        self.stream.start_stream() # Restart for wakeword
                        continue # Go back to listening for wakeword

                    # --- Process the Command ---
                    print("Transcribing audio...")
                    user_text = self.transcribe_audio(audio_data)
                    
                    if user_text:
                        print(f"You: {user_text}")
                        
                        # Clean up punctuation and check for commands
                        user_prompt = user_text.lower().strip().rstrip(".,!?")
                        
                        if user_prompt.startswith("exit") or user_prompt.startswith("goodbye"):
                            self.speak("Goodbye!")
                            break
                        
                        if user_prompt == "new chat" or user_prompt == "reset chat":
                            self.speak("Starting a new conversation.")
                            self.messages = [] # Clear history
                            print(f"\nReady! Listening for '{self.args.wakeword}'...")
                            self.stream.start_stream() # Restart for wakeword
                            continue

                        # Get and speak the response
                        print(f"Sending to {self.args.ollama_model}...")
                        ollama_reply = self.get_ollama_response(user_text)
                        self.speak(ollama_reply)
                    else:
                        self.speak("I'm sorry, I didn't catch that.")
                    
                    print(f"\nReady! Listening for '{self.args.wakeword}'...")
                    self.stream.start_stream() # Restart for wakeword

        except KeyboardInterrupt:
            print("\nStopping assistant...")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.cleanup()

    def cleanup(self):
        """Cleans up PyAudio resource."""
        print("Cleaning up resources...")
        if self.audio:
            self.audio.terminate()

def main():
    """
    Parses arguments and runs the VoiceAssistant.
    """
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
    parser.add_argument('--listen-timeout',
                        type=float,
                        default=5.0,
                        help='Seconds to wait for speech to start before timing out.')
    
    args = parser.parse_args()

    try:
        assistant = VoiceAssistant(args)
        assistant.run()
    except IOError as e:
        print(f"FATAL ERROR: Could not open audio stream.")
        print("Please check if a microphone is connected and permissions are correct.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Script Entry Point ---
if __name__ == "__main__":
    main()
