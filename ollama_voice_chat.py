#!/usr/bin/env python3

"""
ollama_voice_chat.py

A simple, hands-free Python voice assistant that runs 100% locally.
This script uses openwakeword for wakeword detection, webrtcvad for silence
detection, OpenAI's Whisper for transcription, and Ollama for generative AI
responses.

This updated version includes a non-blocking TTS system using a separate
thread and queue, allowing the assistant to listen for the wakeword
while speaking its response.

--- MODIFICATION ---
Includes threading.Event 'is_speaking_event' to prevent the assistant
from listening to its own voice, which could cause false wakeword
triggers.
"""

import ollama
import whisper
import pyttsx3
import pyaudio
import numpy as np
import webrtcvad
import argparse
import logging
import threading
import queue
from openwakeword.model import Model
from typing import List, Dict, Optional
import numpy.typing as npt

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
    - Text-to-Speech (pyttsx3 in a separate thread)
    """
    
    # --- IMPLEMENTS SUGGESTION 2 ---
    # System prompt is now a class constant
    SYSTEM_PROMPT = 'You are a helpful, concise voice assistant.'
    
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the assistant, loads models, and sets up audio.
        """
        self.args: argparse.Namespace = args
        
        # Calculate derived audio settings
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_ms: int = args.pre_buffer_ms
        self.pre_buffer_size_chunks: int = int(self.pre_buffer_size_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        logging.info(f"Loading openwakeword model: {args.wakeword_model}...")
        self.oww_model = Model(wakeword_models=[args.wakeword_model])

        # Whisper Model
        logging.info(f"Loading Whisper model: {args.whisper_model}...")
        self.whisper_model = whisper.load_model(args.whisper_model)

        # TTS Engine
        self.tts_engine = pyttsx3.init()
        self.tts_queue = queue.Queue()
        self.tts_stop_event = threading.Event()
        
        # --- ADDED: Event to signal when TTS is active ---
        self.is_speaking_event = threading.Event()
        
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Conversation history
        # --- ENHANCEMENT: Added a system prompt ---
        # --- IMPLEMENTS SUGGESTION 2 ---
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT}
        ]

    def _tts_worker(self) -> None:
        """Dedicated thread for processing TTS tasks."""
        text = None # Ensure text is defined in finally block
        while not self.tts_stop_event.is_set():
            try:
                # Wait for text to speak, with a timeout
                text = self.tts_queue.get(timeout=1.0)
                
                if text is None: # Sentinel for stopping
                    break
                
                # --- MODIFIED: Set event before speaking ---
                self.is_speaking_event.set()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
            except queue.Empty:
                continue # Just loop again if queue is empty
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
            finally:
                if text is not None:
                    self.tts_queue.task_done()
                
                # --- MODIFIED: Clear event after speaking (or error) ---
                self.is_speaking_event.clear()


    def speak(self, text: str) -> None:
        """Adds text to the TTS queue to be spoken by the worker thread."""
        logging.info(f"Assistant: {text}")
        self.tts_queue.put(text)

    def wait_for_speech(self) -> None:
        """Blocks until the TTS queue is empty and all speech is finished."""
        logging.info("Waiting for speech to finish...")
        self.tts_queue.join()
        logging.info("Speech finished.")

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
            # Transcribe the NumPy array directly
            result = self.whisper_model.transcribe(audio_np)
            return result.get('text', '')
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}")
            return ""

    def get_ollama_response(self, text: str) -> str:
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
            logging.error(f"Ollama Response Error: {e.error}")
            return "I'm sorry, I received an error from Ollama. Please check the console."
        except Exception as e:
            # This often catches connection errors
            logging.error(f"Ollama error: {e}")
            return "I'm sorry, I couldn't connect to Ollama. Is the Ollama server running?"

    def record_command(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Records audio from the user until silence is detected.
        Uses a pre-buffer to catch the start of speech.
        Returns audio data as a 32-bit float NumPy array, or None if no speech is detected.
        """
        logging.info("Listening for command...")
        frames: List[bytes] = []
        silent_chunks = 0
        is_speaking = False
        
        # Store a small buffer of audio *before* speech starts
        pre_buffer: List[bytes] = []
        timeout_chunks = self.pre_speech_timeout_chunks

        while True:
            try:
                if not self.stream:
                    logging.error("Audio stream is not open.")
                    return None
                    
                data = self.stream.read(CHUNK_SIZE)
                is_speech = self.vad.is_speech(data, RATE)

                if is_speaking:
                    # User is actively speaking
                    frames.append(data)
                    if not is_speech:
                        silent_chunks += 1
                        if silent_chunks > self.silence_chunks:
                            logging.info("Silence detected, processing...")
                            break
                    else:
                        silent_chunks = 0 # Reset silence counter
                
                elif is_speech:
                    # Speech has just started
                    logging.info("Speech detected...")
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
                        logging.warning("No speech detected, timing out.")
                        return None
                        
            except IOError as e:
                logging.error(f"Error reading from audio stream: {e}")
                return None


        # Combine all audio chunks
        audio_data: bytes = b''.join(frames)
        
        # Convert raw 16-bit audio data to a 32-bit float NumPy array
        # This is the format Whisper expects
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        return audio_np

    def run(self) -> None:
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
            
            logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
            
            while True:
                if not self.stream:
                    logging.error("Audio stream was unexpectedly closed.")
                    break
                
                try:
                    # --- IMPLEMENTS SUGGESTION 1 ---
                    # Added IOError handling to the main read loop
                    audio_chunk = self.stream.read(CHUNK_SIZE)
                except IOError as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}")
                    logging.error("This may be due to a microphone disconnection. Stopping.")
                    break

                # --- MODIFIED: Only process audio if TTS is not active ---
                if not self.is_speaking_event.is_set():
                    audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

                    # Feed audio to openwakeword
                    prediction = self.oww_model.predict(audio_np_int16)
                    
                    # Check if the desired wakeword score is high
                    if prediction[self.args.wakeword_model] > self.args.wakeword_threshold:
                        logging.info(f"Wakeword '{self.args.wakeword}' detected!")
                        
                        self.stream.stop_stream()  # Stop listening
                        
                        # --- This call should BLOCK ---
                        self.speak("Yes?")
                        self.wait_for_speech()     # Wait for "Yes?" to finish
                        
                        self.stream.start_stream() # Start listening for command
                        
                        # --- Command Recording Loop ---
                        audio_data = self.record_command()
                        
                        self.stream.stop_stream() # Stop while processing
                        
                        if audio_data is None:
                            # --- This call should BLOCK ---
                            self.speak("I didn't hear anything.")
                            self.wait_for_speech()

                            logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                            self.stream.start_stream() # Restart for wakeword
                            continue # Go back to listening for wakeword

                        # --- Process the Command ---
                        logging.info("Transcribing audio...")
                        user_text = self.transcribe_audio(audio_data)
                        
                        if user_text:
                            logging.info(f"You: {user_text}")
                            
                            # Clean up punctuation and check for commands
                            user_prompt = user_text.lower().strip().rstrip(".,!?")
                            
                            # --- IMPLEMENTS SUGGESTION 3 ---
                            # Changed from startswith() to 'in' for flexibility
                            if "exit" in user_prompt or "goodbye" in user_prompt:
                                # --- This call should BLOCK ---
                                self.speak("Goodbye!")
                                self.wait_for_speech()
                                break
                            
                            # --- ENHANCEMENT: More flexible command matching ---
                            if "new chat" in user_prompt or "reset chat" in user_prompt:
                                # --- This call should BLOCK ---
                                self.speak("Starting a new conversation.")
                                self.wait_for_speech()
                                
                                # Reset history, but keep the system prompt
                                # --- IMPLEMENTS SUGGESTION 2 ---
                                self.messages = [
                                    {'role': 'system', 'content': self.SYSTEM_PROMPT}
                                ]
                                logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                                self.stream.start_stream() # Restart for wakeword
                                continue

                            # Get and speak the response
                            logging.info(f"Sending to {self.args.ollama_model}...")
                            ollama_reply = self.get_ollama_response(user_text)
                            
                            # --- This call is NON-BLOCKING ---
                            self.speak(ollama_reply)

                            # Block only if it was an error message
                            if "I'm sorry" in ollama_reply:
                                self.wait_for_speech()

                        else:
                            # --- This call should BLOCK ---
                            logging.warning("Transcription failed.")
                            self.speak("I'm sorry, I didn't catch that.")
                            self.wait_for_speech()
                        
                        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                        self.stream.start_stream() # Restart for wakeword

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up PyAudio and stops the TTS thread."""
        logging.info("Cleaning up resources...")
        
        # Signal TTS thread to stop
        self.tts_stop_event.set()
        self.tts_queue.put(None) # Add sentinel value to unblock queue.get()
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0) # Wait for thread to finish
            
        if self.audio:
            self.audio.terminate()

def main() -> None:
    """
    Parses arguments and runs the VoiceAssistant.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

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
    # --- New Argument ---
    parser.add_argument('--pre-buffer-ms',
                        type=int,
                        default=500,
                        help='Milliseconds of audio to pre-buffer before speech is detected (default: 500).')
    
    args = parser.parse_args()

    try:
        assistant = VoiceAssistant(args)
        assistant.run()
    except IOError as e:
        logging.critical("FATAL ERROR: Could not open audio stream.")
        logging.critical("Please check if a microphone is connected and permissions are correct.")
        logging.critical(f"Details: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

# --- Script Entry Point ---
if __name__ == "__main__":
    main()