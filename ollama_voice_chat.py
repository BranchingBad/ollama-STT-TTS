#!/usr/bin/env python3

"""
ollama_voice_chat.py

A simple, hands-free Python voice assistant that runs 100% locally.
This script uses openwakeword for wakeword detection, webrtcvad for silence
detection, OpenAI's Whisper for transcription, and Ollama for generative AI
responses.

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
import configparser # Added for config.ini reading
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

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the assistant, loads models, and sets up audio.
        """
        self.args: argparse.Namespace = args
        self.system_prompt = args.system_prompt # Store system prompt from args

        # Calculate derived audio settings
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_ms: int = args.pre_buffer_ms
        self.pre_buffer_size_chunks: int = int(self.pre_buffer_size_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        logging.info(f"Loading openwakeword model: {args.wakeword_model}...")
        # Load model using wakeword_model_paths and point to the .onnx file
        # Ensure the .onnx file exists where expected or adjust the path
        try:
            self.oww_model = Model(wakeword_model_paths=[f"{args.wakeword_model}.onnx"])
        except Exception as e:
            logging.error(f"Error loading openwakeword model '{args.wakeword_model}.onnx': {e}")
            logging.error("Ensure the model file exists in the current directory or provide the full path.")
            raise # Re-raise the exception to stop initialization if model fails

        # Whisper Model
        logging.info(f"Loading Whisper model: {args.whisper_model}...")
        try:
            self.whisper_model = whisper.load_model(args.whisper_model)
        except Exception as e:
            logging.error(f"Error loading Whisper model '{args.whisper_model}': {e}")
            logging.error("Ensure the model name is correct and you have enough memory.")
            raise

        # TTS Engine
        self.tts_engine = pyttsx3.init()
        self.tts_queue = queue.Queue()
        self.tts_stop_event = threading.Event()

        # Event to signal when TTS is active
        self.is_speaking_event = threading.Event()

        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None

        # Conversation history - Initialize with the configurable system prompt
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        logging.info(f"Using system prompt: '{self.system_prompt}'") # Log the prompt being used

    def _tts_worker(self) -> None:
        """Dedicated thread for processing TTS tasks."""
        while not self.tts_stop_event.is_set():
            text = None # Reset text for this loop iteration
            try:
                # Wait for text to speak, with a timeout
                text = self.tts_queue.get(timeout=1.0)

                if text is None: # Sentinel for stopping
                    break

                # Set event before speaking
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

                # Clear event after speaking (or error)
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
            # Ensure messages list starts with the system prompt
            if not self.messages or self.messages[0]['role'] != 'system':
                 logging.warning("Messages list did not start with system prompt. Re-adding.")
                 self.messages.insert(0, {'role': 'system', 'content': self.system_prompt})

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
                    logging.info("Processing voice input...") # Added info message
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
                    # Added IOError handling to the main read loop
                    audio_chunk = self.stream.read(CHUNK_SIZE)
                except IOError as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}")
                    logging.error("This may be due to a microphone disconnection. Stopping.")
                    break

                # Only process audio if TTS is not active
                if not self.is_speaking_event.is_set():
                    audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)

                    # Feed audio to openwakeword
                    prediction = self.oww_model.predict(audio_np_int16)

                    # Check if the desired wakeword score is high
                    wakeword_model_key = self.args.wakeword_model # Get the model name from args
                    if wakeword_model_key in prediction and prediction[wakeword_model_key] > self.args.wakeword_threshold:
                        logging.info(f"Wakeword '{self.args.wakeword}' detected!")

                        self.stream.stop_stream()  # Stop listening

                        # This call should BLOCK
                        self.speak("Yes?")
                        self.wait_for_speech()     # Wait for "Yes?" to finish

                        self.stream.start_stream() # Start listening for command

                        # Command Recording Loop
                        audio_data = self.record_command()

                        self.stream.stop_stream() # Stop while processing

                        if audio_data is None:
                            # This call should BLOCK
                            self.speak("I didn't hear anything.")
                            self.wait_for_speech()

                            logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                            self.stream.start_stream() # Restart for wakeword
                            continue # Go back to listening for wakeword

                        # Process the Command
                        logging.info("Transcribing audio...")
                        user_text = self.transcribe_audio(audio_data)

                        if user_text:
                            logging.info(f"You: {user_text}")

                            # Clean up punctuation and check for commands
                            user_prompt = user_text.lower().strip().rstrip(".,!?")

                            # Changed from startswith() to 'in' for flexibility
                            if "exit" in user_prompt or "goodbye" in user_prompt:
                                # This call should BLOCK
                                self.speak("Goodbye!")
                                self.wait_for_speech()
                                break

                            # More flexible command matching
                            if "new chat" in user_prompt or "reset chat" in user_prompt:
                                # This call should BLOCK
                                self.speak("Starting a new conversation.")
                                self.wait_for_speech()

                                # Reset history, but keep the system prompt
                                self.messages = [
                                    {'role': 'system', 'content': self.system_prompt}
                                ]
                                logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                                self.stream.start_stream() # Restart for wakeword
                                continue

                            # Get and speak the response
                            logging.info(f"Sending to {self.args.ollama_model}...")
                            ollama_reply = self.get_ollama_response(user_text)

                            # This call is NON-BLOCKING
                            self.speak(ollama_reply)

                            # Block only if it was an error message
                            if "I'm sorry" in ollama_reply:
                                self.wait_for_speech()

                        else:
                            # This call should BLOCK
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
    Parses arguments, reads config, and runs the VoiceAssistant.
    """
    # --- Read Config File ---
    config = configparser.ConfigParser()
    # Define fallback defaults for argparse, matching config.ini
    argparse_defaults = {
        'ollama_model': 'llama3',
        'whisper_model': 'base.en',
        'wakeword_model': 'hey_glados',
        'wakeword': 'hey glados',
        'wakeword_threshold': 0.6,
        'vad_aggressiveness': 2,
        'silence_seconds': 0.7,
        'listen_timeout': 6.0,
        'pre_buffer_ms': 400,
        'system_prompt': 'You are a helpful, concise voice assistant.',
    }

    try:
        if config.read('config.ini'): # Try reading the file
             logging.info("Loaded configuration from config.ini")
             # Update argparse defaults from the loaded config file
             if 'Models' in config:
                 argparse_defaults['ollama_model'] = config.get('Models', 'ollama_model', fallback=argparse_defaults['ollama_model'])
                 argparse_defaults['whisper_model'] = config.get('Models', 'whisper_model', fallback=argparse_defaults['whisper_model'])
                 argparse_defaults['wakeword_model'] = config.get('Models', 'wakeword_model', fallback=argparse_defaults['wakeword_model'])
             if 'Functionality' in config:
                 argparse_defaults['wakeword'] = config.get('Functionality', 'wakeword', fallback=argparse_defaults['wakeword'])
                 argparse_defaults['wakeword_threshold'] = config.getfloat('Functionality', 'wakeword_threshold', fallback=argparse_defaults['wakeword_threshold'])
                 argparse_defaults['vad_aggressiveness'] = config.getint('Functionality', 'vad_aggressiveness', fallback=argparse_defaults['vad_aggressiveness'])
                 argparse_defaults['silence_seconds'] = config.getfloat('Functionality', 'silence_seconds', fallback=argparse_defaults['silence_seconds'])
                 argparse_defaults['listen_timeout'] = config.getfloat('Functionality', 'listen_timeout', fallback=argparse_defaults['listen_timeout'])
                 argparse_defaults['pre_buffer_ms'] = config.getint('Functionality', 'pre_buffer_ms', fallback=argparse_defaults['pre_buffer_ms'])
                 argparse_defaults['system_prompt'] = config.get('Functionality', 'system_prompt', fallback=argparse_defaults['system_prompt'])
        else:
             logging.warning("config.ini not found or empty, using default settings defined in script.")
    except configparser.Error as e:
        logging.error(f"Error reading config.ini: {e}. Using default settings defined in script.")
        # Keep using the hardcoded argparse_defaults if config reading fails


    # --- Setup Argument Parser with Defaults (from config.ini or hardcoded) ---
    parser = argparse.ArgumentParser(
        description="Ollama STT-TTS Voice Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help
    )

    # Model settings
    parser.add_argument('--ollama-model',
                        type=str,
                        default=argparse_defaults['ollama_model'],
                        help='Ollama model (e.g., "llama3", "mistral")')
    parser.add_argument('--whisper-model',
                        type=str,
                        default=argparse_defaults['whisper_model'],
                        help='Whisper model (e.g., "tiny.en", "base.en")')
    parser.add_argument('--wakeword-model',
                        type=str,
                        default=argparse_defaults['wakeword_model'],
                        help='Openwakeword model name (e.g., "hey_glados"). Assumes .onnx file in cwd.')

    # Functionality settings
    parser.add_argument('--wakeword',
                        type=str,
                        default=argparse_defaults['wakeword'],
                        help='Wakeword phrase.')
    parser.add_argument('--wakeword-threshold',
                        type=float,
                        default=argparse_defaults['wakeword_threshold'],
                        help='Wakeword sensitivity (0.0-1.0).')
    parser.add_argument('--vad-aggressiveness',
                        type=int,
                        default=argparse_defaults['vad_aggressiveness'],
                        choices=[0, 1, 2, 3],
                        help='VAD aggressiveness (0=least, 3=most).')
    parser.add_argument('--silence-seconds',
                        type=float,
                        default=argparse_defaults['silence_seconds'],
                        help='Seconds of silence before stopping recording.')
    parser.add_argument('--listen-timeout',
                        type=float,
                        default=argparse_defaults['listen_timeout'],
                        help='Seconds to wait for speech before timeout.')
    parser.add_argument('--pre-buffer-ms',
                        type=int,
                        default=argparse_defaults['pre_buffer_ms'],
                        help='Milliseconds of audio to pre-buffer.')
    # Added system_prompt argument
    parser.add_argument('--system-prompt',
                        type=str,
                        default=argparse_defaults['system_prompt'],
                        help='The system prompt for the Ollama model.')

    args = parser.parse_args()


    # Configure logging (moved after parsing args in case log level is added later)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Print effective arguments (combination of config and command line)
    logging.info("Starting assistant with the following effective settings:")
    for arg, value in vars(args).items():
        # Truncate long system prompt for cleaner logging if necessary
        if arg == 'system_prompt' and len(str(value)) > 100:
             logging.info(f"  --{arg}: '{str(value)[:100]}...'")
        else:
             logging.info(f"  --{arg}: {value}")
    logging.info("-" * 20) # Separator


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