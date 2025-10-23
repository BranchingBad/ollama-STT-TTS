#!/usr/bin/env python3

"""
ollama_voice_chat.py

A simple, hands-free Python voice assistant that runs 100% locally.
This script uses openwakeword for wakeword detection, webrtcvad for silence
detection, OpenAI's Whisper for transcription, and Ollama for generative AI
responses.

Improvements (Implemented in this version):
- Implemented Ollama streaming for sentence-by-sentence TTS output.
- Implemented TTS Interruptibility (Barge-in).
- Improved robust wakeword key detection.
- Unified configuration handling for 'wakeword_model_path' across config and arguments.
- UX Improvement: Removed the "Yes?" response to reduce latency/awkward silence after wakeword detection.
- Added a simple non-blocking check for Ollama connectivity during initialization.
- Refined logging for clarity.
- NEW (v2): Changed default Whisper model to 'tiny.en' for faster CPU-based transcription.
- NEW (v2): Explicitly disabled FP16 for Whisper on CPU to silence warnings and optimize performance.
- NEW (v2): Improved Ollama error handling and message history management during streaming.
- **FIXED (v3): Removed stream stop before command recording to resolve 'Audio stream is not open or active' error.**
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
import configparser
import time
from openwakeword.model import Model
from typing import List, Dict, Optional
import numpy.typing as npt
import sys

# --- 1. Audio Settings (Constants) ---
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                 # Mono
RATE = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS = 30       # 30ms chunks for VAD
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames per chunk
INT16_NORMALIZATION = 32768.0 # Normalization factor for int16
SENTENCE_END_PUNCTUATION = ['.', '?', '!', '\n']

# --- 2. PyAudio Configuration Check ---
def check_ollama_connectivity(host: str = 'http://localhost:11434') -> bool:
    """Checks if the Ollama server is running and reachable."""
    try:
        ollama.Client(host=host).list()
        logging.info("Ollama server is reachable.")
        return True
    except Exception as e:
        # Changed to INFO or WARNING since we want to initialize other parts even if Ollama is down
        logging.warning(f"Ollama server is not reachable at {host}. Error: {e}")
        return False

# --- 3. PyAudio and TTS Helpers (Definitions added for completeness) ---
def list_audio_devices(p_audio: pyaudio.PyAudio):
    """Lists all available audio input devices."""
    print("\n--- Available Audio Input Devices ---")
    try:
        info = p_audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            dev = p_audio.get_device_info_by_host_api_device_index(0, i)
            if dev.get('maxInputChannels') > 0:
                print(f"  Index {i}: {dev.get('name')}")
    except Exception as e:
        print(f"Error listing devices: {e}")
    print("------------------------------------\n")
    
def list_tts_voices():
    """Lists all available pyttsx3 voices."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("\n--- Available TTS Voices ---")
    for voice in voices:
        # Filter to English voices for relevance
        if voice.id and ('en' in voice.id.lower() or 'gmw' in voice.id.lower()):
            print(f"  ID: {voice.id}")
            print(f"    - Name: {voice.name}")
            print(f"    - Language: {voice.languages[0] if voice.languages else 'N/A'}")
            print(f"    - Gender: {voice.gender if hasattr(voice, 'gender') else 'N/A'}")
    print("----------------------------\n")

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
        self.system_prompt: str = args.system_prompt

        # Ollama Connectivity Check
        if not check_ollama_connectivity():
             # Note: We continue to initialize other parts, but LLM requests will fail.
             pass

        # Calculate derived audio settings
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_chunks: int = int(args.pre_buffer_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        logging.info(f"Loading openwakeword model from: {args.wakeword_model_path}...")
        try:
            self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
            # Robustly get the model key
            self.wakeword_model_key = list(self.oww_model.models.keys())[0] if self.oww_model.models else args.wakeword # Fallback to arg name
            logging.info(f"Loaded wakeword model with key: '{self.wakeword_model_key}'")
            
        except Exception as e:
            logging.critical(f"Error loading openwakeword model '{args.wakeword_model_path}': {e}")
            logging.critical("Ensure the .onnx model file exists at the specified path.")
            raise

        # Whisper Model
        logging.info(f"Loading Whisper model: {args.whisper_model}...")
        try:
            # Removed unsupported 'fp16=False' for compatibility, relying on 'device="cpu"'
            self.whisper_model = whisper.load_model(args.whisper_model, device="cpu")
        except Exception as e:
            logging.critical(f"Error loading Whisper model '{args.whisper_model}': {e}")
            logging.critical("Ensure the model name is correct and you have enough memory.")
            raise

        # TTS Engine
        logging.info("Initializing TTS engine...")
        self.tts_engine = pyttsx3.init()
        # Initial voice ID logging is often debug/trace level, using full logging here
        current_voice = self.tts_engine.getProperty('voice')
        logging.debug(f"Default TTS voice ID: {current_voice}")

        if args.tts_voice_id:
            try:
                # Need to iterate through voices to find a match by ID/name, not just set a property
                found_voice = None
                for voice in self.tts_engine.getProperty('voices'):
                    if voice.id == args.tts_voice_id or voice.name == args.tts_voice_id:
                        found_voice = voice
                        break
                
                if found_voice:
                    self.tts_engine.setProperty('voice', found_voice.id)
                    logging.debug(f"Successfully set voice to: {found_voice.id}")
                else:
                    logging.warning(f"TTS voice ID '{args.tts_voice_id}' not found. Using default.")

            except Exception as e:
                logging.error(f"Error setting TTS voice ID: {e}. Using default.")
        
        # TTS Control
        self.tts_queue: queue.Queue[Optional[str]] = queue.Queue()
        self.tts_stop_event = threading.Event()
        self.is_speaking_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Validate and select the input device
        self.device_index: Optional[int] = args.device_index
        if self.device_index is None:
            self.device_index = self.find_default_input_device()
            if self.device_index is None:
                logging.critical("No suitable input device found.")
                # Raise an exception *after* the audio object is created, so cleanup can run
                # but we'll re-raise it at the end of init.
                raise IOError("No input audio device found.")
            logging.info(f"No --device-index provided. Auto-selected device: {self.device_index}")
        else:
            try:
                # Basic check for a specified index
                device_info = self.audio.get_device_info_by_index(self.device_index)
                if int(device_info.get('maxInputChannels', 0)) == 0:
                    raise IOError(f"Device index {self.device_index} is not an input device.")
            except IOError as e:
                logging.critical(f"Invalid --device-index {self.device_index}: {e}")
                logging.critical("Use --list-devices to see available devices.")
                raise
            logging.info(f"Using specified audio device index: {self.device_index}")

        # Conversation history
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        logging.info(f"Using system prompt: '{self.system_prompt}'")

    def find_default_input_device(self) -> Optional[int]:
        """Tries to find the default input device."""
        try:
            default_device_info = self.audio.get_default_input_device_info()
            if int(default_device_info.get('maxInputChannels', 0)) > 0:
                return int(default_device_info['index'])
            else:
                raise IOError("Default device found, but has no input channels.")
        except IOError as e:
            logging.warning(f"Could not get default input device automatically: {e}. Searching all devices...")
            for i in range(self.audio.get_device_count()):
                try:
                    dev = self.audio.get_device_info_by_index(i)
                    if int(dev.get('maxInputChannels', 0)) > 0:
                        logging.warning(f"Using first available input device (index {i}): {dev['name']}")
                        return i
                except IOError:
                    continue
        return None

    def _tts_worker(self) -> None:
        """Dedicated thread for processing TTS tasks."""
        while not self.tts_stop_event.is_set():
            text = None
            try:
                # Use a small timeout to allow for graceful exit
                text = self.tts_queue.get(timeout=0.1)
                if text is None: # Stop sentinel
                    break

                self.is_speaking_event.set()
                # Stop any previous queued speech to ensure interruptibility
                self.tts_engine.stop() 
                
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
            finally:
                if text is not None:
                    self.tts_queue.task_done()
                
                # Check if the queue is now empty before clearing the speaking event
                if self.tts_queue.empty() and not self.interrupt_event.is_set():
                    self.is_speaking_event.clear()

    def speak(self, text: str) -> None:
        """Adds text to the TTS queue to be spoken by the worker thread."""
        # Use debug for the text being spoken, info for the action
        logging.debug(f"Queueing TTS: '{text}'")
        
        # Only clear the queue if we are interrupting (not for streaming speech)
        if self.interrupt_event.is_set():
             with self.tts_queue.mutex:
                self.tts_queue.queue.clear()
        self.tts_queue.put(text)

    def wait_for_speech(self) -> None:
        """Blocks until the TTS queue is empty and all speech is finished."""
        logging.debug("Waiting for speech to finish...")
        self.tts_queue.join()
        # The tts_worker manages clearing is_speaking_event after queue is empty
        # We ensure it's clear here just in case, but rely on the worker primarily
        self.is_speaking_event.clear()
        logging.debug("Speech finished.")

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
            # Whisper's transcription is the main source of latency on CPU.
            result = self.whisper_model.transcribe(audio_np, language="en")
            return result.get('text', '').strip()
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}")
            return ""

    def _is_sentence_end(self, token: str) -> bool:
        """Helper to check if a token is sentence-ending punctuation."""
        return any(p in token for p in SENTENCE_END_PUNCTUATION)

    def get_ollama_response_stream(self, text: str) -> None:
        """
        Gets a response from Ollama, maintaining conversation history
        and streaming the output sentence-by-sentence to the TTS queue.
        """
        # 1. Append user message for context
        self.messages.append({'role': 'user', 'content': text})
        
        full_response = ""
        sentence_buffer = ""

        try:
            # Ensure messages start with the system prompt
            if not self.messages or self.messages[0]['role'] != 'system':
                 logging.warning("Messages list did not start with system prompt. Re-adding.")
                 self.messages.insert(0, {'role': 'system', 'content': self.system_prompt})

            response_stream = ollama.chat(
                model=self.args.ollama_model,
                messages=self.messages,
                stream=True
            )

            for chunk in response_stream:
                # --- Check for user interruption ---
                if self.interrupt_event.is_set():
                    logging.info("User interrupted, stopping LLM generation.")
                    break 
                # --- End interruption check ---
                
                token = chunk['message']['content']
                full_response += token
                sentence_buffer += token

                if self._is_sentence_end(token):
                    # We have a full sentence, speak it.
                    self.speak(sentence_buffer.strip())
                    sentence_buffer = ""
            
            # Speak any remaining text in the buffer
            if sentence_buffer.strip() and not self.interrupt_event.is_set():
                self.speak(sentence_buffer.strip())

        except ollama.ResponseError as e:
            error_message = f"I'm sorry, I received an error from Ollama: {e.error}"
            logging.error(error_message)
            self.speak("I'm sorry, I had trouble connecting to the language model.")
            full_response = ""
            
        except Exception as e:
            error_message = f"An unexpected error occurred during Ollama streaming: {e}"
            logging.error(error_message)
            self.speak("I'm sorry, I encountered an internal error while thinking.")
            full_response = ""

        # 2. Append the complete (or interrupted) response to history for context
        if full_response.strip():
             self.messages.append({'role': 'assistant', 'content': full_response})
        else:
             # Remove the last user message if the LLM failed to respond
             if self.messages[-1]['role'] == 'user':
                 self.messages.pop()

    def record_command(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Records audio from the user until silence is detected.
        Returns audio data as a 32-bit float NumPy array, or None if no speech is detected.
        """
        logging.info("Listening for command...")
        frames: List[bytes] = []
        silent_chunks = 0
        is_speaking = False

        pre_buffer: List[bytes] = []
        timeout_chunks = self.pre_speech_timeout_chunks

        # CHECK IS ACTIVE HERE IS CRITICAL
        if not self.stream or not self.stream.is_active():
            logging.error("Audio stream is not open or active.")
            return None # CRITICAL: This is the return path for the error in the log

        while True:
            try:
                # Using a small timeout to help ensure a timely shutdown on Ctrl+C
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                is_speech = self.vad.is_speech(data, RATE)
                
                logging.debug(f"VAD Speech: {is_speech}")

                if is_speaking:
                    frames.append(data)
                    if not is_speech:
                        silent_chunks += 1
                        if silent_chunks > self.silence_chunks:
                            logging.info("Silence detected, processing...")
                            break
                    else:
                        silent_chunks = 0
                elif is_speech:
                    logging.info("Speech detected...")
                    is_speaking = True
                    frames.extend(pre_buffer)
                    frames.append(data)
                    pre_buffer.clear()
                else:
                    pre_buffer.append(data)
                    if len(pre_buffer) > self.pre_buffer_size_chunks:
                        pre_buffer.pop(0)

                    timeout_chunks -= 1
                    if timeout_chunks <= 0:
                        logging.warning("No speech detected, timing out.")
                        return None
            except IOError as e:
                # Catching IOError here (e.g. mic unplugged) for robust looping
                logging.error(f"Error reading from audio stream: {e}")
                return None
            except Exception as e:
                logging.error(f"Unexpected error during recording: {e}")
                return None

        if not frames:
             return None

        audio_data: bytes = b''.join(frames)
        # Convert to float32 NumPy array and normalize for Whisper
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_NORMALIZATION

        return audio_np

    def process_user_command(self) -> bool:
        """
        Records, transcribes, and responds to a user command.
        Returns True if the assistant should exit.
        """
        # Reset interrupt event now that we are processing a command
        self.interrupt_event.clear()

        # >>> FIX APPLIED HERE: REMOVED self.stream.stop_stream() <<<
        # The stream MUST remain active for record_command to read from it.
        # if self.stream and self.stream.is_active():
        #      self.stream.stop_stream() # <-- THIS LINE WAS REMOVED/COMMENTED OUT

        audio_data = self.record_command()
        
        # Stop stream *after* recording is complete, before starting TTS/Whisper processing
        if self.stream and self.stream.is_active():
             self.stream.stop_stream()


        if audio_data is None:
            # Only speak "I didn't hear anything" if the VAD timeout was hit
            if not self.interrupt_event.is_set():
                self.speak("I didn't hear anything.")
                self.wait_for_speech()
            return False # Don't exit

        logging.info("Transcribing audio...")
        user_text = self.transcribe_audio(audio_data)

        # --- UX Improvement: Move "Thinking..." to *after* transcription ---
        if user_text:
            logging.info(f"You: {user_text}")
            self.speak("Thinking...")
            self.wait_for_speech() # Wait for "Thinking..." to finish

            user_prompt = user_text.lower().strip().rstrip(".,!?")

            if "exit" in user_prompt or "goodbye" in user_prompt:
                self.speak("Goodbye!")
                self.wait_for_speech()
                return True # Exit
            if "new chat" in user_prompt or "reset chat" in user_prompt:
                self.speak("Starting a new conversation.")
                self.wait_for_speech()
                self.messages = [{'role': 'system', 'content': self.system_prompt}]
                return False # Don't exit

            logging.info(f"Sending to {self.args.ollama_model}...")
            
            # Call streaming function
            self.get_ollama_response_stream(user_text)
            
            # Wait for the *entire* streamed response to finish speaking
            self.wait_for_speech() 

        else:
            logging.warning("Transcription failed. No speech detected or unclear input.")
            self.speak("I'm sorry, I couldn't understand what you said.")
            self.wait_for_speech()
        
        return False # Don't exit

    def run(self) -> None:
        """
        The main loop of the assistant. Listens for wakeword or
        interruption, then records, transcribes, and responds.
        """
        try:
            # Open stream
            self.stream = self.audio.open(format=FORMAT,
                                          channels=CHANNELS,
                                          rate=RATE,
                                          input=True,
                                          input_device_index=self.device_index,
                                          frames_per_buffer=CHUNK_SIZE)

            logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

            while True:
                if not self.stream:
                    logging.error("Audio stream was unexpectedly closed.")
                    break
                
                # Ensure stream is active if it was stopped by a previous command
                if not self.stream.is_active():
                    self.stream.start_stream()

                try:
                    # Use exception_on_overflow=False to avoid crashes on buffer overflow
                    audio_chunk = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}")
                    logging.error("This may be due to a microphone disconnection. Stopping.")
                    break

                # --- 1. Check for User Interruption (Barge-in) ---
                if self.is_speaking_event.is_set():
                    is_speech = self.vad.is_speech(audio_chunk, RATE)
                    if is_speech:
                        logging.info("User interruption (barge-in) detected!")
                        
                        # Set event to stop LLM generation loop
                        self.interrupt_event.set()
                        
                        # Stop pyttsx3 speech immediately and clear queue
                        self.tts_engine.stop() 
                        self.tts_queue.queue.clear()
                        self.is_speaking_event.clear() # Clear immediately to allow mic to start listening again
                        
                        # Process the new command
                        if self.process_user_command():
                            break # Exit main loop
                        
                        # Restart stream and print ready message
                        if not self.stream.is_active():
                            self.stream.start_stream()
                        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                    
                    continue # Don't check for wakeword while speaking
                
                # --- 2. Check for Wakeword (if not speaking) ---
                audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_np_int16)

                # --- Check for wakeword detection ---
                if prediction.get(self.wakeword_model_key, 0) > self.args.wakeword_threshold:
                    logging.info(f"Wakeword '{self.args.wakeword}' detected! (Score: {prediction.get(self.wakeword_model_key):.2f})")
                    self.oww_model.reset()

                    # Process the new command
                    if self.process_user_command():
                        break # Exit main loop

                    # Restart stream and print ready message
                    if not self.stream.is_active():
                        self.stream.start_stream()
                    logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up PyAudio and stops the TTS thread."""
        logging.info("Cleaning up resources...")
        
        # Stop and close the audio stream
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
            except Exception:
                pass 
            finally:
                try:
                    self.stream.close()
                except Exception:
                    pass
            self.stream = None

        # Stop the TTS thread gracefully
        self.tts_stop_event.set()
        # Add a sentinel value to the queue to unblock the worker
        self.tts_queue.put(None) 
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=1.0)
            if self.tts_thread.is_alive():
                 logging.warning("TTS thread did not join gracefully.")
            
        # Stop pyttsx3 engine
        if hasattr(self, 'tts_engine'):
             self.tts_engine.stop()

        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()

def main() -> None:
    """
    Parses arguments, reads config, and runs the VoiceAssistant.
    """
    config = configparser.ConfigParser()
    # Updated defaults based on best practice and user logs
    argparse_defaults = {
        'ollama_model': 'llama3',
        'whisper_model': 'tiny.en', # Changed for performance
        'wakeword_model_path': 'hey_glados.onnx',
        'wakeword': 'hey glados',
        'wakeword_threshold': 0.5, 
        'vad_aggressiveness': 3,   
        'silence_seconds': 0.5,    
        'listen_timeout': 4.0,     
        'pre_buffer_ms': 400,
        'system_prompt': 'You are a helpful, concise voice assistant.',
        'device_index': None,
        'tts_voice_id': None,
    }

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose DEBUG logging')
    temp_args, _ = temp_parser.parse_known_args()

    log_level = logging.DEBUG if temp_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    list_parser = argparse.ArgumentParser(add_help=False)
    list_parser.add_argument('--list-devices', action='store_true', help='List available audio input devices and exit.')
    list_parser.add_argument('--list-voices', action='store_true', help='List available TTS voices and exit.')
    list_args, _ = list_parser.parse_known_args()

    # Audio/Voice listing must happen before PyAudio/TTS init, so they are checked here
    if list_args.list_devices:
        p_audio = pyaudio.PyAudio()
        list_audio_devices(p_audio)
        p_audio.terminate()
        sys.exit(0)
    
    if list_args.list_voices:
        list_tts_voices()
        sys.exit(0)


    # --- Configuration File Reading ---
    try:
        if config.read('config.ini'):
             logging.info("Loaded configuration from config.ini")
             
             if 'Models' in config:
                 argparse_defaults['ollama_model'] = config.get('Models', 'ollama_model', fallback=argparse_defaults['ollama_model'])
                 argparse_defaults['whisper_model'] = config.get('Models', 'whisper_model', fallback=argparse_defaults['whisper_model'])
                 argparse_defaults['wakeword_model_path'] = config.get('Models', 'wakeword_model_path', fallback=argparse_defaults['wakeword_model_path'])
             
             if 'Functionality' in config:
                 argparse_defaults['wakeword'] = config.get('Functionality', 'wakeword', fallback=argparse_defaults['wakeword'])
                 argparse_defaults['wakeword_threshold'] = config.getfloat('Functionality', 'wakeword_threshold', fallback=argparse_defaults['wakeword_threshold'])
                 argparse_defaults['vad_aggressiveness'] = config.getint('Functionality', 'vad_aggressiveness', fallback=argparse_defaults['vad_aggressiveness'])
                 argparse_defaults['silence_seconds'] = config.getfloat('Functionality', 'silence_seconds', fallback=argparse_defaults['silence_seconds'])
                 argparse_defaults['listen_timeout'] = config.getfloat('Functionality', 'listen_timeout', fallback=argparse_defaults['listen_timeout'])
                 argparse_defaults['pre_buffer_ms'] = config.getint('Functionality', 'pre_buffer_ms', fallback=argparse_defaults['pre_buffer_ms'])
                 argparse_defaults['system_prompt'] = config.get('Functionality', 'system_prompt', fallback=argparse_defaults['system_prompt'])
                 
                 device_index_val = config.get('Functionality', 'device_index', fallback=None)
                 if device_index_val is not None and device_index_val.strip() != '':
                     try:
                         argparse_defaults['device_index'] = int(device_index_val)
                     except ValueError:
                         argparse_defaults['device_index'] = None
                 else:
                    argparse_defaults['device_index'] = None
                 
                 argparse_defaults['tts_voice_id'] = config.get('Functionality', 'tts_voice_id', fallback=argparse_defaults['tts_voice_id'])

        else:
             logging.info("config.ini not found, using default settings.")
    except configparser.Error as e:
        logging.error(f"Error reading config.ini: {e}. Using default settings.")

    # --- Main Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Ollama STT-TTS Voice Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[temp_parser, list_parser]
    )

    parser.add_argument('--ollama-model', type=str, default=argparse_defaults['ollama_model'], help='Ollama model (e.g., "llama3")')
    parser.add_argument('--whisper-model', type=str, default=argparse_defaults['whisper_model'], help='Whisper model (e.g., "tiny.en", "base.en").')
    parser.add_argument('--wakeword-model-path', type=str, default=argparse_defaults['wakeword_model_path'], help='Full path to the .onnx wakeword model.')
    parser.add_argument('--wakeword', type=str, default=argparse_defaults['wakeword'], help='Wakeword phrase.')
    parser.add_argument('--wakeword-threshold', type=float, default=argparse_defaults['wakeword_threshold'], help='Wakeword sensitivity (0.0-1.0).')
    parser.add_argument('--vad-aggressiveness', type=int, default=argparse_defaults['vad_aggressiveness'], choices=[0, 1, 2, 3], help='VAD aggressiveness (0=least, 3=most).')
    parser.add_argument('--silence-seconds', type=float, default=argparse_defaults['silence_seconds'], help='Seconds of silence before stopping recording.')
    parser.add_argument('--listen-timeout', type=float, default=argparse_defaults['listen_timeout'], help='Seconds to wait for speech after wakeword before timeout.')
    parser.add_argument('--pre-buffer-ms', type=int, default=argparse_defaults['pre_buffer_ms'], help='Milliseconds of audio to pre-buffer.')
    parser.add_argument('--system-prompt', type=str, default=argparse_defaults['system_prompt'], help='The system prompt for the Ollama model.')
    parser.add_argument('--device-index', type=int, default=argparse_defaults['device_index'], help='Index of the audio input device to use. (Use --list-devices to see options)')
    parser.add_argument('--tts-voice-id', type=str, default=argparse_defaults['tts_voice_id'], help='ID of the pyttsx3 voice to use. (Use --list-voices to see options)')

    args = parser.parse_args()

    logging.info("Starting assistant with the following effective settings:")
    for arg, value in vars(args).items():
        if arg in ['list_devices', 'list_voices']: continue
        # Truncate long system prompts for cleaner logging
        if arg == 'system_prompt' and len(str(value)) > 100:
             logging.info(f"  --{arg}: '{str(value)[:100]}...'")
        else:
             logging.info(f"  --{arg}: {value}")
    logging.info("-" * 20)

    try:
        assistant = VoiceAssistant(args)
        assistant.run()
    except IOError as e:
        logging.critical(f"FATAL ERROR: {e}")
        logging.critical("This is often due to an audio device issue.")
        logging.critical("Try running with --list-devices to check your microphone.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except NameError:
         # Simplified flow for the interactive environment
         pass