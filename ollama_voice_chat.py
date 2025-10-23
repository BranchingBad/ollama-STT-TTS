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
import configparser
import time
from openwakeword.model import Model
from typing import List, Dict, Optional, Any, Tuple
import numpy.typing as npt
import sys
import os # NEW: Import os for file path checking

# Import torch for CUDA check (needed for Whisper device selection)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not found. GPU (CUDA) support for Whisper will be disabled.")


# --- 1. Audio Settings (Constants) ---
FORMAT = pyaudio.paInt16       # 16-bit audio
CHANNELS = 1                 # Mono
RATE = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS = 30       # 30ms chunks for VAD
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames per chunk
INT16_NORMALIZATION = 32768.0 # Normalization factor for int16
SENTENCE_END_PUNCTUATION = ['.', '?', '!', '\n']
MAX_TTS_ERRORS = 5           # Max consecutive errors before stopping TTS worker
DEFAULT_OLLAMA_HOST = 'http://localhost:11434' # Define default for logging check
MAX_HISTORY_MESSAGES = 20    # Maximum *turns* (user/assistant pairs) to keep in conversation history

# --- 2. Centralized Configuration Defaults ---
DEFAULT_SETTINGS: Dict[str, Any] = {
    'ollama_model': 'llama3',
    'whisper_model': 'tiny.en',
    'wakeword_model_path': 'hey_glados.onnx',
    'ollama_host': DEFAULT_OLLAMA_HOST,
    'wakeword': 'hey glados',
    'wakeword_threshold': 0.5, 
    'vad_aggressiveness': 3,   
    'silence_seconds': 0.5,    
    'listen_timeout': 4.0,     
    'pre_buffer_ms': 400,
    'system_prompt': 'You are a helpful, concise voice assistant.',
    'device_index': None,
    'tts_voice_id': None,
    'tts_volume': 1.0, 
    'max_words_per_command': 60, 
    'whisper_device': 'cpu', # NEW: Default Whisper device
}


# --- 3. External Dependency Checks ---

def check_ollama_connectivity(host: str) -> bool:
    """Checks if the Ollama server is running and reachable."""
    try:
        ollama.Client(host=host).list()
        logging.info("Ollama server is reachable.")
        return True
    except Exception as e:
        # Changed to ERROR for the initial check to be more prominent
        logging.error(f"Ollama server is not reachable at {host}. Error: {e}")
        return False

def check_local_files_exist(args: argparse.Namespace) -> bool:
    """Checks if the local model files (Whisper and Wakeword) exist."""
    success = True

    # 1. Check Wakeword model file
    wakeword_path = args.wakeword_model_path
    if not os.path.exists(wakeword_path):
        logging.error(f"Wakeword model file not found: '{wakeword_path}'")
        logging.error("Please ensure the file path is correct or download the necessary .onnx model.")
        success = False
    else:
        logging.debug(f"Wakeword model file found: '{wakeword_path}'")

    # 2. Check Whisper model (only if it's a local file path, not a name like 'tiny.en')
    # Whisper can load models by name, so we only check if the model name contains a path separator.
    whisper_model = args.whisper_model
    if os.path.sep in whisper_model or '/' in whisper_model or '\\' in whisper_model:
        # User supplied a path, so check if it exists
        if not os.path.exists(whisper_model):
            logging.error(f"Whisper model file not found: '{whisper_model}'")
            logging.error("If you intended to use a local file, ensure the path is correct.")
            success = False
        else:
            logging.debug(f"Whisper model file found: '{whisper_model}'")
    else:
        logging.debug(f"Whisper model is a standard name ('{whisper_model}'), relying on whisper package download/cache.")

    return success

# --- 4. PyAudio and TTS Helpers ---
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
            # pyttsx3 doesn't expose gender consistently, so check for existence
            gender_info = voice.gender if hasattr(voice, 'gender') else 'N/A'
            print(f"    - Gender: {gender_info}")
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
        # Assumes local file checks passed in load_config_and_args
        self.ollama_is_connected = check_ollama_connectivity(args.ollama_host)

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
            logging.critical("This error usually means the .onnx file is corrupted or not accessible.")
            raise

        # Whisper Model Device Selection (ENHANCEMENT)
        self.whisper_device = args.whisper_device
        if self.whisper_device == 'cuda':
            if not TORCH_AVAILABLE:
                logging.warning("PyTorch not installed. Falling back to CPU for Whisper.")
                self.whisper_device = 'cpu'
            elif not torch.cuda.is_available():
                logging.warning("CUDA device not found. Falling back to CPU for Whisper.")
                self.whisper_device = 'cpu'
            else:
                logging.info(f"CUDA device found. Using {self.whisper_device} for Whisper.")
        else:
            self.whisper_device = 'cpu'
            logging.info(f"Using CPU for Whisper.")
            
        logging.info(f"Loading Whisper model: {args.whisper_model} on device '{self.whisper_device}'...")
        try:
            # We use the determined device
            self.whisper_model = whisper.load_model(args.whisper_model, device=self.whisper_device)
        except (RuntimeError, OSError, ValueError, Exception) as e:
            logging.critical(f"Error loading Whisper model '{args.whisper_model}': {e}")
            logging.critical("This may be due to a missing model file, low system memory, or an incompatible PyTorch version.")
            raise

        # TTS Engine
        logging.info("Initializing TTS engine...")
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            logging.critical(f"Failed to initialize pyttsx3 engine: {e}")
            self.tts_has_failed = threading.Event()
            self.tts_has_failed.set()
            
        else:
            current_voice_id = self.tts_engine.getProperty('voice')
            logging.debug(f"Default TTS voice ID: {current_voice_id}")
            
            # Set volume
            try:
                volume = max(0.0, min(1.0, args.tts_volume))
                self.tts_engine.setProperty('volume', volume)
                logging.info(f"Set TTS volume to: {volume:.2f}")
            except Exception as e:
                logging.error(f"Error setting TTS volume: {e}. Using default.")


            if args.tts_voice_id:
                try:
                    found_voice = None
                    for voice in self.tts_engine.getProperty('voices'):
                        if voice.id == args.tts_voice_id or voice.name == args.tts_voice_id:
                            found_voice = voice
                            break
                    
                    if found_voice:
                        self.tts_engine.setProperty('voice', found_voice.id)
                        logging.info(f"Successfully set voice to: {found_voice.id} ({found_voice.name})")
                    else:
                        logging.warning(f"TTS voice ID/Name '{args.tts_voice_id}' not found. Using default voice: {current_voice_id}.")

                except Exception as e:
                    logging.error(f"Error setting TTS voice ID: {e}. Using default.")
        
            # TTS Control
            self.tts_queue: queue.Queue[Optional[str]] = queue.Queue()
            self.tts_stop_event = threading.Event()
            self.is_speaking_event = threading.Event()
            self.interrupt_event = threading.Event()
            self.tts_has_failed = threading.Event()
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
                logging.critical("No suitable audio input device found. Cannot listen.")
            else:
                logging.info(f"Auto-selected audio device index: {self.device_index}")
        else:
            try:
                device_info = self.audio.get_device_info_by_index(self.device_index)
                if int(device_info.get('maxInputChannels', 0)) == 0:
                    raise IOError(f"Device index {self.device_index} is not an input device.")
                logging.info(f"Using specified audio device index: {self.device_index} ({device_info.get('name')})")
            except IOError as e:
                logging.critical(f"Invalid --device-index {self.device_index}: {e}")
                logging.critical("Use --list-devices to see available devices.")

        # Conversation history
        self.messages: List[Dict[str, str]] = []
        self.reset_history()
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
            logging.warning(f"Could could not get default input device automatically: {e}. Searching all devices...")
            for i in range(self.audio.get_device_count()):
                try:
                    dev = self.audio.get_device_info_by_index(i)
                    if int(dev.get('maxInputChannels', 0)) > 0:
                        logging.warning(f"Using first available input device (index {i}): {dev['name']}")
                        return i
                except IOError:
                    continue
            logging.error("No available input device found after searching all devices.")
        return None

    def reset_history(self) -> None:
        """Resets the conversation history to only the system prompt."""
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        logging.debug("Conversation history reset.")

    def _tts_worker(self) -> None:
        """Dedicated thread for processing TTS tasks with error resilience."""
        consecutive_errors = 0
        while not self.tts_stop_event.is_set():
            text = None
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text is None:
                    break

                self.is_speaking_event.set()
                self.tts_engine.stop() 
                
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                
                consecutive_errors = 0

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error while processing '{text[:20]}...': {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_TTS_ERRORS:
                    logging.critical(f"TTS worker failed {MAX_TTS_ERRORS} consecutive times. Stopping TTS thread.")
                    self.tts_has_failed.set()
                    with self.tts_queue.mutex:
                        self.tts_queue.queue.clear()
                    self.tts_stop_event.set()
                    break

            finally:
                if text is not None:
                    self.tts_queue.task_done()
                
                if self.tts_queue.empty() and not self.interrupt_event.is_set():
                    self.is_speaking_event.clear()

    def speak(self, text: str) -> None:
        """Adds text to the TTS queue to be spoken by the worker thread."""
        if hasattr(self, 'tts_has_failed') and self.tts_has_failed.is_set():
            logging.error(f"Cannot speak: TTS engine has failed permanently. Text: '{text[:20]}...'")
            return

        logging.debug(f"Queueing TTS: '{text}'")
        
        if self.interrupt_event.is_set():
             with self.tts_queue.mutex:
                self.tts_queue.queue.clear()
        self.tts_queue.put(text)

    def wait_for_speech(self) -> None:
        """Blocks until the TTS queue is empty and all speech is finished."""
        if hasattr(self, 'tts_has_failed') and self.tts_has_failed.is_set():
             return

        logging.debug("Waiting for speech to finish...")
        self.tts_queue.join()
        self.is_speaking_event.clear()
        logging.debug("Speech finished.")

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
            # Device is now dynamically set during model loading
            result = self.whisper_model.transcribe(audio_np, language="en")
            return result.get('text', '').strip()
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}")
            return ""

    def _is_sentence_end(self, token: str) -> bool:
        """Helper to check if a token is sentence-ending punctuation."""
        return any(p in token for p in SENTENCE_END_PUNCTUATION)

    def _manage_history(self) -> None:
        """
        Implements a sliding window for conversation history to prevent
        the list from growing indefinitely, ensuring the system prompt is always
        the first message.
        
        MAX_HISTORY_MESSAGES is interpreted as the maximum number of user/assistant
        turns (pairs) to keep.
        """
        if not self.messages or self.messages[0]['role'] != 'system':
             logging.warning("History corrupted, resetting to only system prompt.")
             self.messages = [{'role': 'system', 'content': self.system_prompt}]
             return
        
        # Total messages excluding the system prompt
        current_context_length = len(self.messages) - 1 

        # Current conversation turns (pairs)
        current_turns = current_context_length // 2

        if current_turns > MAX_HISTORY_MESSAGES:
            # Calculate how many full turns (user + assistant) to remove
            turns_to_remove = current_turns - MAX_HISTORY_MESSAGES
            
            # Each turn is 2 messages (user, assistant)
            messages_to_remove = turns_to_remove * 2
            
            # The system message is always at index 0. Removal starts at index 1.
            end_index_to_remove = 1 + messages_to_remove 

            if end_index_to_remove >= len(self.messages):
                 logging.error(f"History pruning error: calculated end_index {end_index_to_remove} is out of bounds (len: {len(self.messages)}). Skipping pruning.")
                 return

            # Remove the oldest pairs (from index 1 up to end_index_to_remove)
            self.messages = [self.messages[0]] + self.messages[end_index_to_remove:]
            
            # Calculate new size in conversation turns (pairs)
            new_turns = (len(self.messages) - 1) // 2

            logging.warning(f"Conversation history pruned. Removed {turns_to_remove} oldest turns. New size: {new_turns} turns (Target: {MAX_HISTORY_MESSAGES}).")
            logging.debug(f"New history length: {len(self.messages)}")


    def _update_history(self, user_text: str, assistant_response: str) -> None:
        """
        Manages conversation history update, including rollback on empty response.
        """
        if assistant_response.strip():
            # Add assistant's response to the history
            self.messages.append({'role': 'assistant', 'content': assistant_response})
            logging.debug(f"History updated with assistant response of length {len(assistant_response)}.")
        else:
            # Rollback: Remove the last user message if the LLM failed to respond
            if self.messages and self.messages[-1]['role'] == 'user' and self.messages[-1]['content'] == user_text:
                 self.messages.pop()
                 logging.warning("LLM response was empty or interrupted, user message rolled back from history.")
            elif self.messages:
                 logging.error(f"Failed to rollback user message: {self.messages[-1]['role']} != 'user' or content mismatch.")


    def get_ollama_response_stream(self, user_text: str) -> None:
        """
        Gets a response from Ollama, maintains conversation history
        and streams the output sentence-by-sentence to the TTS queue.
        """
        if not self.ollama_is_connected:
             logging.warning("Ollama connection is known to be down. Skipping API call.")
             self.speak("I can't talk right now, my language model isn't connected.")
             self.wait_for_speech()
             return

        # 1. Append user message for context *before* the API call
        self.messages.append({'role': 'user', 'content': user_text})
        
        self._manage_history()

        full_response = ""
        sentence_buffer = ""
        response_stream = None

        try:
            response_stream = ollama.chat(
                model=self.args.ollama_model,
                messages=self.messages,
                stream=True,
                host=self.args.ollama_host
            )

            for chunk in response_stream:
                # --- Check for user interruption ---
                if self.interrupt_event.is_set():
                    logging.info("User interrupted, stopping LLM generation.")
                    break 
                
                token = chunk.get('message', {}).get('content', '')
                if not token:
                    continue

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
            logging.error(error_message, exc_info=True)
            self.speak("I'm sorry, I encountered an internal error while thinking.")
            full_response = ""

        finally:
            if response_stream:
                try:
                    if hasattr(response_stream, 'close'):
                         response_stream.close()
                except Exception as e:
                    logging.error(f"Error closing Ollama stream: {e}")
            
            # 2. Update the history with the complete (or interrupted/failed) response
            self._update_history(user_text, full_response)


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

        if not self.stream or not self.stream.is_active():
            logging.error("Audio stream is not open or active. Cannot record.")
            return None

        while True:
            try:
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
                        logging.warning(f"No speech detected after {self.args.listen_timeout} seconds, timing out.")
                        return None
            except IOError as e:
                logging.error(f"Error reading from audio stream: {e}")
                return None
            except Exception as e:
                logging.error(f"Unexpected error during recording: {e}")
                return None

        if not frames:
             return None

        audio_data: bytes = b''.join(frames)
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_NORMALIZATION

        return audio_np

    def process_user_command(self) -> bool:
        """
        Records, transcribes, and responds to a user command.
        Returns True if the assistant should exit.
        """
        self.interrupt_event.clear()

        audio_data = self.record_command()
        
        if self.stream and self.stream.is_active():
             self.stream.stop_stream()


        if audio_data is None:
            if not self.interrupt_event.is_set():
                self.speak("I didn't hear anything.")
                self.wait_for_speech()
            return False

        logging.info("Transcribing audio...")
        user_text = self.transcribe_audio(audio_data)
        
        # --- Word count check ---
        if user_text:
            word_count = len(user_text.split())
            if word_count > self.args.max_words_per_command:
                logging.warning(f"Transcription rejected: {word_count} words (Max: {self.args.max_words_per_command}). Likely noise or misfire.")
                self.speak("That command was too long or contained too much background noise. Please try again.")
                self.wait_for_speech()
                return False

            logging.info(f"You: {user_text}")
            
            user_prompt = user_text.lower().strip().rstrip(".,!?")

            if "exit" in user_prompt or "goodbye" in user_prompt:
                self.speak("Goodbye!")
                self.wait_for_speech()
                return True
            
            if "new chat" in user_prompt or "reset chat" in user_prompt:
                self.reset_history()
                self.speak("Starting a new conversation.")
                self.wait_for_speech()
                return False

            self.speak("Thinking...")
            self.wait_for_speech()

            logging.info(f"Sending to {self.args.ollama_model}...")
            
            self.get_ollama_response_stream(user_text)
            
            self.wait_for_speech() 

        else:
            logging.warning("Transcription failed. No speech detected or unclear input.")
            self.speak("I'm sorry, I couldn't understand what you said.")
            self.wait_for_speech()
        
        return False

    def run(self) -> None:
        """
        The main loop of the assistant.
        """
        if self.device_index is None:
            logging.critical("Cannot run assistant: No valid audio input device index is set.")
            return

        MAX_STREAM_RETRIES = 3 
        RETRY_DELAY = 1.0

        # --- Stream Initialization with Retry ---
        for attempt in range(MAX_STREAM_RETRIES):
            try:
                self.stream = self.audio.open(format=FORMAT,
                                              channels=CHANNELS,
                                              rate=RATE,
                                              input=True,
                                              input_device_index=self.device_index,
                                              frames_per_buffer=CHUNK_SIZE)
                logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                break
            except IOError as e:
                if attempt < MAX_STREAM_RETRIES - 1:
                    logging.warning(f"Error opening audio stream (Attempt {attempt+1}/{MAX_STREAM_RETRIES}): {e}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    logging.critical(f"FATAL: Failed to open audio stream after {MAX_STREAM_RETRIES} attempts. {e}")
                    return
        
        if not self.stream:
             return

        # --- Main Loop ---
        try:
            while True:
                if hasattr(self, 'tts_has_failed') and self.tts_has_failed.is_set():
                     logging.critical("TTS engine failed permanently. Shutting down gracefully.")
                     break

                if not self.stream:
                    logging.error("Audio stream was unexpectedly closed.")
                    break
                
                if not self.stream.is_active():
                     self.stream.start_stream()


                try:
                    audio_chunk = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}")
                    logging.error("This may be due to a microphone disconnection. Stopping.")
                    break

                # --- 1. Check for User Interruption (Barge-in) ---
                if hasattr(self, 'is_speaking_event') and self.is_speaking_event.is_set():
                    is_speech = self.vad.is_speech(audio_chunk, RATE)
                    if is_speech:
                        logging.info("User interruption (barge-in) detected!")
                        
                        self.tts_engine.stop() 
                        with self.tts_queue.mutex:
                            self.tts_queue.queue.clear()
                        
                        self.interrupt_event.set()
                        self.is_speaking_event.clear()
                        
                        if self.process_user_command():
                            break
                        
                        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                    
                    continue
                
                # --- 2. Check for Wakeword (if not speaking) ---
                audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_np_int16)

                if prediction.get(self.wakeword_model_key, 0) > self.args.wakeword_threshold:
                    logging.info(f"Wakeword '{self.args.wakeword}' detected! (Score: {prediction.get(self.wakeword_model_key):.2f})")
                    self.oww_model.reset()

                    if self.process_user_command():
                        break

                    logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up PyAudio and stops the TTS thread."""
        logging.info("Cleaning up resources...")
        
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
            except Exception as e:
                logging.warning(f"Error stopping audio stream: {e}")
            
            try:
                self.stream.close()
            except Exception as e:
                logging.warning(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        if hasattr(self, 'whisper_model'):
            try:
                # This explicitly releases the model from the device (important for VRAM on GPU)
                if hasattr(self.whisper_model, 'to') and self.whisper_device != 'cpu':
                    self.whisper_model.to('cpu')
                
                # These methods are often placeholders but good practice to call
                if hasattr(self.whisper_model, 'release'):
                    self.whisper_model.release() 
                elif hasattr(self.whisper_model, 'reset'):
                    self.whisper_model.reset()
                
                logging.debug("Whisper model resources released.")
            except Exception as e:
                logging.warning(f"Error releasing Whisper model resources: {e}")

        if hasattr(self, 'tts_stop_event'):
            self.tts_stop_event.set()
            self.tts_queue.put(None) 
            if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=1.0)
                if self.tts_thread.is_alive():
                     logging.warning("TTS thread did not join gracefully.")
            
        if hasattr(self, 'tts_engine'):
             self.tts_engine.stop()

        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                 logging.warning(f"Error terminating PyAudio: {e}")


def load_config_and_args() -> Tuple[argparse.Namespace, argparse.ArgumentParser]:
    """
    Loads configuration from config.ini, defines command-line arguments,
    and returns parsed arguments along with the parser for list commands.
    """
    config = configparser.ConfigParser()
    argparse_defaults = DEFAULT_SETTINGS.copy()

    # 1. Initial Parsing for Verbose and List flags
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose DEBUG logging')
    temp_args, remaining_args = temp_parser.parse_known_args()

    log_level = logging.DEBUG if temp_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    list_parser = argparse.ArgumentParser(add_help=False)
    list_parser.add_argument('--list-devices', action='store_true', help='List available audio input devices and exit.')
    list_parser.add_argument('--list-voices', action='store_true', help='List available TTS voices and exit.')
    list_args, _ = list_parser.parse_known_args(remaining_args)

    # 2. Configuration File Reading
    try:
        if config.read('config.ini'):
             logging.info("Loaded configuration from config.ini")
             
             if 'Models' in config:
                 argparse_defaults['ollama_model'] = config.get('Models', 'ollama_model', fallback=argparse_defaults['ollama_model'])
                 argparse_defaults['whisper_model'] = config.get('Models', 'whisper_model', fallback=argparse_defaults['whisper_model'])
                 argparse_defaults['wakeword_model_path'] = config.get('Models', 'wakeword_model_path', fallback=argparse_defaults['wakeword_model_path'])
                 argparse_defaults['ollama_host'] = config.get('Models', 'ollama_host', fallback=argparse_defaults['ollama_host'])
             
             if 'Functionality' in config:
                 argparse_defaults['wakeword'] = config.get('Functionality', 'wakeword', fallback=argparse_defaults['wakeword'])
                 argparse_defaults['wakeword_threshold'] = config.getfloat('Functionality', 'wakeword_threshold', fallback=argparse_defaults['wakeword_threshold'])
                 argparse_defaults['vad_aggressiveness'] = config.getint('Functionality', 'vad_aggressiveness', fallback=argparse_defaults['vad_aggressiveness'])
                 argparse_defaults['silence_seconds'] = config.getfloat('Functionality', 'silence_seconds', fallback=argparse_defaults['silence_seconds'])
                 argparse_defaults['listen_timeout'] = config.getfloat('Functionality', 'listen_timeout', fallback=argparse_defaults['listen_timeout'])
                 argparse_defaults['pre_buffer_ms'] = config.getint('Functionality', 'pre_buffer_ms', fallback=argparse_defaults['pre_buffer_ms'])
                 argparse_defaults['system_prompt'] = config.get('Functionality', 'system_prompt', fallback=argparse_defaults['system_prompt'])
                 
                 argparse_defaults['tts_voice_id'] = config.get('Functionality', 'tts_voice_id', fallback=argparse_defaults['tts_voice_id'])
                 argparse_defaults['tts_volume'] = config.getfloat('Functionality', 'tts_volume', fallback=argparse_defaults['tts_volume'])
                 argparse_defaults['max_words_per_command'] = config.getint('Functionality', 'max_words_per_command', fallback=argparse_defaults['max_words_per_command']) 
                 argparse_defaults['whisper_device'] = config.get('Functionality', 'whisper_device', fallback=argparse_defaults['whisper_device']) # NEW: Load device setting
                 
                 device_index_val = config.get('Functionality', 'device_index', fallback=None)
                 if device_index_val is not None and device_index_val.strip() != '' and device_index_val.strip().lower() != 'none':
                     try:
                         argparse_defaults['device_index'] = int(device_index_val)
                     except ValueError:
                         logging.error(f"Invalid integer value for device_index in config.ini: '{device_index_val}'. Using auto-select (None).")
                         argparse_defaults['device_index'] = None
                 else:
                    argparse_defaults['device_index'] = None


        else:
             logging.info("config.ini not found, using default settings.")
    except configparser.Error as e:
        logging.error(f"Error reading config.ini: {e}. Using default settings.")

    # 3. Main Argument Parsing
    parser = argparse.ArgumentParser(
        description="Ollama STT-TTS Voice Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[temp_parser, list_parser]
    )

    parser.add_argument('--ollama-model', type=str, default=argparse_defaults['ollama_model'], help='Ollama model (e.g., "llama3")')
    parser.add_argument('--ollama-host', type=str, default=argparse_defaults['ollama_host'], help='The URL for the Ollama server (e.g., "http://192.168.1.10:11434").')
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
    parser.add_argument('--tts-volume', type=float, default=argparse_defaults['tts_volume'], help='TTS speaking volume (0.0 to 1.0).')
    parser.add_argument('--max-words-per-command', type=int, default=argparse_defaults['max_words_per_command'], help='Maximum number of words allowed in a command transcription.')
    parser.add_argument('--whisper-device', type=str, default=argparse_defaults['whisper_device'], choices=['cpu', 'cuda'], help="Device to use for Whisper transcription ('cpu' or 'cuda').") # NEW: Argparse for device


    args = parser.parse_args()

    # 4. Check Local Files and Exit if Critical Files are Missing
    if not list_args.list_devices and not list_args.list_voices:
        if not check_local_files_exist(args):
            logging.critical("FATAL: One or more local model files are missing. Cannot start assistant.")
            sys.exit(1)


    # Logging final effective settings
    logging.info("Starting assistant with the following effective settings:")
    for arg, value in vars(args).items():
        if arg in ['list_devices', 'list_voices', 'verbose']: continue
        
        if arg == 'ollama_host' and value == DEFAULT_OLLAMA_HOST:
             logging.info(f"  --{arg}: {value} (Default)")
             continue

        if arg == 'system_prompt' and len(str(value)) > 100:
             logging.info(f"  --{arg}: '{str(value)[:100]}...'")
        else:
             logging.info(f"  --{arg}: {value}")
    logging.info("-" * 20)

    # Return main arguments and the list_args object (which contains list_devices/list_voices)
    return args, list_args


def main() -> None:
    """
    Parses arguments, handles list commands, and runs the VoiceAssistant.
    """
    try:
        # Load configuration and arguments
        # The function handles logging and critical file checks internally
        args, list_args = load_config_and_args()

        # Handle list commands immediately
        if list_args.list_devices:
            p_audio = pyaudio.PyAudio()
            list_audio_devices(p_audio)
            p_audio.terminate()
            sys.exit(0)
        
        if list_args.list_voices:
            list_tts_voices()
            sys.exit(0)

        # Run the main assistant logic
        assistant = VoiceAssistant(args)
        assistant.run()

    except IOError as e:
        logging.critical(f"FATAL ERROR during initialization: {e}")
        logging.critical("This is often due to an audio device issue or missing required file.")
        logging.critical("Try running with --list-devices to check your microphone.")
    except Exception as e:
        # Only catch generic exceptions here. Specific setup exceptions (like file not found)
        # should have been caught and handled/re-raised in VoiceAssistant.__init__
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except NameError:
         pass