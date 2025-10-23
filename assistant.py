#!/usr/bin/env python3

"""
assistant.py (formerly ollama_voice_chat.py)

The core logic for the hands-free Python voice assistant.
Uses openwakeword, webrtcvad, Whisper, Ollama, and pyttsx3.

Refactoring Updates:
- Core VoiceAssistant class moved here.
- Configuration and Audio constants/helpers moved to separate files.
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
import time
from openwakeword.model import Model
from typing import List, Dict, Optional, Any
import numpy.typing as npt
import sys

# Import external modules
from config_manager import load_config_and_args, check_ollama_connectivity
from audio_utils import (
    FORMAT, CHANNELS, RATE, CHUNK_SIZE, INT16_NORMALIZATION, 
    SENTENCE_END_PUNCTUATION, MAX_TTS_ERRORS, MAX_HISTORY_MESSAGES
)

# Import torch for CUDA check (needed for Whisper device selection)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
        self.max_history_tokens: int = args.max_history_tokens

        # Ollama Client and Connectivity Check
        self.ollama_is_connected: bool = check_ollama_connectivity(args.ollama_host)
        self.ollama_client = ollama.Client(host=args.ollama_host)
        
        # Audio stream control (Shared resource)
        # self.stream_lock = threading.Lock() # Not strictly needed with queue
        self.stream_buffer: queue.Queue[bytes] = queue.Queue() # Buffer for audio chunks read by the main thread/monitor

        # Calculate derived audio settings
        CHUNK_DURATION_MS: int = 30 # Must be 10, 20, or 30 for VAD
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_chunks: int = int(args.pre_buffer_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        try:
            logging.info(f"Loading openwakeword model from: {args.wakeword_model_path}...")
            self.oww_model: Model = Model(wakeword_model_paths=[args.wakeword_model_path])
            self.wakeword_model_key: str = list(self.oww_model.models.keys())[0] if self.oww_model.models else args.wakeword 
            logging.info(f"Loaded wakeword model with key: '{self.wakeword_model_key}'")
        except Exception as e:
            logging.critical(f"Error loading openwakeword model '{args.wakeword_model_path}': {e}")
            raise

        # Whisper Model Device Selection
        self.whisper_device: str = args.whisper_device
        if self.whisper_device == 'cuda' and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            logging.warning("CUDA not available. Falling back to CPU for Whisper.")
            self.whisper_device = 'cpu'
        elif self.whisper_device == 'cuda':
            logging.info(f"CUDA device found. Using {self.whisper_device} for Whisper.")
        else:
            self.whisper_device = 'cpu'
            logging.info(f"Using CPU for Whisper.")
            
        logging.info(f"Loading Whisper model: {args.whisper_model} on device '{self.whisper_device}'...")
        try:
            self.whisper_model = whisper.load_model(args.whisper_model, device=self.whisper_device)
        except (RuntimeError, OSError, ValueError, Exception) as e:
            logging.critical(f"Error loading Whisper model '{args.whisper_model}': {e}")
            raise

        # TTS Engine Setup
        logging.info("Initializing TTS engine...")
        try:
            self.tts_engine = pyttsx3.init()
        except Exception as e:
            logging.critical(f"Failed to initialize pyttsx3 engine: {e}")
            self.tts_has_failed = threading.Event()
            self.tts_has_failed.set()
        else:
            # Set volume and voice properties
            volume = max(0.0, min(1.0, args.tts_volume))
            self.tts_engine.setProperty('volume', volume)
            if args.tts_voice_id:
                found_voice = next((v for v in self.tts_engine.getProperty('voices') if v.id == args.tts_voice_id or v.name == args.tts_voice_id), None)
                if found_voice:
                    self.tts_engine.setProperty('voice', found_voice.id)
                    logging.info(f"Set TTS volume to: {volume:.2f}, Voice: {found_voice.id}")
                else:
                    logging.warning(f"TTS voice ID/Name '{args.tts_voice_id}' not found. Using default voice.")
            else:
                 logging.info(f"Set TTS volume to: {volume:.2f}, Using default voice.")

            # TTS Control Events/Queue
            self.tts_queue: queue.Queue[Optional[str]] = queue.Queue()
            self.tts_stop_event = threading.Event()
            self.is_speaking_event = threading.Event() 
            self.interrupt_event = threading.Event() 
            self.is_processing_command = threading.Event() 
            self.tts_has_failed = threading.Event()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Audio Device Setup
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
                raise

        # Conversation history
        self.messages: List[Dict[str, str]] = []
        self.reset_history()


    # --- PyAudio Helpers ---

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

    # --- TTS and Speaking Logic ---

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
                if text is None: break

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
                    with self.tts_queue.mutex: self.tts_queue.queue.clear()
                    self.tts_stop_event.set()
                    break

            finally:
                if text is not None: self.tts_queue.task_done()
                
                # Clear the speaking event only if nothing else is waiting
                if self.tts_queue.empty() and not self.interrupt_event.is_set() and not self.is_processing_command.is_set():
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
        self.interrupt_event.clear() # Clear interrupt once the new command is queued/starting

    def wait_for_speech(self) -> None:
        """Blocks until the TTS queue is empty and all speech is finished."""
        if hasattr(self, 'tts_has_failed') and self.tts_has_failed.is_set():
             return

        logging.debug("Waiting for speech to finish...")
        self.tts_queue.join()
        self.is_speaking_event.clear()
        logging.debug("Speech finished.")

    # --- Audio Monitoring and Barge-in ---

    def _audio_monitor_worker(self) -> None:
        """
        Dedicated thread to continuously monitor the microphone for speech during LLM generation
        or TTS playback, enabling 'barge-in' interruption.
        """
        logging.debug("Audio monitor started.")
        while self.is_processing_command.is_set():
            if self.interrupt_event.is_set():
                 time.sleep(0.01)
                 continue

            chunk = None
            try:
                chunk = self.stream_buffer.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if chunk:
                is_speech = self.vad.is_speech(chunk, RATE)
                
                if is_speech and (self.is_speaking_event.is_set() or self.is_processing_command.is_set()):
                    logging.info("User interruption (barge-in) detected by monitor!")
                    
                    self.interrupt_event.set()
                    if hasattr(self, 'tts_engine'):
                        self.tts_engine.stop() 
                    
                    with self.tts_queue.mutex:
                        self.tts_queue.queue.clear()
                    
                    self.is_speaking_event.clear()
                    
                self.stream_buffer.task_done()
                
        logging.debug("Audio monitor stopped.")

    # --- Recording and Transcription ---

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
            result = self.whisper_model.transcribe(audio_np, language="en")
            return result.get('text', '').strip()
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}")
            return ""

    def record_command(self) -> Optional[npt.NDArray[np.float32]]:
        """
        Records audio from the user until silence is detected.
        Reads chunks from the stream buffer (filled by the main loop).
        """
        logging.info("Listening for command...")
        frames: List[bytes] = []
        silent_chunks = 0
        is_speaking = False

        pre_buffer: List[bytes] = []
        timeout_chunks = self.pre_speech_timeout_chunks

        with self.stream_buffer.mutex:
            self.stream_buffer.queue.clear()

        while True:
            try:
                data = self.stream_buffer.get(timeout=self.args.listen_timeout) 
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
            
            except queue.Empty:
                logging.warning(f"No audio chunk received within listen timeout. Aborting recording.")
                return None
            except Exception as e:
                logging.error(f"Unexpected error during recording: {e}")
                return None

        if not frames: return None

        audio_data: bytes = b''.join(frames)
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_NORMALIZATION

        return audio_np

    # --- Ollama and History Management ---

    def _is_sentence_end(self, token: str) -> bool:
        """Helper to check if a token is sentence-ending punctuation."""
        return any(p in token for p in SENTENCE_END_PUNCTUATION)

    def _manage_history(self) -> None:
        """
        Implements a sliding window for conversation history based on token count
        or falls back to turn-based pruning.
        """
        if not self.messages or self.messages[0]['role'] != 'system':
             logging.warning("History corrupted, resetting to only system prompt.")
             self.messages = [{'role': 'system', 'content': self.system_prompt}]
             return
        
        if len(self.messages) <= 1: return

        current_token_count = 0
        try:
             full_token_count = self.ollama_client.count_tokens(
                 model=self.args.ollama_model, 
                 messages=self.messages
             ).get('count', 0)
             current_token_count = full_token_count
             
        except Exception as e:
             logging.error(f"Ollama count_tokens failed: {e}. Falling back to turn-based pruning.")
             self._manage_history_turn_fallback()
             return


        # Prune oldest messages (user/assistant pair) until the token count is under the limit
        while current_token_count > self.max_history_tokens and len(self.messages) > 1:
            if len(self.messages) >= 3 and self.messages[1]['role'] == 'user' and self.messages[2]['role'] == 'assistant':
                del self.messages[1:3]
                new_token_count = self.ollama_client.count_tokens(
                    model=self.args.ollama_model, messages=self.messages
                ).get('count', 0)
                
                tokens_removed = current_token_count - new_token_count
                logging.warning(f"History pruned: Removed oldest user/assistant pair ({tokens_removed} tokens). New size: {new_token_count} tokens.")
                current_token_count = new_token_count
            else:
                logging.error("History appears corrupted or too short to prune a turn. Stopping token-based pruning.")
                break
        
        if current_token_count > self.max_history_tokens:
            logging.critical(f"FATAL: Conversation history ({current_token_count} tokens) still exceeds limit. LLM call may fail.")

    def _manage_history_turn_fallback(self) -> None:
        """Fallback implementation for history pruning based on MAX_HISTORY_MESSAGES (turns)."""
        current_context_length = len(self.messages) - 1 
        current_turns = current_context_length // 2

        if current_turns > MAX_HISTORY_MESSAGES:
            turns_to_remove = current_turns - MAX_HISTORY_MESSAGES
            messages_to_remove = turns_to_remove * 2
            end_index_to_remove = 1 + messages_to_remove 

            if end_index_to_remove >= len(self.messages):
                 logging.error(f"History pruning error: calculated end_index {end_index_to_remove} is out of bounds. Skipping pruning.")
                 return

            self.messages = [self.messages[0]] + self.messages[end_index_to_remove:]
            new_turns = (len(self.messages) - 1) // 2
            logging.warning(f"Conversation history pruned (TURN FALLBACK). Removed {turns_to_remove} oldest turns. New size: {new_turns} turns.")


    def _update_history(self, user_text: str, assistant_response: str) -> None:
        """Manages conversation history update, including rollback on empty response."""
        if assistant_response.strip():
            self.messages.append({'role': 'assistant', 'content': assistant_response})
            logging.debug(f"History updated with assistant response of length {len(assistant_response)}.")
        else:
            if self.messages and self.messages[-1]['role'] == 'user' and self.messages[-1]['content'] == user_text:
                 self.messages.pop()
                 logging.warning("LLM response was empty or interrupted, user message rolled back from history.")
            elif self.messages:
                 logging.error(f"Failed to rollback user message: {self.messages[-1]['role']} != 'user' or content mismatch.")


    def get_ollama_response_stream(self, user_text: str) -> None:
        """
        Gets a streaming response from Ollama, maintains history, and pipes
        sentence-by-sentence output to the TTS queue.
        """
        self.is_processing_command.set()
        
        if not self.ollama_is_connected:
             self.speak("I can't talk right now, my language model isn't connected.")
             self.wait_for_speech()
             self.is_processing_command.clear()
             return

        self.messages.append({'role': 'user', 'content': user_text})
        self._manage_history()

        full_response = ""
        sentence_buffer = ""
        response_stream = None
        
        monitor_thread = threading.Thread(target=self._audio_monitor_worker, daemon=True)
        monitor_thread.start()

        try:
            response_stream = self.ollama_client.chat(
                model=self.args.ollama_model, messages=self.messages, stream=True, host=self.args.ollama_host
            )

            for chunk in response_stream:
                if self.interrupt_event.is_set():
                    logging.info("User interrupted, stopping LLM generation.")
                    break 
                
                token = chunk.get('message', {}).get('content', '')
                if not token: continue

                full_response += token
                sentence_buffer += token

                if self._is_sentence_end(token):
                    self.speak(sentence_buffer.strip())
                    sentence_buffer = ""
            
            if sentence_buffer.strip() and not self.interrupt_event.is_set():
                self.speak(sentence_buffer.strip())

        except Exception as e:
            logging.error(f"Error during Ollama streaming: {e}", exc_info=True)
            self.speak("I'm sorry, I encountered an error while thinking.")
            full_response = ""

        finally:
            self.is_processing_command.clear() 
            if monitor_thread.is_alive(): monitor_thread.join(timeout=0.2)
            
            if response_stream and hasattr(response_stream, 'close'):
                 response_stream.close()

            self._update_history(user_text, full_response)


    def process_user_command(self) -> bool:
        """
        Records, transcribes, and responds to a user command.
        Returns True if the assistant should exit.
        """
        
        audio_data = self.record_command()
        
        if audio_data is None:
            if not self.interrupt_event.is_set():
                self.speak("I didn't hear anything.")
                self.wait_for_speech()
            return False

        logging.info("Transcribing audio...")
        user_text = self.transcribe_audio(audio_data)
        
        if user_text:
            word_count = len(user_text.split())
            if word_count > self.args.max_words_per_command:
                logging.warning(f"Transcription rejected: {word_count} words (Max: {self.args.max_words_per_command}).")
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
        """The main loop of the assistant."""
        if self.device_index is None:
            logging.critical("Cannot run assistant: No valid audio input device index is set.")
            return

        # --- Stream Initialization ---
        MAX_STREAM_RETRIES = 3 
        RETRY_DELAY = 1.0

        for attempt in range(MAX_STREAM_RETRIES):
            try:
                self.stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                              input_device_index=self.device_index, frames_per_buffer=CHUNK_SIZE)
                logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                break
            except IOError as e:
                if attempt < MAX_STREAM_RETRIES - 1:
                    logging.warning(f"Error opening audio stream (Attempt {attempt+1}): {e}. Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    logging.critical(f"FATAL: Failed to open audio stream after {MAX_STREAM_RETRIES} attempts. {e}")
                    return
        
        if not self.stream: return

        # --- Main Loop ---
        try:
            while True:
                if hasattr(self, 'tts_has_failed') and self.tts_has_failed.is_set():
                     logging.critical("TTS engine failed permanently. Shutting down gracefully.")
                     break

                if not self.stream: break
                if not self.stream.is_active(): self.stream.start_stream()

                audio_chunk = None
                try:
                    audio_chunk = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except Exception as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}. Stopping.")
                    break
                    
                
                # Shared Buffer Management
                if audio_chunk:
                    try:
                        self.stream_buffer.put_nowait(audio_chunk)
                    except queue.Full:
                        logging.debug("Dropped audio chunk due to full stream buffer.")
                    
                
                # Skip wakeword check if processing a command (allows monitor thread to work)
                if self.is_processing_command.is_set():
                    time.sleep(0.001)
                    continue

                # Wakeword Check
                audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                prediction = self.oww_model.predict(audio_np_int16)

                if prediction.get(self.wakeword_model_key, 0) > self.args.wakeword_threshold:
                    logging.info(f"Wakeword '{self.args.wakeword}' detected! (Score: {prediction.get(self.wakeword_model_key):.2f})")
                    self.oww_model.reset()
                    
                    self.is_processing_command.set()
                    
                    if self.process_user_command():
                        break
                    
                    logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                    self.is_processing_command.clear()

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up PyAudio and stops the TTS thread."""
        logging.info("Cleaning up resources...")
        
        if self.stream:
            try:
                if self.stream.is_active(): self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logging.warning(f"Error managing audio stream: {e}")
            finally:
                self.stream = None
        
        if hasattr(self, 'whisper_model'):
            try:
                if hasattr(self.whisper_model, 'to') and self.whisper_device != 'cpu':
                    self.whisper_model.to('cpu')
                logging.debug("Whisper model resources released.")
            except Exception as e:
                logging.warning(f"Error releasing Whisper model resources: {e}")

        if hasattr(self, 'tts_stop_event'):
            self.tts_stop_event.set()
            self.tts_queue.put(None) 
            if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=1.0)
            
        if hasattr(self, 'tts_engine'): self.tts_engine.stop()

        if self.audio: self.audio.terminate()


def main() -> None:
    """The entry point for the assistant application."""
    try:
        args, _ = load_config_and_args()
        assistant = VoiceAssistant(args)
        assistant.run()

    except IOError as e:
        logging.critical(f"FATAL ERROR during audio initialization: {e}")
        logging.critical("Check microphone connectivity or use --list-devices.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except NameError:
         # Happens in some environments if main() isn't defined correctly
         pass