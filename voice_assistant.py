#!/usr/bin/env python3

"""
voice_assistant.py

Contains the core VoiceAssistant class logic.
Uses sounddevice, webrtcvad, faster-whisper, Ollama, and Piper TTS.
"""

import ollama
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import webrtcvad
import argparse
import logging
import threading
import queue
import time
from openwakeword.model import Model
from typing import Any, Optional
import numpy.typing as npt
import sys
from contextlib import contextmanager
import os
import gc
import json
from piper import PiperVoice


# Import external modules
from audio_utils import (
    FORMAT_NP, CHANNELS, RATE, CHUNK_SIZE, INT16_NORMALIZATION,
    SENTENCE_END_PUNCTUATION, MAX_TTS_ERRORS, MAX_HISTORY_MESSAGES
)

# Import torch for CUDA check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class VoiceAssistant:
    """
    Manages all components of the voice assistant.
    """

    def __init__(self, args: argparse.Namespace, client: Optional[ollama.Client]) -> None:
        """
        Initializes the assistant, loads models, and sets up audio.
        """
        self.args: argparse.Namespace = args
        self.system_prompt: str = args.system_prompt
        self.max_history_tokens: int = args.max_history_tokens
        self.ollama_client: Optional[ollama.Client] = client
        self.stream_buffer: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.last_buffer_drop_warning: float = 0.0

        # Calculate derived audio settings
        CHUNK_DURATION_MS: int = 30
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_chunks: int = int(args.pre_buffer_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        try:
            logging.info(f"Loading openwakeword model from: {args.wakeword_model_path}...")
            if not os.path.exists(args.wakeword_model_path):
                 raise FileNotFoundError(f"Wakeword model file not found at: {args.wakeword_model_path}")

            self.oww_model: Model = Model(wakeword_model_paths=[args.wakeword_model_path])
            base_filename = os.path.basename(args.wakeword_model_path)
            self.wakeword_model_key: str = base_filename.split('.')[0]
            if self.wakeword_model_key not in self.oww_model.models:
                 available_keys = list(self.oww_model.models.keys())
                 if not available_keys:
                     raise ValueError("Openwakeword model loaded but contains no keys.")
                 self.wakeword_model_key = available_keys[0]
                 logging.warning(f"Default key '{base_filename.split('.')[0]}' not found. Using first available: '{self.wakeword_model_key}'")
            logging.info(f"Loaded wakeword model with key: '{self.wakeword_model_key}'")
        except FileNotFoundError as e:
            logging.critical(str(e))
            raise
        except Exception as e:
            logging.critical(f"Error loading openwakeword model '{args.wakeword_model_path}': {e}")
            raise

        # Whisper Model
        self.whisper_device: str = args.whisper_device
        self.whisper_compute_type: str = args.whisper_compute_type
        
        if self.whisper_device == 'cuda' and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            logging.warning("CUDA not available. Falling back to CPU for Whisper.")
            self.whisper_device = 'cpu'
        elif self.whisper_device == 'cuda':
            logging.info(f"CUDA device found. Using {self.whisper_device} for Whisper.")
        else:
            self.whisper_device = 'cpu'
            logging.info(f"Using CPU for Whisper.")

        logging.info(f"Loading faster-whisper model: {args.whisper_model} on device '{self.whisper_device}' (Compute: {self.whisper_compute_type})...")
        try:
            self.whisper_model = WhisperModel(
                args.whisper_model,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type
            )
        except (RuntimeError, OSError, ValueError, Exception) as e:
            logging.critical(f"Error loading faster-whisper model '{args.whisper_model}': {e}")
            raise

        # --- NEW: Piper TTS Engine Setup ---
        logging.info("Initializing TTS engine (Piper)...")
        self.piper_voice: Optional[PiperVoice] = None
        self.piper_sample_rate: int = 16000  # Default, will be overwritten
        try:
            self.piper_model_path: str = args.piper_model_path
            self.piper_config_path: str = args.piper_model_path + ".json"

            if not os.path.exists(self.piper_model_path):
                raise FileNotFoundError(f"Piper model file not found at: {self.piper_model_path}")
            if not os.path.exists(self.piper_config_path):
                raise FileNotFoundError(f"Piper config file not found at: {self.piper_config_path}")

            # Load Piper model config to get sample rate
            with open(self.piper_config_path, 'r') as f:
                config = json.load(f)
                self.piper_sample_rate = int(config['audio']['sample_rate'])
                logging.info(f"Piper model sample rate: {self.piper_sample_rate} Hz")

            # Load Piper voice
            self.piper_voice = PiperVoice.load(self.piper_model_path, self.piper_config_path)
            logging.info(f"Loaded Piper TTS model: {self.piper_model_path}")

            # Setup TTS queue and thread
            self.tts_queue: queue.Queue[str | None] = queue.Queue()
            self.tts_stop_event = threading.Event()
            self.is_speaking_event = threading.Event()
            self.interrupt_event = threading.Event()
            self.is_processing_command = threading.Event()
            self.tts_has_failed = threading.Event()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()

        except Exception as e:
            logging.critical(f"Failed to initialize Piper TTS engine: {e}")
            logging.critical("Please ensure the Piper model and config files exist and are valid.")
            self.tts_has_failed = threading.Event()
            self.tts_has_failed.set()
        # --- END: Piper TTS ---

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)
        self.stream: sd.InputStream | None = None

        # Audio Device Setup (Input)
        self.device_index: int | None = args.device_index
        if self.device_index is None:
            self.device_index = self.find_default_input_device()
            if self.device_index is None:
                logging.critical("No suitable audio input device found. Cannot listen.")
            else:
                logging.info(f"Auto-selected audio input device index: {self.device_index}")
        else:
            try:
                device_info = sd.query_devices(self.device_index)
                if int(device_info.get('max_input_channels', 0)) == 0:
                    raise IOError(f"Device index {self.device_index} is not an input device.")
                logging.info(f"Using specified audio input device index: {self.device_index} ({device_info.get('name')})")
            except Exception as e:
                logging.critical(f"Invalid --device-index {self.device_index}: {e}")
                raise

        self.messages: list[dict[str, str]] = []
        self.reset_history()
        self.last_known_token_count: int = 0


    def find_default_input_device(self) -> int | None:
        """Tries to find the default input device using sounddevice."""
        try:
            default_device = sd.query_devices(kind='input')
            if default_device and isinstance(default_device, dict):
                 index = int(default_device.get('index', -1))
                 if index >= 0:
                     return index
            raise IOError("Default device found, but has no index.")
        except Exception as e:
            logging.warning(f"Could not get default input device automatically: {e}. Searching all devices...")
            for i, dev in enumerate(sd.query_devices()):
                if int(dev.get('max_input_channels', 0)) > 0:
                    logging.warning(f"Using first available input device (index {i}): {dev['name']}")
                    return i
            logging.error("No available input device found after searching all devices.")
        return None

    def reset_history(self) -> None:
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.last_known_token_count = 0
        logging.debug("Conversation history reset.")

    def _tts_worker(self) -> None:
        consecutive_errors = 0
        while not self.tts_stop_event.is_set():
            text: str | None = None
            stream: sd.OutputStream | None = None
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text is None:  # Shutdown signal
                    break

                if self.piper_voice is None:
                    raise RuntimeError("Piper voice is not initialized.")

                self.is_speaking_event.set()

                stream = sd.OutputStream(
                    samplerate=self.piper_sample_rate,
                    device=self.args.piper_output_device_index,
                    channels=1,
                    dtype='int16'
                )
                stream.start()

                # Synthesize and play audio chunk by chunk
                for audio_chunk in self.piper_voice.synthesize(text):
                    if self.interrupt_event.is_set():
                        break
                    
                    # --- FINAL-FINAL-FINAL FIX: Cast the AudioChunk object to bytes ---
                    audio_np = np.frombuffer(bytes(audio_chunk), dtype=np.int16)
                    # --- END FIX ---
                    
                    stream.write(audio_np)

                consecutive_errors = 0

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_TTS_ERRORS:
                    logging.critical(f"TTS worker failed {MAX_TTS_ERRORS} times. Stopping.")
                    self.tts_has_failed.set()
                    with self.tts_queue.mutex: self.tts_queue.queue.clear()
                    self.tts_stop_event.set()
                    break
            finally:
                if stream:
                    stream.stop()
                    stream.close()

                if text is not None:
                    self.tts_queue.task_done()
                
                if self.tts_queue.empty() and not self.interrupt_event.is_set():
                    self.is_speaking_event.clear()

    def speak(self, text: str) -> None:
        if self.tts_has_failed.is_set():
            logging.error(f"Cannot speak: TTS engine failed. Text: '{text[:20]}...'")
            return

        logging.info(f"Assistant: {text}")
        
        if self.interrupt_event.is_set():
             with self.tts_queue.mutex:
                self.tts_queue.queue.clear()

        self.tts_queue.put(text)
        self.interrupt_event.clear()

    def wait_for_speech(self) -> None:
        if self.tts_has_failed.is_set():
             return
        logging.debug("Waiting for speech...")
        self.tts_queue.join()
        self.is_speaking_event.clear()
        logging.debug("Speech finished.")

    def stop_generation(self, log_message: str) -> None:
        logging.info(log_message)
        self.interrupt_event.set()
        
        sd.stop()

        with self.tts_queue.mutex:
            self.tts_queue.queue.clear()
        self.is_speaking_event.clear()
        self.is_processing_command.clear()

    def _audio_monitor_worker(self) -> None:
        """Monitors stream_buffer for barge-in during LLM/TTS."""
        logging.debug("Audio monitor started.")
        while self.is_processing_command.is_set():
            if self.interrupt_event.is_set():
                 time.sleep(0.01)
                 continue
            try:
                chunk: bytes = self.stream_buffer.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk:
                if self.is_speaking_event.is_set():
                    is_speech = self.vad.is_speech(chunk, RATE)
                    if is_speech:
                        self.stop_generation("User interruption (barge-in) detected!")
                self.stream_buffer.task_done()
        logging.debug("Audio monitor stopped.")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        """
        sounddevice stream callback.
        This runs in a separate thread, managed by sounddevice.
        It pushes audio data into our shared buffer.
        """
        if status:
            logging.warning(f"Audio stream callback status: {status}")
        try:
            self.stream_buffer.put_nowait(indata.tobytes())
        except queue.Full:
            now = time.time()
            if now - self.last_buffer_drop_warning > 5.0:
                logging.warning("Audio buffer is full, dropping audio chunks! (Callback is faster than consumer)")
                self.last_buffer_drop_warning = now

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        try:
            segments, info = self.whisper_model.transcribe(audio_np, language="en")
            full_text = "".join(segment.text for segment in segments)
            return full_text.strip()
        except Exception as e:
            logging.error(f"faster-whisper transcription error: {e}")
            return ""

    def record_command(self) -> npt.NDArray[np.float32] | None:
        """Records audio from the stream_buffer until silence."""
        logging.info("Listening for command...")
        frames: list[bytes] = []
        silent_chunks = 0
        is_speaking = False
        pre_buffer: list[bytes] = []
        timeout_chunks = self.pre_speech_timeout_chunks

        with self.stream_buffer.mutex:
            self.stream_buffer.queue.clear()

        CHUNK_DURATION_MS: float = 30.0

        while True:
            if self.interrupt_event.is_set():
                 logging.warning("Recording interrupted.")
                 return None
            try:
                data: bytes = self.stream_buffer.get(timeout=0.1)
                self.stream_buffer.task_done()

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
                        logging.debug("VAD: Still hearing speech...")
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
                        logging.warning(f"No speech detected after {self.args.listen_timeout}s, timing out.")
                        return None

            except queue.Empty:
                logging.debug("Buffer empty, waiting...")
                timeout_chunks -= int(0.1 * 1000 / CHUNK_DURATION_MS)
                if timeout_chunks <= 0:
                    logging.warning(f"Recording aborted (timeout).")
                    return None
            except Exception as e:
                logging.error(f"Unexpected error during recording: {e}")
                return None

        if not frames: return None
        audio_data: bytes = b''.join(frames)
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_NORMALIZATION
        return audio_np

    @contextmanager
    def _ollama_stream_manager(self, stream_response: Any):
        try:
            yield stream_response
        finally:
            if stream_response and hasattr(stream_response, 'close'):
                 stream_response.close()

    def _is_sentence_end(self, token: str) -> bool:
        return any(p in token for p in SENTENCE_END_PUNCTUATION)

    def _manage_history(self) -> None:
        if not self.messages or self.messages[0]['role'] != 'system':
             logging.error("History corrupted, resetting.")
             self.reset_history()
             return

        if self.last_known_token_count == 0 and len(self.messages) > 1:
            logging.warning("Last token count unknown, using turn-based fallback.")
            self._manage_history_turn_fallback()
            return

        if self.last_known_token_count > self.max_history_tokens and len(self.messages) > 3:
            if self.messages[1]['role'] == 'user' and self.messages[2]['role'] == 'assistant':
                del self.messages[1:3]
                logging.warning(f"History context ({self.last_known_token_count} tokens) exceeded limit ({self.max_history_tokens}). Pruning one exchange.")
                self.last_known_token_count = 0
            else:
                logging.error("History structure mismatch. Cannot prune tokens.")
        elif self.last_known_token_count > 0:
            logging.debug(f"History token count OK: {self.last_known_token_count}")

    def _manage_history_turn_fallback(self) -> None:
        current_context_length = len(self.messages) - 1
        current_turns = current_context_length // 2
        if current_turns > MAX_HISTORY_MESSAGES:
            turns_to_remove = current_turns - MAX_HISTORY_MESSAGES
            messages_to_remove = turns_to_remove * 2
            end_index_to_remove = 1 + messages_to_remove
            if end_index_to_remove >= len(self.messages):
                 logging.error(f"History pruning fallback error. Skipping.")
                 return
            self.messages = [self.messages[0]] + self.messages[end_index_to_remove:]
            new_turns = (len(self.messages) - 1) // 2
            logging.warning(f"History pruned (TURN FALLBACK). New size: {new_turns} turns.")

    def _update_history(self, user_text: str, assistant_response: str) -> None:
        if assistant_response.strip():
            self.messages.append({'role': 'assistant', 'content': assistant_response})
        else:
            if self.messages and self.messages[-1]['role'] == 'user' and self.messages[-1]['content'] == user_text:
                 self.messages.pop()
                 logging.warning("LLM response empty, user message rolled back.")
            elif self.messages:
                 logging.error(f"Failed to rollback user message.")

    def get_ollama_response_stream(self, user_text: str) -> None:
        self.is_processing_command.set()
        full_response = ""
        final_chunk: dict[str, Any] = {}
        monitor_thread: threading.Thread | None = None

        try:
            if self.ollama_client is None:
                logging.error("Ollama client not available. Cannot get response.")
                self.speak("I'm sorry, I'm not connected to my brain. Please check the Ollama server.")
                return

            self.messages.append({'role': 'user', 'content': user_text})
            self._manage_history()
            sentence_buffer = ""
            monitor_thread = threading.Thread(target=self._audio_monitor_worker, daemon=True)
            monitor_thread.start()
            response_stream: Any = None

            try:
                response_stream = self.ollama_client.chat(
                    model=self.args.ollama_model, messages=self.messages, stream=True
                )
            except (ollama.ResponseError, ollama.RequestError) as e:
                logging.error(f"Error connecting to Ollama: {e}")
                self.speak("I'm sorry, I seem to have lost my connection. Please check if the Ollama server is running.")
                full_response = "" 
                self.messages.pop() 
                return

            with self._ollama_stream_manager(response_stream) as stream:
                for chunk in stream:
                    if self.interrupt_event.is_set():
                        self.stop_generation("LLM stream stopped by user.")
                        break

                    if chunk.get('done', False):
                        final_chunk = chunk

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
            self.speak("I'm sorry, I encountered an error.")
            full_response = ""
        finally:
            self.is_processing_command.clear()
            if monitor_thread and monitor_thread.is_alive():
                monitor_thread.join(timeout=0.2)

            self._update_history(user_text, full_response)
            if final_chunk:
                self.last_known_token_count = final_chunk.get('prompt_eval_count', 0)
                logging.debug(f"Updated history. New prompt token count: {self.last_known_token_count}")
            elif not full_response:
                self.last_known_token_count = 0

    def process_user_command(self) -> bool:
        """Returns True if the assistant should exit."""
        if self.interrupt_event.is_set():
             logging.info("Resetting interrupt event.")
             self.interrupt_event.clear()
        
        audio_data: npt.NDArray[np.float32] | None = self.record_command()

        if audio_data is None:
            if not self.interrupt_event.is_set():
                self.speak("I didn't hear anything.")
                self.wait_for_speech()
            return False

        logging.info("Transcribing audio...")
        user_text: str = self.transcribe_audio(audio_data)
        if not user_text:
            logging.warning("Transcription failed.")
            self.speak("I'm sorry, I couldn't understand.")
            self.wait_for_speech()
            return False

        word_count = len(user_text.split())
        if word_count > self.args.max_words_per_command:
            logging.warning(f"Transcription rejected: {word_count} words.")
            self.speak("That was too long. Please try again.")
            self.wait_for_speech()
            return False

        logging.info(f"You: {user_text}")
        user_prompt: str = user_text.lower().strip().rstrip(".,!?")

        if "exit" in user_prompt or "goodbye" in user_prompt:
            self.speak("Goodbye!")
            self.wait_for_speech()
            return True
        if "new chat" in user_prompt or "reset chat" in user_prompt:
            self.reset_history()
            self.speak("Starting a new conversation.")
            self.wait_for_speech()
            return False

        logging.info(f"Sending to {self.args.ollama_model}...")
        self.get_ollama_response_stream(user_text)
        
        if not self.interrupt_event.is_set():
            self.wait_for_speech()

        self.interrupt_event.clear()
        return False

    def run(self) -> None:
        """The main loop of the assistant."""
        if self.device_index is None:
            logging.critical("Cannot run: No valid audio input device.")
            return
        
        if self.tts_has_failed.is_set():
            logging.critical("Cannot run: TTS engine failed to initialize.")
            return

        try:
            self.stream = sd.InputStream(
                samplerate=RATE,
                blocksize=CHUNK_SIZE,
                device=self.device_index,
                channels=CHANNELS,
                dtype=FORMAT_NP,
                callback=self._audio_callback
            )
            self.stream.start()
            logging.debug("sounddevice audio stream started.")

        except Exception as e:
            logging.critical(f"FATAL: Failed to open audio stream: {e}")
            return

        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

        try:
            while True:
                if self.tts_has_failed.is_set():
                     logging.critical("TTS engine failed. Shutting down.")
                     break

                if self.is_processing_command.is_set():
                    time.sleep(0.01)
                    continue

                audio_chunk: bytes | None = None
                try:
                    audio_chunk = self.stream_buffer.get(timeout=0.01)
                    if audio_chunk:
                        self.stream_buffer.task_done()
                    else:
                        continue
                except queue.Empty:
                    time.sleep(0.001)
                    continue

                if audio_chunk:
                    audio_np_int16: npt.NDArray[np.int16] = np.frombuffer(audio_chunk, dtype=np.int16)
                    prediction: dict[str, float] = self.oww_model.predict(audio_np_int16)
                    score = prediction.get(self.wakeword_model_key, 0)
                    
                    if score > 0.1: # Only log if there's *some* sound
                        logging.debug(f"Wakeword score: {score:.2f} (Threshold: {self.args.wakeword_threshold})")

                    if score > self.args.wakeword_threshold:
                        logging.info(f"Wakeword '{self.args.wakeword}' detected!")
                        self.oww_model.reset()
                        self.speak("Yes?")
                        self.wait_for_speech()

                        if self.process_user_command():
                            break

                        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up resources."""
        logging.info("Cleaning up resources...")

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logging.debug("sounddevice stream closed.")
            except Exception as e:
                logging.warning(f"Error closing audio stream: {e}")
            finally:
                self.stream = None

        if hasattr(self, 'whisper_model'):
            try:
                if self.whisper_device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
                    logging.info("Releasing CUDA memory...")
                    del self.whisper_model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    del self.whisper_model
                logging.debug("Whisper model released.")
            except Exception as e:
                logging.warning(f"Error releasing Whisper model: {e}")
            finally:
                 if 'whisper_model' in self.__dict__: del self.whisper_model

        if hasattr(self, 'tts_stop_event'):
            self.tts_stop_event.set()
            
            sd.stop() # Stop any final playback

            if hasattr(self, 'tts_queue'):
                self.tts_queue.put(None) # Send shutdown signal

            if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
                self.tts_thread.join(timeout=1.0)
                if self.tts_thread.is_alive():
                    logging.warning("TTS thread did not shut down cleanly.")