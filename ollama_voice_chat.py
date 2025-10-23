#!/usr/bin/env python3

"""
ollama_voice_chat.py

A simple, hands-free Python voice assistant that runs 100% locally.
This script uses openwakeword for wakeword detection, webrtcvad for silence
detection, OpenAI's Whisper for transcription, and Ollama for generative AI
responses.

Improvements:
- Added audio device selection (--list-devices, --device-index).
- Added TTS voice selection (--list-voices, --tts-voice-id).
- Added "Thinking..." feedback to reduce perceived latency.
- Added verbose logging option (-v, --verbose).
- Fixed wakeword model path to accept a full path.

Further Improvements (in this version):
- Implemented Ollama streaming for sentence-by-sentence TTS output,
  dramatically
  reducing time-to-first-speech.
- Made "Thinking..." feedback more accurate by placing it *after*
  transcription.
- Made wakeword model key detection robust by querying the loaded model
  directly,
  instead of parsing filenames.
- Made audio device listing more robust.
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
        self.system_prompt = args.system_prompt

        # Calculate derived audio settings
        self.silence_chunks: int = int(args.silence_seconds * 1000 / CHUNK_DURATION_MS)
        self.pre_speech_timeout_chunks: int = int(args.listen_timeout * 1000 / CHUNK_DURATION_MS)
        self.pre_buffer_size_ms: int = args.pre_buffer_ms
        self.pre_buffer_size_chunks: int = int(self.pre_buffer_size_ms / CHUNK_DURATION_MS)

        logging.info("Loading models...")

        # Wakeword Model
        logging.info(f"Loading openwakeword model from: {args.wakeword_model_path}...")
        try:
            self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
            
            # --- IMPROVEMENT: Robust wakeword key detection ---
            # Instead of parsing the filename, ask the model what key it loaded.
            self.wakeword_model_key = list(self.oww_model.models.keys())[0]
            logging.info(f"Loaded wakeword model with key: '{self.wakeword_model_key}'")
            
        except Exception as e:
            logging.error(f"Error loading openwakeword model '{args.wakeword_model_path}': {e}")
            logging.error("Ensure the .onnx model file exists at the specified path.")
            raise

        # Whisper Model
        logging.info(f"Loading Whisper model: {args.whisper_model}...")
        try:
            self.whisper_model = whisper.load_model(args.whisper_model)
        except Exception as e:
            logging.error(f"Error loading Whisper model '{args.whisper_model}': {e}")
            logging.error("Ensure the model name is correct and you have enough memory.")
            raise

        # TTS Engine
        logging.info("Initializing TTS engine...")
        self.tts_engine = pyttsx3.init()
        if args.tts_voice_id:
            try:
                self.tts_engine.setProperty('voice', args.tts_voice_id)
                logging.info(f"Set TTS voice ID to: {args.tts_voice_id}")
            except Exception as e:
                logging.warning(f"Could not set TTS voice ID: {e}. Using default.")
        
        self.tts_queue = queue.Queue()
        self.tts_stop_event = threading.Event()
        self.is_speaking_event = threading.Event()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

        # VAD
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)

        # PyAudio
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # Validate and select the input device
        self.device_index = args.device_index
        if self.device_index is None:
            self.device_index = self.find_default_input_device()
            if self.device_index is None:
                logging.critical("No suitable input device found.")
                raise IOError("No input audio device found.")
            logging.info(f"No --device-index provided. Auto-selected device: {self.device_index}")
        else:
            try:
                self.audio.get_device_info_by_index(self.device_index)
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
            return int(default_device_info['index'])
        except IOError as e:
            logging.warning(f"Could not get default input device: {e}. Searching all devices...")
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
                text = self.tts_queue.get(timeout=1.0)
                if text is None: # Stop sentinel
                    break

                self.is_speaking_event.set()
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
            finally:
                if text is not None:
                    self.tts_queue.task_done()
                self.is_speaking_event.clear()

    def speak(self, text: str) -> None:
        """Adds text to the TTS queue to be spoken by the worker thread."""
        logging.info(f"Assistant: {text}")
        self.tts_queue.put(text)

    def wait_for_speech(self) -> None:
        """Blocks until the TTS queue is empty and all speech is finished."""
        logging.debug("Waiting for speech to finish...")
        self.tts_queue.join()
        logging.debug("Speech finished.")

    def transcribe_audio(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Transcribes audio data (NumPy array) using Whisper."""
        try:
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
        self.messages.append({'role': 'user', 'content': text})
        
        full_response = ""
        sentence_buffer = ""

        try:
            if not self.messages or self.messages[0]['role'] != 'system':
                 logging.warning("Messages list did not start with system prompt. Re-adding.")
                 self.messages.insert(0, {'role': 'system', 'content': self.system_prompt})

            # --- IMPROVEMENT: Use stream=True ---
            response_stream = ollama.chat(
                model=self.args.ollama_model,
                messages=self.messages,
                stream=True
            )

            for chunk in response_stream:
                token = chunk['message']['content']
                full_response += token
                sentence_buffer += token

                if self._is_sentence_end(token):
                    # We have a full sentence, speak it.
                    self.speak(sentence_buffer.strip())
                    sentence_buffer = ""
            
            # Speak any remaining text in the buffer
            if sentence_buffer.strip():
                self.speak(sentence_buffer.strip())

            # Add the complete response to history
            self.messages.append({'role': 'assistant', 'content': full_response})

        except ollama.ResponseError as e:
            logging.error(f"Ollama Response Error: {e.error}")
            self.speak("I'm sorry, I received an error from Ollama. Please check the console.")
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            self.speak("I'm sorry, I couldn't connect to Ollama. Is the Ollama server running?")


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

        while True:
            try:
                if not self.stream:
                    logging.error("Audio stream is not open.")
                    return None

                data = self.stream.read(CHUNK_SIZE)
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
                logging.error(f"Error reading from audio stream: {e}")
                return None

        audio_data: bytes = b''.join(frames)
        audio_np: npt.NDArray[np.float32] = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_NORMALIZATION

        return audio_np

    def run(self) -> None:
        """
        The main loop of the assistant. Listens for wakeword, then
        records, transcribes, and responds.
        """
        try:
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

                try:
                    audio_chunk = self.stream.read(CHUNK_SIZE)
                except IOError as e:
                    logging.error(f"Error reading from audio stream in main loop: {e}")
                    logging.error("This may be due to a microphone disconnection. Stopping.")
                    break

                if not self.is_speaking_event.is_set():
                    audio_np_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                    prediction = self.oww_model.predict(audio_np_int16)

                    # --- IMPROVEMENT: Use robust key and safer .get() ---
                    if prediction.get(self.wakeword_model_key, 0) > self.args.wakeword_threshold:
                        logging.info(f"Wakeword '{self.args.wakeword}' detected!")
                        self.oww_model.reset()

                        self.stream.stop_stream()
                        self.speak("Yes?")
                        self.wait_for_speech()
                        self.stream.start_stream()

                        audio_data = self.record_command()
                        self.stream.stop_stream()

                        if audio_data is None:
                            self.speak("I didn't hear anything.")
                            self.wait_for_speech()
                            logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                            self.stream.start_stream()
                            continue
                        
                        logging.info("Transcribing audio...")
                        user_text = self.transcribe_audio(audio_data)

                        if user_text:
                            logging.info(f"You: {user_text}")
                            
                            # --- IMPROVEMENT: Move "Thinking..." to *after* transcription ---
                            # This feels more responsive and accurate.
                            self.speak("Thinking...")
                            
                            user_prompt = user_text.lower().strip().rstrip(".,!?")

                            if "exit" in user_prompt or "goodbye" in user_prompt:
                                self.speak("Goodbye!")
                                self.wait_for_speech()
                                break
                            if "new chat" in user_prompt or "reset chat" in user_prompt:
                                self.speak("Starting a new conversation.")
                                self.wait_for_speech()
                                self.messages = [
                                    {'role': 'system', 'content': self.system_prompt}
                                ]
                                logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                                self.stream.start_stream()
                                continue

                            logging.info(f"Sending to {self.args.ollama_model}...")
                            
                            # --- IMPROVEMENT: Call streaming function ---
                            # This function now handles its own speech.
                            self.get_ollama_response_stream(user_text)
                            
                            # Wait for the *entire* streamed response to finish speaking
                            self.wait_for_speech() 

                        else:
                            logging.warning("Transcription failed.")
                            self.speak("I'm sorry, I didn't catch that.")
                            self.wait_for_speech()

                        logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")
                        self.stream.start_stream()

        except KeyboardInterrupt:
            logging.info("\nStopping assistant...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleans up PyAudio and stops the TTS thread."""
        logging.info("Cleaning up resources...")
        
        if self.stream:
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        self.tts_stop_event.set()
        self.tts_queue.put(None)
        if hasattr(self, 'tts_thread') and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=2.0)

        if self.audio:
            self.audio.terminate()

# --- Helper Functions for main() ---

def list_audio_devices(p_audio: pyaudio.PyAudio):
    """Lists all available audio input devices."""
    logging.info("Available Audio Input Devices:")
    
    # --- IMPROVEMENT: Iterate all device indices, not just from one API ---
    num_devices = p_audio.get_device_count()
    
    found_devices = False
    for i in range(num_devices):
        try:
            dev = p_audio.get_device_info_by_index(i)
            if int(dev.get('maxInputChannels', 0)) > 0: # Check if it's an input device
                logging.info(f"  Index {i}: {dev.get('name')} (Channels: {dev.get('maxInputChannels')})")
                found_devices = True
        except IOError as e:
            logging.warning(f"Could not query device index {i}: {e}")
            
    if not found_devices:
        logging.warning("No audio input devices found.")

def list_tts_voices():
    """Lists all available pyttsx3 voices."""
    logging.info("Available TTS Voices:")
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for i, voice in enumerate(voices):
            logging.info(f"  Voice {i}: ID: {voice.id}")
            logging.info(f"    Name: {voice.name}")
            logging.info(f"    Lang: {voice.languages}")
            logging.info(f"     Age: {voice.age}")
        engine.stop()
    except Exception as e:
        logging.error(f"Could not list TTS voices: {e}")

def main() -> None:
    """
    Parses arguments, reads config, and runs the VoiceAssistant.
    """
    config = configparser.ConfigParser()
    argparse_defaults = {
        'ollama_model': 'llama3',
        'whisper_model': 'base.en',
        'wakeword_model_path': 'hey_glados.onnx',
        'wakeword': 'hey glados',
        'wakeword_threshold': 0.6,
        'vad_aggressiveness': 2,
        'silence_seconds': 0.7,
        'listen_timeout': 6.0,
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

    if list_args.list_devices:
        list_audio_devices(pyaudio.PyAudio())
        sys.exit(0)
        
    if list_args.list_voices:
        list_tts_voices()
        sys.exit(0)

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
                 argparse_defaults['device_index'] = config.getint('Functionality', 'device_index', fallback=argparse_defaults['device_index'])
                 argparse_defaults['tts_voice_id'] = config.get('Functionality', 'tts_voice_id', fallback=argparse_defaults['tts_voice_id'])
        else:
             logging.warning("config.ini not found, using default settings.")
    except configparser.Error as e:
        logging.error(f"Error reading config.ini: {e}. Using default settings.")

    parser = argparse.ArgumentParser(
        description="Ollama STT-TTS Voice Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[temp_parser, list_parser]
    )

    parser.add_argument('--ollama-model', type=str, default=argparse_defaults['ollama_model'], help='Ollama model (e.g., "llama3")')
    parser.add_argument('--whisper-model', type=str, default=argparse_defaults['whisper_model'], help='Whisper model (e.g., "tiny.en", "base.en")')
    parser.add_argument('--wakeword-model-path', type=str, default=argparse_defaults['wakeword_model_path'], help='Full path to the .onnx wakeword model.')
    parser.add_argument('--wakeword', type=str, default=argparse_defaults['wakeword'], help='Wakeword phrase.')
    parser.add_argument('--wakeword-threshold', type=float, default=argparse_defaults['wakeword_threshold'], help='Wakeword sensitivity (0.0-1.0).')
    parser.add_argument('--vad-aggressiveness', type=int, default=argparse_defaults['vad_aggressiveness'], choices=[0, 1, 2, 3], help='VAD aggressiveness (0=least, 3=most).')
    parser.add_argument('--silence-seconds', type=float, default=argparse_defaults['silence_seconds'], help='Seconds of silence before stopping recording.')
    parser.add_argument('--listen-timeout', type=float, default=argparse_defaults['listen_timeout'], help='Seconds to wait for speech before timeout.')
    parser.add_argument('--pre-buffer-ms', type=int, default=argparse_defaults['pre_buffer_ms'], help='Milliseconds of audio to pre-buffer.')
    parser.add_argument('--system-prompt', type=str, default=argparse_defaults['system_prompt'], help='The system prompt for the Ollama model.')
    parser.add_argument('--device-index', type=int, default=argparse_defaults['device_index'], help='Index of the audio input device to use. (Use --list-devices to see options)')
    parser.add_argument('--tts-voice-id', type=str, default=argparse_defaults['tts_voice_id'], help='ID of the pyttsx3 voice to use. (Use --list-voices to see options)')

    args = parser.parse_args()

    logging.info("Starting assistant with the following effective settings:")
    for arg, value in vars(args).items():
        if arg in ['list_devices', 'list_voices']: continue
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
    main()