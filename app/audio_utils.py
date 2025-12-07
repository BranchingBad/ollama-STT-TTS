import sounddevice as sd
from typing import Any
import sys # Import sys for sys.stdout.write

# --- 1. Audio Settings (Constants) ---
FORMAT_NP: str = 'int16'          # Data type for sounddevice
CHANNELS: int = 1                 # Mono
RATE: int = 16000                 # 16kHz sample rate (for VAD and Whisper)
CHUNK_DURATION_MS: int = 30       # 30ms chunks for VAD
CHUNK_SIZE: int = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames
INT16_MAX: float = 32768.0        # Normalization factor for int16
SENTENCE_END_PUNCTUATION: list[str] = ['.', '?', '!', '\n']
MAX_TTS_ERRORS: int = 5
MAX_HISTORY_MESSAGES: int = 20

# FIX #2: Configurable audio buffer size with larger default (200 instead of 100)
DEFAULT_AUDIO_BUFFER_SIZE: int = 200

# --- 2. Centralized Configuration Defaults ---
DEFAULT_SETTINGS: dict[str, Any] = {
    'ollama_model': 'llama3',
    'whisper_model': 'base.en',
    'wakeword_model_path': 'models/hey_jarvis_v2.onnx',
    'piper_model_path': 'models/en_US-lessac-medium.onnx',
    'ollama_host': 'http://localhost:11434',
    'wakeword': 'hey jarvis',
    'wakeword_threshold': 0.35,
    'vad_aggressiveness': 2,
    'silence_seconds': 0.3,
    'listen_timeout': 4.0,
    'pre_buffer_ms': 400,
    'system_prompt': 'You are a friendly, concise, and intelligent voice assistant named Jarvis. Keep your responses short and witty.',
    'device_index': None,
    'piper_output_device_index': None,
    'max_words_per_command': 60,
    'whisper_device': 'cpu',
    'whisper_compute_type': 'int8',
    'max_history_tokens': 2048,
    'audio_buffer_size': DEFAULT_AUDIO_BUFFER_SIZE,  # FIX #2: Added buffer size config
}       

# --- 3. Audio Helpers (Updated for sounddevice) ---
def list_audio_input_devices() -> None:
    """Lists all available audio input devices using sounddevice."""
    sys.stdout.write("\n--- Available Audio Input Devices (sounddevice) ---\n")
    try:
        devices = sd.query_devices()
        input_devices_found = False
        for i, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                sys.stdout.write(f"  Index {i}: {dev.get('name')}\n")
                input_devices_found = True
        if not input_devices_found:
            sys.stdout.write("  No input devices found.\n")
    except Exception as e:
        sys.stdout.write(f"Error listing input devices: {e}\n")
    sys.stdout.write("-------------------------------------------------\n")

def list_audio_output_devices() -> None:
    """Lists all available audio output devices using sounddevice."""
    sys.stdout.write("\n--- Available Audio Output Devices (sounddevice) ---\n")
    try:
        devices = sd.query_devices()
        output_devices_found = False
        for i, dev in enumerate(devices):
            if dev.get('max_output_channels', 0) > 0:
                sys.stdout.write(f"  Index {i}: {dev.get('name')}\n")
                output_devices_found = True
        if not output_devices_found:
            sys.stdout.write("  No output devices found.\n")
    except Exception as e:
        sys.stdout.write(f"Error listing output devices: {e}\n")
    sys.stdout.write("--------------------------------------------------\n")