import sounddevice as sd
import pyttsx3
from typing import Any

# --- 1. Audio Settings (Constants) ---
FORMAT_NP: str = 'int16'          # Data type for sounddevice
CHANNELS: int = 1                 # Mono
RATE: int = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS: int = 30       # 30ms chunks for VAD
CHUNK_SIZE: int = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames
INT16_NORMALIZATION: float = 32768.0 # Normalization factor for int16
SENTENCE_END_PUNCTUATION: list[str] = ['.', '?', '!', '\n']
MAX_TTS_ERRORS: int = 5
DEFAULT_OLLAMA_HOST: str = 'http://localhost:11434'
MAX_HISTORY_MESSAGES: int = 20
STREAM_READ_TIMEOUT: float = 0.05 

# --- 2. Centralized Configuration Defaults ---
DEFAULT_SETTINGS: dict[str, Any] = {
    'ollama_model': 'llama3',
    'whisper_model': 'tiny.en',
    'wakeword_model_path': 'hey_glados.onnx',
    'ollama_host': DEFAULT_OLLAMA_HOST,
    'wakeword': 'hey glados',
    'wakeword_threshold': 0.45, 
    'vad_aggressiveness': 2,   
    'silence_seconds': 0.5,    
    'listen_timeout': 4.0,     
    'pre_buffer_ms': 400,
    'system_prompt': 'You are a friendly, concise, and intelligent voice assistant named GLaDOS. Keep your responses short and witty.',
    'device_index': None, 
    'tts_voice_id': None,
    'tts_volume': 1.0, 
    'max_words_per_command': 60, 
    'whisper_device': 'cpu',
    'whisper_compute_type': 'int8',
    'max_history_tokens': 4096,
}

# --- 3. Audio Helpers (Updated for sounddevice) ---
def list_audio_devices() -> None:
    """Lists all available audio input devices using sounddevice."""
    print("\n--- Available Audio Input Devices (sounddevice) ---")
    try:
        devices = sd.query_devices()
        input_devices_found = False
        for i, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                print(f"  Index {i}: {dev.get('name')}")
                input_devices_found = True
        if not input_devices_found:
            print("  No input devices found.")
    except Exception as e:
        print(f"Error listing devices: {e}")
    print("-------------------------------------------------\n")

def list_tts_voices() -> None:
    """Lists all available pyttsx3 voices."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("\n--- Available TTS Voices (English/General) ---")
    for voice in voices:
        is_english_or_general = False
        if voice.id and ('en' in voice.id.lower()):
            is_english_or_general = True
        
        if not is_english_or_general and voice.languages:
             for lang in voice.languages:
                 if 'en' in lang.lower() or 'gmw' in lang.lower(): 
                     is_english_or_general = True
                     break

        if is_english_or_general:
            print(f"  ID: {voice.id}")
            print(f"    - Name: {voice.name}")
            print(f"    - Language: {voice.languages[0] if voice.languages else 'N/A'}")
            gender_info = voice.gender if hasattr(voice, 'gender') else 'N/A'
            print(f"    - Gender: {gender_info}")
    print("--------------------------------------------\n")