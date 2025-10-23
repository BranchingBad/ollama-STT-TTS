import pyaudio
import pyttsx3
from typing import Any

# --- 1. Audio Settings (Constants) ---
FORMAT: int = pyaudio.paInt16       # 16-bit audio
CHANNELS: int = 1                 # Mono
RATE: int = 16000                 # 16kHz sample rate
CHUNK_DURATION_MS: int = 30       # 30ms chunks for VAD (Must be 10, 20, or 30 for webrtcvad)
CHUNK_SIZE: int = int(RATE * CHUNK_DURATION_MS / 1000) # 480 frames per chunk (16000 * 0.03 = 480)
INT16_NORMALIZATION: float = 32768.0 # Normalization factor for int16
SENTENCE_END_PUNCTUATION: list[str] = ['.', '?', '!', '\n']
MAX_TTS_ERRORS: int = 5           # Max consecutive errors before stopping TTS worker
DEFAULT_OLLAMA_HOST: str = 'http://localhost:11434' # Define default for logging check
MAX_HISTORY_MESSAGES: int = 20    # Max *turns* (user/assistant pairs) to keep (Fallback)
STREAM_READ_TIMEOUT: float = 0.05 # Timeout for non-blocking read in the main loop (seconds)

# --- 2. Centralized Configuration Defaults ---
DEFAULT_SETTINGS: dict[str, Any] = {
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
    'device_index': None, # Must be None or int
    'tts_voice_id': None,
    'tts_volume': 1.0, 
    'max_words_per_command': 60, 
    'whisper_device': 'cpu',
    'max_history_tokens': 4096,
}


# --- 3. PyAudio and TTS Helpers ---
def list_audio_devices(p_audio: pyaudio.PyAudio) -> None:
    """Lists all available audio input devices."""
    print("\n--- Available Audio Input Devices ---")
    try:
        # Assuming index 0 is the default host API
        info = p_audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            dev = p_audio.get_device_info_by_host_api_device_index(0, i)
            if dev.get('maxInputChannels') > 0:
                print(f"  Index {i}: {dev.get('name')}")
    except Exception as e:
        print(f"Error listing devices: {e}")
    print("------------------------------------\n")
    
def list_tts_voices() -> None:
    """Lists all available pyttsx3 voices, prioritizing English ones."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    print("\n--- Available TTS Voices (English/General) ---")
    for voice in voices:
        # Check for English or general language ID/tag
        is_english_or_general = False
        if voice.id and ('en' in voice.id.lower()):
            is_english_or_general = True
        
        if not is_english_or_general and voice.languages:
             for lang in voice.languages:
                 # Check for 'en' (English) or 'gmw' (Germanic/general) language codes
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