import argparse
import configparser
import logging
import sys
import os
import ollama
from typing import Tuple, Dict, Any
import pyaudio

# Import constants and helpers from audio_utils
from audio_utils import (
    DEFAULT_SETTINGS, DEFAULT_OLLAMA_HOST, list_audio_devices, list_tts_voices
)


# --- External Dependency Checks ---

def check_ollama_connectivity(host: str) -> bool:
    """Checks if the Ollama server is running and reachable."""
    try:
        ollama.Client(host=host).list()
        logging.info(f"Ollama server is reachable at {host}.")
        return True
    except Exception as e:
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

    # 2. Check Whisper model
    whisper_model = args.whisper_model
    if os.path.sep in whisper_model or '/' in whisper_model or '\\' in whisper_model:
        if not os.path.exists(whisper_model):
            logging.error(f"Whisper model file not found: '{whisper_model}'")
            success = False
        else:
            logging.debug(f"Whisper model file found: '{whisper_model}'")
    else:
        logging.debug(f"Whisper model is a standard name ('{whisper_model}'), relying on whisper package download/cache.")

    return success

# --- Argument/Config Loader ---

def load_config_and_args() -> Tuple[argparse.Namespace, argparse.Namespace]:
    """
    Loads configuration from config.ini, defines command-line arguments,
    and returns parsed arguments along with the list-command arguments.
    """
    config = configparser.ConfigParser()
    argparse_defaults: Dict[str, Any] = DEFAULT_SETTINGS.copy()

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

    # If listing is requested, we stop here after listing
    if list_args.list_devices or list_args.list_voices:
        if list_args.list_devices:
            p_audio = pyaudio.PyAudio()
            list_audio_devices(p_audio)
            p_audio.terminate()
        if list_args.list_voices:
            list_tts_voices()
        sys.exit(0)


    # 2. Configuration File Reading
    try:
        if config.read('config.ini'):
             logging.info("Loaded configuration from config.ini")
             
             # Configuration reading logic (same as original, just moved)
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
                 argparse_defaults['whisper_device'] = config.get('Functionality', 'whisper_device', fallback=argparse_defaults['whisper_device'])
                 argparse_defaults['max_history_tokens'] = config.getint('Functionality', 'max_history_tokens', fallback=argparse_defaults['max_history_tokens'])
                 
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

    # Argument definitions (using argparse_defaults as defaults)
    parser.add_argument('--ollama-model', type=str, default=argparse_defaults['ollama_model'], help='Ollama model (e.g., "llama3")')
    parser.add_argument('--ollama-host', type=str, default=argparse_defaults['ollama_host'], help='The URL for the Ollama server (e.g., "http://192.168.1.10:11434").')
    parser.add_argument('--whisper-model', type=str, default=argparse_defaults['whisper_model'], help='Whisper model (e.g., "tiny.en", "base.en").')
    parser.add_argument('--wakeword-model-path', type=str, default=argparse_defaults['wakeword_model_path'], help='Full path to the .onnx wakeword model.')
    parser.add_argument('--wakeword', type=str, default=argparse_defaults['wakeword'], help='Wakeword phrase.')
    parser.add_argument('--wakeword-threshold', type=float, default=argparse_defaults['wakeword_threshold'], help='Wakeword sensitivity (0.0-1.0).')
    parser.add_argument('--vad-aggressiveness', type=int, default=argparse_defaults['vad_aggressiveness'], choices=[0, 1, 2, 3], help='VAD aggressiveness (0=least, 3=most).')
    parser.add_argument('--silence-seconds', type=float, default=argparse_defaults['silence_seconds'], help='Seconds of silence before stopping recording.')
    parser.add_argument('--listen-timeout', type=float, default=argparse_defaults['listen_timeout'], help='Seconds to wait for speech after wakeword before timeout.')
    parser.add_argument('--pre-buffer-ms', type=int, default=argparse_defaults['pre-buffer-ms'], help='Milliseconds of audio to pre-buffer.')
    parser.add_argument('--system-prompt', type=str, default=argparse_defaults['system_prompt'], help='The system prompt for the Ollama model.')
    parser.add_argument('--device-index', type=int, default=argparse_defaults['device_index'], help='Index of the audio input device to use. (Use --list-devices to see options)')
    parser.add_argument('--tts-voice-id', type=str, default=argparse_defaults['tts_voice_id'], help='ID of the pyttsx3 voice to use. (Use --list-voices to see options)')
    parser.add_argument('--tts-volume', type=float, default=argparse_defaults['tts_volume'], help='TTS speaking volume (0.0 to 1.0).')
    parser.add_argument('--max-words-per-command', type=int, default=argparse_defaults['max_words_per_command'], help='Maximum number of words allowed in a command transcription.')
    parser.add_argument('--whisper-device', type=str, default=argparse_defaults['whisper_device'], choices=['cpu', 'cuda'], help="Device to use for Whisper transcription ('cpu' or 'cuda').")
    parser.add_argument('--max-history-tokens', type=int, default=argparse_defaults['max_history_tokens'], help='Maximum token count for conversation history (system message + all turns).')


    args = parser.parse_args()

    # 4. Check Local Files
    if not check_local_files_exist(args):
        logging.critical("FATAL: One or more local model files are missing. Cannot start assistant.")
        sys.exit(1)


    # Logging final effective settings
    logging.info("Starting assistant with the following effective settings:")
    for arg, value in vars(args).items():
        if arg in ['list_devices', 'list_voices', 'verbose']: continue
        
        # Explicitly log default host
        is_default_host = arg == 'ollama_host' and value == DEFAULT_OLLAMA_HOST
        
        if arg == 'system_prompt' and len(str(value)) > 100:
             logging.info(f"  --{arg}: '{str(value)[:100]}...'")
        elif is_default_host:
             logging.info(f"  --{arg}: {value} (Default)")
        else:
             logging.info(f"  --{arg}: {value}")
    logging.info("-" * 20)

    # list_args are not needed after this point but returned for completeness if necessary
    return args, list_args