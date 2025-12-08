#!/usr/bin/env python3

"""
config_manager.py

Handles loading configuration from config.ini and command-line arguments.

FIXES APPLIED:
- Added path sanitization for security (High Priority #16)
- Improved file path validation
- Added system prompt file size limit
"""

import configparser
import argparse
import logging
import sys
import os
import ollama
from typing import Any, Tuple, Optional

# Import defaults and helpers from audio_utils
try:
    from audio_utils import (
        DEFAULT_SETTINGS,
        list_audio_input_devices,
        list_audio_output_devices
    )
except ImportError:
    print("FATAL ERROR: Could not import from audio_utils.py. Ensure the file is present.")
    sys.exit(1)

# --- Start of new robust path handling logic ---

# The absolute path to the project's root directory.
# This assumes the script is located in the 'app/' directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE_NAME = os.path.join(PROJECT_ROOT, 'config.ini')

# FIX #16: Security constants for path validation
MAX_SYSTEM_PROMPT_FILE_SIZE = 10 * 1024  # 10KB limit for system prompt files
ALLOWED_MODEL_DIRECTORIES = [os.path.join(PROJECT_ROOT, 'models'), os.path.join(PROJECT_ROOT, 'Models')]

def sanitize_file_path(file_path: str, description: str = "file") -> str:
    """
    Sanitizes and validates file paths. If a relative path is given,
    it's resolved relative to the project root.
    
    Args:
        file_path: The path to sanitize
        description: Description of the file for error messages
    
    Returns:
        Absolute path if valid
        
    Raises:
        ValueError: If path is invalid or potentially malicious
    """
    if not file_path:
        raise ValueError(f"Empty {description} path provided")
    
    original_path = file_path # Keep original for checks

    # If the path is not absolute, treat it as relative to the project root
    if not os.path.isabs(file_path):
        file_path = os.path.join(PROJECT_ROOT, file_path)
    
    # Normalize the path to resolve '..' etc. and get a clean, absolute path
    abs_path = os.path.normpath(file_path)
    
    # Security check: Prevent path traversal attacks.
    # Check if the normalized path is still within the project root.
    # We allow paths outside the project root ONLY if the user provided an absolute path initially.
    if not abs_path.startswith(PROJECT_ROOT) and not os.path.isabs(original_path):
        raise ValueError(f"Invalid {description} path: Path traversal detected for '{original_path}'")

    # For model files, ensure they're in the allowed models/ directory if a relative path was given
    if 'model' in description.lower() and not os.path.isabs(original_path):
        path_valid = False
        for allowed_dir in ALLOWED_MODEL_DIRECTORIES:
            if abs_path.startswith(allowed_dir):
                path_valid = True
                break
        if not path_valid:
            raise ValueError(f"Invalid {description} path: '{original_path}' must be in the 'models/' directory.")
            
    return abs_path

def get_ollama_client(ollama_host: str) -> Optional[ollama.Client]:
    """
    Tries to connect to the Ollama server and returns a client instance.
    Returns None if the connection fails.
    """
    logging.info(f"Attempting to connect to Ollama at {ollama_host}...")
    try:
        client = ollama.Client(host=ollama_host)
        client.list()
        logging.info("Ollama server connection successful.")
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Ollama at {ollama_host}: {e}")
        logging.warning("Please ensure Ollama is running and the 'ollama_host' in config.ini is correct.")
        return None

# Define a custom type converter for device indices that handles 'none'
def device_index_type(value: str) -> Optional[int]:
    """Converts a string argument to an int index or None."""
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid device index: '{value}'. Must be an integer or 'none'.")

def load_config_and_args() -> Tuple[argparse.Namespace, configparser.ConfigParser, bool]:
    """
    Loads settings from config.ini, parses command-line arguments,
    and sets up logging. Paths are resolved relative to the project root.
    
    Returns: A tuple containing (parsed arguments, config object, should_exit_flag)
    """

    config = configparser.ConfigParser()
    config_loaded = False
    if os.path.exists(CONFIG_FILE_NAME):
        config.read(CONFIG_FILE_NAME)
        logging.info(f"Loaded configuration from {CONFIG_FILE_NAME}")
        config_loaded = True
    else:
        logging.warning(f"{os.path.basename(CONFIG_FILE_NAME)} not found. Using default settings and CLI args.")

    config_models = config['Models'] if 'Models' in config else {}
    config_func = config['Functionality'] if 'Functionality' in config else {}
    config_perf = config['Performance'] if 'Performance' in config else {}

    def get_config_val(section: configparser.SectionProxy, key: str, default: Any, type_converter: type) -> Any:
        if not config_loaded:
             return default

        # configparser's getboolean is robust, so use it directly for bools
        if type_converter == bool:
            try:
                return section.getboolean(key, fallback=default)
            except ValueError: # Handle cases where boolean value is malformed
                logging.warning(f"Invalid boolean value for '{key}' in config.ini. Using default: {default}")
                return default
        
        raw_val = section.get(key) # Get raw value first

        processed_val = None
        if raw_val is not None:
            # Strip inline comments for string values
            if isinstance(raw_val, str):
                processed_val = raw_val.split('#')[0].strip()
            else:
                processed_val = raw_val # Keep non-string values as is (configparser usually returns strings, but for safety)
        
        # If after processing, we have an empty string or None, use the default
        if processed_val is None or (isinstance(processed_val, str) and processed_val == ''):
            processed_val = default

        try:
            # For int type, handle 'none' string specially if it's meant to be None
            if type_converter == int and isinstance(processed_val, str) and processed_val.lower() == 'none':
                return None
            
            # Attempt to convert to the target type
            return type_converter(processed_val)
        except (ValueError, TypeError):
            # Log specific warning for non-boolean type conversion issues
            logging.warning(f"Invalid value '{raw_val}' for '{key}' in config.ini. Using default: {default}")
            return default

    parser = argparse.ArgumentParser(description="A hands-free voice assistant for Ollama.")

    parser.add_argument('--list-devices', action='store_true', help="List available audio input devices and exit.")
    parser.add_argument('--list-output-devices', action='store_true', help="List available audio output devices and exit.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging.")

    model_group = parser.add_argument_group('Models')
    model_group.add_argument('--ollama-model', type=str, help="Name of the Ollama model to use.")
    model_group.add_argument('--whisper-model', type=str, help="Name of the faster-whisper model.")
    model_group.add_argument('--wakeword-model-path', type=str, help="Path to the .onnx wakeword model file.")
    model_group.add_argument('--piper-model-path', type=str, help="Path to the .onnx Piper TTS model file.")
    model_group.add_argument('--ollama-host', type=str, help="URL of the Ollama server.")

    func_group = parser.add_argument_group('Functionality')
    func_group.add_argument('--wakeword', type=str, help="The wakeword phrase.")
    func_group.add_argument('--wakeword-threshold', type=float, help="Wakeword detection threshold (0.0 to 1.0).")
    func_group.add_argument('--vad-aggressiveness', type=int, help="VAD aggressiveness (0-3).")
    func_group.add_argument('--silence-seconds', type=float, help="Seconds of silence to detect end of speech.")
    func_group.add_argument('--listen-timeout', type=float, help="Seconds to wait for speech before timing out.")
    func_group.add_argument('--pre-buffer-ms', type=int, help="Milliseconds of audio to keep before speech starts.")
    func_group.add_argument('--gain', type=float, help="Input gain to apply to the microphone volume.")
    func_group.add_argument('--system-prompt', type=str, help="The system prompt or a path to a .txt file.")
    func_group.add_argument('--device-index', type=device_index_type, help="Index of the audio input device.")
    func_group.add_argument('--piper-output-device-index', type=device_index_type, help="Index of the audio output device.")
    func_group.add_argument('--max-words-per-command', type=int, help="Maximum words allowed in a command.")
    func_group.add_argument('--max-phrase-duration', type=float, help="Maximum duration for a spoken phrase in seconds.")
    func_group.add_argument('--whisper-device', type=str, help="Device for Whisper (e.g., 'cpu', 'cuda').")
    func_group.add_argument('--whisper-compute-type', type=str, help="Compute type for Whisper (e.g., 'int8').")
    func_group.add_argument('--whisper-avg-logprob', type=float, help="Whisper avg_logprob threshold.")
    func_group.add_argument('--whisper-no-speech-prob', type=float, help="Whisper no_speech_prob threshold.")
    func_group.add_argument('--max-history-tokens', type=int, help="Maximum token context for chat history.")
    func_group.add_argument('--audio-buffer-size', type=int, help="Size of the audio buffer queue.")
    func_group.add_argument('--trim-wake-word', action='store_true', help="Enable trimming the wake word from the start of transcription.")

    perf_group = parser.add_argument_group('Performance')
    perf_group.add_argument('--gc-interval', type=int, help="Force garbage collection every N conversations.")
    perf_group.add_argument('--memory-profiling', action='store_true', help="Enable memory profiling in debug mode.")


    parser.set_defaults(
        ollama_model=get_config_val(config_models, 'ollama_model', DEFAULT_SETTINGS['ollama_model'], str),
        whisper_model=get_config_val(config_models, 'whisper_model', DEFAULT_SETTINGS['whisper_model'], str),
        wakeword_model_path=get_config_val(config_models, 'wakeword_model_path', DEFAULT_SETTINGS['wakeword_model_path'], str),
        piper_model_path=get_config_val(config_models, 'piper_model_path', DEFAULT_SETTINGS['piper_model_path'], str),
        ollama_host=get_config_val(config_models, 'ollama_host', DEFAULT_SETTINGS['ollama_host'], str),
        wakeword=get_config_val(config_func, 'wakeword', DEFAULT_SETTINGS['wakeword'], str),
        wakeword_threshold=get_config_val(config_func, 'wakeword_threshold', DEFAULT_SETTINGS['wakeword_threshold'], float),
        vad_aggressiveness=get_config_val(config_func, 'vad_aggressiveness', DEFAULT_SETTINGS['vad_aggressiveness'], int),
        silence_seconds=get_config_val(config_func, 'silence_seconds', DEFAULT_SETTINGS['silence_seconds'], float),
        listen_timeout=get_config_val(config_func, 'listen_timeout', DEFAULT_SETTINGS['listen_timeout'], float),
        pre_buffer_ms=get_config_val(config_func, 'pre_buffer_ms', DEFAULT_SETTINGS['pre_buffer_ms'], int),
        gain=get_config_val(config_func, 'gain', DEFAULT_SETTINGS['gain'], float),
        system_prompt=get_config_val(config_func, 'system_prompt', DEFAULT_SETTINGS['system_prompt'], str),
        device_index=get_config_val(config_func, 'device_index', DEFAULT_SETTINGS['device_index'], int),
        piper_output_device_index=get_config_val(config_func, 'piper_output_device_index', DEFAULT_SETTINGS['piper_output_device_index'], int),
        max_words_per_command=get_config_val(config_func, 'max_words_per_command', DEFAULT_SETTINGS['max_words_per_command'], int),
        max_phrase_duration=get_config_val(config_func, 'max_phrase_duration', DEFAULT_SETTINGS['max_phrase_duration'], float),
        whisper_device=get_config_val(config_func, 'whisper_device', DEFAULT_SETTINGS['whisper_device'], str),
        whisper_compute_type=get_config_val(config_func, 'whisper_compute_type', DEFAULT_SETTINGS['whisper_compute_type'], str),
        whisper_avg_logprob=get_config_val(config_func, 'whisper_avg_logprob', DEFAULT_SETTINGS['whisper_avg_logprob'], float),
        whisper_no_speech_prob=get_config_val(config_func, 'whisper_no_speech_prob', DEFAULT_SETTINGS['whisper_no_speech_prob'], float),
        max_history_tokens=get_config_val(config_func, 'max_history_tokens', DEFAULT_SETTINGS['max_history_tokens'], int),
        audio_buffer_size=get_config_val(config_func, 'audio_buffer_size', DEFAULT_SETTINGS['audio_buffer_size'], int),
        trim_wake_word=get_config_val(config_func, 'trim_wake_word', DEFAULT_SETTINGS['trim_wake_word'], bool),
        gc_interval=get_config_val(config_perf, 'gc_interval', DEFAULT_SETTINGS['gc_interval'], int),
        memory_profiling=get_config_val(config_perf, 'memory_profiling', DEFAULT_SETTINGS['memory_profiling'], bool)
    )

    args = parser.parse_args()
    should_exit_flag = False

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("DEBUG logging enabled.")

    # Sanitize and resolve all relevant file paths
    try:
        args.wakeword_model_path = sanitize_file_path(args.wakeword_model_path, "wakeword model")
        args.piper_model_path = sanitize_file_path(args.piper_model_path, "Piper TTS model")
        
        # Handle system prompt, which could be a string or a file path
        # Try to resolve it as a file path first
        potential_path = sanitize_file_path(args.system_prompt, "system prompt file")
        if os.path.isfile(potential_path):
            logging.info(f"Loading system prompt from file: {potential_path}")
            file_size = os.path.getsize(potential_path)
            if file_size > MAX_SYSTEM_PROMPT_FILE_SIZE:
                raise ValueError(f"System prompt file too large ({file_size} bytes). Max: {MAX_SYSTEM_PROMPT_FILE_SIZE} bytes")
            
            with open(potential_path, 'r', encoding='utf-8') as f:
                args.system_prompt = f.read().strip()
            if not args.system_prompt:
                 logging.warning(f"System prompt file '{potential_path}' is empty. Using default.")
                 args.system_prompt = DEFAULT_SETTINGS['system_prompt']
                 
    except ValueError as e:
        logging.critical(f"Configuration error: {e}")
        logging.critical("Please check your file paths in config.ini or command-line arguments.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Failed during path sanitation: {e}")
        sys.exit(1)
    
    # Device listing logic
    if args.list_devices or args.list_output_devices:
        if args.list_devices:
            list_audio_input_devices()
        if args.list_output_devices:
            list_audio_output_devices()
        should_exit_flag = True

    return args, config, should_exit_flag