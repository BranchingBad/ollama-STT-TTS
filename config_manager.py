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

CONFIG_FILE_NAME = 'config.ini'

# FIX #16: Security constants for path validation
MAX_SYSTEM_PROMPT_FILE_SIZE = 10 * 1024  # 10KB limit for system prompt files
ALLOWED_MODEL_DIRECTORIES = ['models', 'Models', './models', './Models']

def sanitize_file_path(file_path: str, description: str = "file") -> str:
    """
    FIX #16: Sanitizes and validates file paths to prevent path traversal attacks.
    
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
    
    # Get absolute path and normalize
    abs_path = os.path.abspath(file_path)
    
    # Check for path traversal attempts
    if '..' in file_path:
        logging.warning(f"Path traversal detected in {description} path: {file_path}")
        raise ValueError(f"Invalid {description} path: path traversal not allowed")
    
    # For model files, ensure they're in allowed directories
    if 'model' in description.lower():
        path_valid = False
        current_dir = os.path.abspath('.')
        
        for allowed_dir in ALLOWED_MODEL_DIRECTORIES:
            allowed_abs = os.path.abspath(allowed_dir)
            try:
                # Check if the file is within an allowed directory
                os.path.commonpath([allowed_abs, abs_path])
                if abs_path.startswith(allowed_abs):
                    path_valid = True
                    break
            except ValueError:
                # Paths are on different drives (Windows) or not related
                continue
        
        if not path_valid:
            # Allow absolute paths that exist (for custom installations)
            if os.path.exists(abs_path):
                logging.warning(f"{description} path outside standard directories: {abs_path}")
                path_valid = True
        
        if not path_valid:
            raise ValueError(f"Invalid {description} path: must be in models/ directory or provide absolute path")
    
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
    and sets up logging.
    
    Returns: A tuple containing (parsed arguments, config object, should_exit_flag)
    """

    config = configparser.ConfigParser()
    config_loaded = False
    if os.path.exists(CONFIG_FILE_NAME):
        config.read(CONFIG_FILE_NAME)
        logging.info(f"Loaded configuration from {CONFIG_FILE_NAME}")
        config_loaded = True
    else:
        logging.warning(f"{CONFIG_FILE_NAME} not found. Using default settings and CLI args.")

    config_models = config['Models'] if 'Models' in config else {}
    config_func = config['Functionality'] if 'Functionality' in config else {}

    def get_config_val(section: configparser.SectionProxy, key: str, default: Any, type_converter: type) -> Any:
        """Helper to get and convert config values, handling 'none' string and missing config file."""
        if not config_loaded:
             return default

        val = section.get(key)
        if val is None:
            return default
        try:
            if type_converter == bool:
                return section.getboolean(key)
            # Custom handling for int device indices that may be set to 'None'
            if type_converter == int and val.lower() == 'none':
                return None
            
            return type_converter(val)
        except (ValueError, configparser.NoOptionError):
            logging.warning(f"Invalid value '{val}' for '{key}' in config.ini. Using default: {default}")
            return default

    parser = argparse.ArgumentParser(description="A hands-free voice assistant for Ollama.")

    parser.add_argument('--list-devices', action='store_true', help="List available audio input devices and exit.")
    parser.add_argument('--list-output-devices', action='store_true', help="List available audio output devices and exit.")
    parser.add_argument('--debug', action='store_true', help="Enable debug logging.")

    model_group = parser.add_argument_group('Models')
    model_group.add_argument('--ollama-model', type=str, default=DEFAULT_SETTINGS['ollama_model'], help="Name of the Ollama model to use.")
    model_group.add_argument('--whisper-model', type=str, default=DEFAULT_SETTINGS['whisper_model'], help="Name of the faster-whisper model to use (e.g., tiny.en, base.en).")
    model_group.add_argument('--wakeword-model-path', type=str, default=DEFAULT_SETTINGS['wakeword_model_path'], help="Path to the .onnx wakeword model file.")
    model_group.add_argument('--piper-model-path', type=str, default=DEFAULT_SETTINGS['piper_model_path'], help="Path to the .onnx Piper TTS model file.")
    model_group.add_argument('--ollama-host', type=str, default=DEFAULT_SETTINGS['ollama_host'], help="URL of the Ollama server.")

    func_group = parser.add_argument_group('Functionality')
    func_group.add_argument('--wakeword', type=str, default=DEFAULT_SETTINGS['wakeword'], help="The wakeword phrase.")
    func_group.add_argument('--wakeword-threshold', type=float, default=DEFAULT_SETTINGS['wakeword_threshold'], help="Wakeword detection threshold (0.0 to 1.0).")
    func_group.add_argument('--vad-aggressiveness', type=int, default=DEFAULT_SETTINGS['vad_aggressiveness'], help="VAD aggressiveness (0=least aggressive, 3=most aggressive).")
    func_group.add_argument('--silence-seconds', type=float, default=DEFAULT_SETTINGS['silence_seconds'], help="Seconds of silence to detect end of speech.")
    func_group.add_argument('--listen-timeout', type=float, default=DEFAULT_SETTINGS['listen_timeout'], help="Seconds to wait for speech before timing out.")
    func_group.add_argument('--pre-buffer-ms', type=int, default=DEFAULT_SETTINGS['pre_buffer_ms'], help="Milliseconds of audio to keep before speech starts.")
    func_group.add_argument('--system-prompt', type=str, default=DEFAULT_SETTINGS['system_prompt'], help="The system prompt for the assistant (or a path to a .txt file).")
    
    func_group.add_argument('--device-index', type=device_index_type, default=DEFAULT_SETTINGS['device_index'], help="Index of the audio input device (integer or 'none').")
    func_group.add_argument('--piper-output-device-index', type=device_index_type, default=DEFAULT_SETTINGS['piper_output_device_index'], help="Index of the audio output device (integer or 'none').")

    func_group.add_argument('--max-words-per-command', type=int, default=DEFAULT_SETTINGS['max_words_per_command'], help="Maximum words allowed in a single transcribed command.")
    func_group.add_argument('--whisper-device', type=str, default=DEFAULT_SETTINGS['whisper_device'], help="Device for Whisper (e.g., 'cpu', 'cuda').")
    func_group.add_argument('--whisper-compute-type', type=str, default=DEFAULT_SETTINGS['whisper_compute_type'], help="Compute type for Whisper (e.g., 'int8', 'float16').")
    func_group.add_argument('--max-history-tokens', type=int, default=DEFAULT_SETTINGS['max_history_tokens'], help="Maximum token context for chat history.")
    func_group.add_argument('--audio-buffer-size', type=int, default=DEFAULT_SETTINGS['audio_buffer_size'], help="Size of the audio buffer queue (default: 200).")

    # Apply configuration defaults
    parser.set_defaults(
        # Models Section (Using str as the type_converter for consistency)
        ollama_model=get_config_val(config_models, 'ollama_model', DEFAULT_SETTINGS['ollama_model'], str),
        whisper_model=get_config_val(config_models, 'whisper_model', DEFAULT_SETTINGS['whisper_model'], str),
        wakeword_model_path=get_config_val(config_models, 'wakeword_model_path', DEFAULT_SETTINGS['wakeword_model_path'], str),
        piper_model_path=get_config_val(config_models, 'piper_model_path', DEFAULT_SETTINGS['piper_model_path'], str),
        ollama_host=get_config_val(config_models, 'ollama_host', DEFAULT_SETTINGS['ollama_host'], str),

        # Functionality Section
        wakeword=get_config_val(config_func, 'wakeword', DEFAULT_SETTINGS['wakeword'], str),
        wakeword_threshold=get_config_val(config_func, 'wakeword_threshold', DEFAULT_SETTINGS['wakeword_threshold'], float),
        vad_aggressiveness=get_config_val(config_func, 'vad_aggressiveness', DEFAULT_SETTINGS['vad_aggressiveness'], int),
        silence_seconds=get_config_val(config_func, 'silence_seconds', DEFAULT_SETTINGS['silence_seconds'], float),
        listen_timeout=get_config_val(config_func, 'listen_timeout', DEFAULT_SETTINGS['listen_timeout'], float),
        pre_buffer_ms=get_config_val(config_func, 'pre_buffer_ms', DEFAULT_SETTINGS['pre_buffer_ms'], int),
        system_prompt=get_config_val(config_func, 'system_prompt', DEFAULT_SETTINGS['system_prompt'], str),

        # Use int for config defaults, allowing get_config_val to return None if 'none' is found
        device_index=get_config_val(config_func, 'device_index', DEFAULT_SETTINGS['device_index'], int),
        piper_output_device_index=get_config_val(config_func, 'piper_output_device_index', DEFAULT_SETTINGS['piper_output_device_index'], int),

        max_words_per_command=get_config_val(config_func, 'max_words_per_command', DEFAULT_SETTINGS['max_words_per_command'], int),
        whisper_device=get_config_val(config_func, 'whisper_device', DEFAULT_SETTINGS['whisper_device'], str),
        whisper_compute_type=get_config_val(config_func, 'whisper_compute_type', DEFAULT_SETTINGS['whisper_compute_type'], str),
        max_history_tokens=get_config_val(config_func, 'max_history_tokens', DEFAULT_SETTINGS['max_history_tokens'], int),
        audio_buffer_size=get_config_val(config_func, 'audio_buffer_size', DEFAULT_SETTINGS['audio_buffer_size'], int)
    )

    args = parser.parse_args()
    should_exit_flag = False

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("DEBUG logging enabled.")

    # FIX #16: Sanitize model file paths
    try:
        args.wakeword_model_path = sanitize_file_path(args.wakeword_model_path, "wakeword model")
        args.piper_model_path = sanitize_file_path(args.piper_model_path, "Piper TTS model")
    except ValueError as e:
        logging.critical(f"Security error: {e}")
        logging.critical("Please check your model paths in config.ini or command-line arguments.")
        sys.exit(1)

    # FIX #16: Check if system_prompt is a file path with validation
    if args.system_prompt and os.path.isfile(args.system_prompt):
        logging.info(f"Loading system prompt from file: {args.system_prompt}")
        
        try:
            # Sanitize the path
            prompt_file_path = sanitize_file_path(args.system_prompt, "system prompt file")
            
            # FIX #16: Check file size before reading
            file_size = os.path.getsize(prompt_file_path)
            if file_size > MAX_SYSTEM_PROMPT_FILE_SIZE:
                raise ValueError(f"System prompt file too large ({file_size} bytes). Maximum allowed: {MAX_SYSTEM_PROMPT_FILE_SIZE} bytes")
            
            with open(prompt_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
            
            if file_content:
                args.system_prompt = file_content
                logging.info(f"Loaded system prompt ({len(file_content)} characters) from file.")
            else:
                logging.warning(f"System prompt file '{args.system_prompt}' is empty. Using default.")
                args.system_prompt = DEFAULT_SETTINGS['system_prompt']
                
        except ValueError as e:
            logging.error(f"Security error with system prompt file: {e}")
            logging.warning("Using the default system prompt instead.")
            args.system_prompt = DEFAULT_SETTINGS['system_prompt']
        except Exception as e:
            logging.error(f"Failed to read system prompt file '{args.system_prompt}': {e}")
            logging.warning("Using the default system prompt instead.")
            args.system_prompt = DEFAULT_SETTINGS['system_prompt']
    
    # Device listing logic
    if args.list_devices or args.list_output_devices:
        if args.list_devices:
            try:
                list_audio_input_devices()
            except Exception as e:
                logging.error(f"Could not list audio input devices: {e}")
        
        if args.list_output_devices:
            try:
                list_audio_output_devices()
            except Exception as e:
                logging.error(f"Could not list audio output devices: {e}")

        should_exit_flag = True # Signal the main loop to exit

    return args, config, should_exit_flag