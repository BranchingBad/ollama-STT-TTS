#!/usr/bin/env python3

"""
config_manager.py

Handles loading configuration from config.ini and command-line arguments.
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

def setup_logging() -> None:
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)03d - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Quieten noisy libraries
    logging.getLogger("piper").setLevel(logging.WARNING)
    logging.getLogger("openwakeword").setLevel(logging.WARNING)
    logging.getLogger("webrtcvad").setLevel(logging.WARNING)

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
        logging.error("Please ensure Ollama is running and the 'ollama_host' in config.ini is correct.")
        return None

def load_config_and_args() -> Tuple[argparse.Namespace, configparser.ConfigParser]:
    """
    Loads settings from config.ini, parses command-line arguments,
    and sets up logging.
    """

    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE_NAME):
        setup_logging()
        config.read(CONFIG_FILE_NAME)
        logging.info(f"Loaded configuration from {CONFIG_FILE_NAME}")
    else:
        setup_logging()
        logging.warning(f"{CONFIG_FILE_NAME} not found. Using default settings and CLI args.")

    config_models = config['Models'] if 'Models' in config else {}
    config_func = config['Functionality'] if 'Functionality' in config else {}

    def get_config_val(section, key, default, type_converter):
        val = section.get(key)
        if val is None:
            return default
        try:
            if type_converter == bool:
                return section.getboolean(key)
            if val.lower() == 'none':
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
    func_group.add_argument('--device-index', type=lambda x: int(x) if x.lower() != 'none' else None, default=DEFAULT_SETTINGS['device_index'], help="Index of the audio input device (use --list-devices).")
    func_group.add_argument('--piper-output-device-index', type=lambda x: int(x) if x.lower() != 'none' else None, default=DEFAULT_SETTINGS['piper_output_device_index'], help="Index of the audio output device (use --list-output-devices).")
    func_group.add_argument('--max-words-per-command', type=int, default=DEFAULT_SETTINGS['max_words_per_command'], help="Maximum words allowed in a single transcribed command.")
    func_group.add_argument('--whisper-device', type=str, default=DEFAULT_SETTINGS['whisper_device'], help="Device for Whisper (e.g., 'cpu', 'cuda').")
    func_group.add_argument('--whisper-compute-type', type=str, default=DEFAULT_SETTINGS['whisper_compute_type'], help="Compute type for Whisper (e.g., 'int8', 'float16').")
    func_group.add_argument('--max-history-tokens', type=int, default=DEFAULT_SETTINGS['max_history_tokens'], help="Maximum token context for chat history.")

    parser.set_defaults(
        ollama_model=config_models.get('ollama_model', DEFAULT_SETTINGS['ollama_model']),
        whisper_model=config_models.get('whisper_model', DEFAULT_SETTINGS['whisper_model']),
        wakeword_model_path=config_models.get('wakeword_model_path', DEFAULT_SETTINGS['wakeword_model_path']),
        piper_model_path=config_models.get('piper_model_path', DEFAULT_SETTINGS['piper_model_path']),
        ollama_host=config_models.get('ollama_host', DEFAULT_SETTINGS['ollama_host']),

        wakeword=config_func.get('wakeword', DEFAULT_SETTINGS['wakeword']),
        wakeword_threshold=get_config_val(config_func, 'wakeword_threshold', DEFAULT_SETTINGS['wakeword_threshold'], float),
        vad_aggressiveness=get_config_val(config_func, 'vad_aggressiveness', DEFAULT_SETTINGS['vad_aggressiveness'], int),
        silence_seconds=get_config_val(config_func, 'silence_seconds', DEFAULT_SETTINGS['silence_seconds'], float),
        listen_timeout=get_config_val(config_func, 'listen_timeout', DEFAULT_SETTINGS['listen_timeout'], float),
        pre_buffer_ms=get_config_val(config_func, 'pre_buffer_ms', DEFAULT_SETTINGS['pre_buffer_ms'], int),
        system_prompt=config_func.get('system_prompt', DEFAULT_SETTINGS['system_prompt']),

        device_index=get_config_val(config_func, 'device_index', DEFAULT_SETTINGS['device_index'], lambda x: int(x) if x.lower() != 'none' else None),
        piper_output_device_index=get_config_val(config_func, 'piper_output_device_index', DEFAULT_SETTINGS['piper_output_device_index'], lambda x: int(x) if x.lower() != 'none' else None),

        max_words_per_command=get_config_val(config_func, 'max_words_per_command', DEFAULT_SETTINGS['max_words_per_command'], int),
        whisper_device=config_func.get('whisper_device', DEFAULT_SETTINGS['whisper_device']),
        whisper_compute_type=config_func.get('whisper_compute_type', DEFAULT_SETTINGS['whisper_compute_type']),
        max_history_tokens=get_config_val(config_func, 'max_history_tokens', DEFAULT_SETTINGS['max_history_tokens'], int)
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("DEBUG logging enabled.")

    # Check if system_prompt is a file path
    if args.system_prompt and os.path.isfile(args.system_prompt):
        logging.info(f"Loading system prompt from file: {args.system_prompt}")
        try:
            with open(args.system_prompt, 'r', encoding='utf-8') as f:
                args.system_prompt = f.read().strip()
        except Exception as e:
            logging.error(f"Failed to read system prompt file '{args.system_prompt}': {e}")
            logging.warning("Using the default system prompt instead.")
            args.system_prompt = DEFAULT_SETTINGS['system_prompt']
    elif not args.system_prompt:
         logging.warning("System prompt is empty. Using default.")
         args.system_prompt = DEFAULT_SETTINGS['system_prompt']


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

        sys.exit(0)

    return args, config