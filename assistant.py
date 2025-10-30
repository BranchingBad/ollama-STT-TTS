#!/usr/bin/env python3

"""
assistant.py

The main entry point for the hands-free Python voice assistant.
Loads configuration, initializes, and runs the VoiceAssistant class.
"""

import logging
import sys

# Import components from other files
from config_manager import load_config_and_args, get_ollama_client
from voice_assistant import VoiceAssistant

def main() -> None:
    """The entry point for the assistant application."""
    assistant: VoiceAssistant | None = None
    try:
        # load_config_and_args now returns an additional flag
        args, _, should_exit = load_config_and_args()
        
        # --- NEW LOGIC: Handle device listing exit flag ---
        if should_exit:
            # config_manager has already printed the device list.
            # Exit cleanly without running the rest of the assistant initialization.
            sys.exit(0)
        # --- END NEW LOGIC ---

        # Get the Ollama client *once* and pass it to the assistant.
        # This function returns None if connection fails.
        ollama_client = get_ollama_client(args.ollama_host)
        
        # --- IMPROVEMENT: Simplify the warning check ---
        # The ollama_client is only None here if the connection was attempted and failed.
        if ollama_client is None:
            logging.warning("Ollama server not reachable. Assistant will run but cannot respond.")
        # --- END IMPROVEMENT ---

        assistant = VoiceAssistant(args, ollama_client)
        
        assistant.run()

    except IOError as e:
        # This catches PyAudio/sounddevice stream errors during initialization
        logging.critical(f"FATAL ERROR during audio initialization: {e}")
        logging.critical("Check microphone connectivity or use --list-devices.")
    except (RuntimeError, OSError, ValueError) as e:
        # This catches model loading errors (Whisper, Piper, OpenWakeWord)
        logging.critical(f"FATAL ERROR during model loading: {e}")
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # Ensure cleanup runs even if initialization failed
        if assistant:
            assistant.cleanup()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # This allows --list-devices and --list-output-devices to exit cleanly
        pass