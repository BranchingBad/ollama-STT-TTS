#!/usr/bin/env python3

"""
assistant.py

The main entry point for the hands-free Python voice assistant.
Loads configuration, initializes, and runs the VoiceAssistant class.
"""

import logging
import sys

# Import components from other files
from config_manager import load_config_and_args, check_ollama_connectivity
from voice_assistant import VoiceAssistant

def main() -> None:
    """The entry point for the assistant application."""
    assistant: VoiceAssistant | None = None
    try:
        args, _ = load_config_and_args()
        
        # Run connectivity check *before* initializing the full class
        if not check_ollama_connectivity(args.ollama_host):
            logging.warning("Ollama server not reachable. Assistant will run but cannot respond.")
            # We still allow it to run, in case the user wants to fix Ollama
            # while the script is live. The class will handle the disconnected state.

        assistant = VoiceAssistant(args)
        assistant.run()

    except IOError as e:
        # This catches PyAudio stream errors during initialization
        logging.critical(f"FATAL ERROR during audio initialization: {e}")
        logging.critical("Check microphone connectivity or use --list-devices.")
    except (RuntimeError, OSError, ValueError) as e:
        # This catches model loading errors
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
        # This allows --list-devices and --list-voices to exit cleanly
        pass