#!/usr/bin/env python3

"""
run.py

A simple script to run the voice assistant.
This is useful for development and for users who have not installed the package.
"""

import sys
import os

# Add the src directory to the path to allow imports from voice_assistant
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from voice_assistant.assistant import main

if __name__ == "__main__":
    main()
