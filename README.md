# Ollama STT-TTS Voice Assistant

A simple, hands-free Python voice assistant that runs 100% locally. This script uses Mycroft Precise for wakeword detection, OpenAI's Whisper for transcription, and Ollama for generative AI responses.

## How It Works

The assistant operates in a continuous loop with the following flow:

[Microphone] ➡️ [Mycroft Precise] ➡️ [Whisper STT] ➡️ [Ollama LLM] ➡️ [pyttsx3 TTS] ➡️ [Speaker]
            (Listens for "Hey Mycroft")  (Transcribes audio)  (Generates response) (Speaks response)


## Features

100% Local: No cloud services are required for STT, TTS, or the LLM.

Hands-Free: Uses Mycroft Precise for "Hey Mycroft" wakeword detection.

High-Quality Transcription: Leverages OpenAI's Whisper model for accurate speech-to-text.

Flexible LLM: Easily configurable to use any model supported by your local Ollama instance (e.g., llama3, mistral, phi3).

### 1. Prerequisites

Before you begin, ensure you have the following installed and running:

A. Ollama

You must have the Ollama application installed and running.

B. Pull an Ollama Model

You need at least one model downloaded for Ollama to use.

# We recommend Llama 3
ollama pull llama3

# Or, use another model
ollama pull mistral


C. System Dependencies

The PyAudio library requires portaudio.

On macOS (via Homebrew):

brew install portaudio


On Debian/Ubuntu Linux:

sudo apt-get install portaudio19-dev


### 2. Installation

Clone this repository to your local machine:

git clone [https://github.com/BranchingBad/ollama-STT-TTS.git](https://github.com/BranchingBad/ollama-STT-TTS.git)
cd ollama-STT-TTS


(Recommended) Create a Python virtual environment:

python3 -m venv venv
source venv/bin/activate


Install the required Python libraries using the requirements.txt file:

pip install -r requirements.txt


### 3. Usage

Make sure your Ollama application is running in the background.

Run the main script:

python ollama_voice_chat.py


On the first run, the script will automatically download the Whisper (base.en) and Mycroft Precise ("Hey Mycroft") models.

You will see the message: Listening for wakeword ('hey mycroft')...

Say "Hey Mycroft".

The assistant will respond "Yes?" and begin listening for your command.

Speak your prompt (e.g., "What's the capital of France?").

The script will show Ollama is thinking..., generate a response, and speak it back to you. It will then automatically return to listening for the wakeword.

You can stop the script at any time with Ctrl+C.

### 4. Configuration

To change the Ollama model, simply edit the MODEL_NAME variable at the top of the ollama_voice_chat.py script:

# Change this to your preferred Ollama model
MODEL_NAME = 'llama3'
