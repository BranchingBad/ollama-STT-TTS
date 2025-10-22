# Ollama STT-TTS Voice Assistant

A simple, hands-free Python voice assistant that runs 100% locally. This script uses openwakeword for wakeword detection, webrtcvad for silence detection, OpenAI's Whisper for transcription, and Ollama for generative AI responses.

## How It Works

The assistant operates in a continuous loop with the following flow:

[Microphone] ➡️ [openwakeword] ➡️ [webrtcvad] ➡️ [Whisper STT] ➡️ [Ollama LLM] ➡️ [pyttsx3 TTS] ➡️ [Speaker]

(Listens for "hey mycroft") (Records until silence) (Transcribes audio) (Generates response) (Speaks response)

## Features
- **100% Local**: No cloud services are required for STT, TTS, or the LLM.
- **Hands-Free**: Uses openwakeword for wakeword detection.
- **Smart Recording**: Uses webrtcvad (Voice Activity Detection) to automatically stop recording when you finish speaking.
- **High-Quality Transcription**: Leverages OpenAI's Whisper model for accurate speech-to-text.
- **Flexible LLM**: Easily configurable to use any model supported by your local Ollama instance (e.g., llama3, mistral, phi3).

## 1. Prerequisites

Before you begin, ensure you have the following installed and running:

### A. Ollama

You must have the Ollama application installed and running.

### B. Pull an Ollama Model

You need at least one model downloaded for Ollama to use.
```bash 
#We recommend Llama 3
ollama pull llama3

#Or, use another model
ollama pull mistral
```

### C. System Dependencies

The ``PyAudio`` library requires ``portaudio``.

On macOS (via Homebrew):
```bash
brew install portaudio ffmpeg
```

On Debian/Ubuntu Linux:
```bash
sudo apt-get install portaudio19-dev ffmpeg
```

## 2. Installation

Clone this repository to your local machine:
```bash
git clone [https://github.com/BranchingBad/ollama-STT-TTS.git]
cd ollama-STT-TTS
```

(Recommended) Create a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python libraries using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## 3. Usage

Make sure your Ollama application is running in the background.

Run the main script:
```bash
python ollama_voice_chat.py
```

On the first run, the script will automatically download the Whisper (``base.en``) and openwakeword models.

You will see the message: ``Ready! Listening for 'hey mycroft'...``

Say "Hey mycroft".

The assistant will respond "Yes?" and begin listening for your command.

Speak your prompt (e.g., "What's the capital of France?"). The script will listen until you stop talking for about 2 seconds.

The script will transcribe your audio, send it to Ollama, and speak the response back to you. It will then automatically return to listening for the wakeword.

You can stop the script at any time with Ctrl+C.

## 4. Configuration

You can customize the assistant's behavior using command-line arguments.

Run with defaults (llama3, base.en, hey mycroft):
```Bash
python ollama_voice_chat.py
```
Example: Run with different models and settings:
```Bash
python ollama_voice_chat.py --ollama-model mistral --whisper-model small.en --wakeword "hey computer"
```

All Arguments
``--ollama-model``: The Ollama model to use (e.g., "llama3", "mistral", "phi3").
Default: ``llama3``

``--whisper-model``: The Whisper model to use (e.g., "tiny.en", "base.en", "small.en").
Default: ``base.en``

``--wakeword-model``: The openwakeword model file to use.
Default: ``hey_mycroft_v0.1``

``--wakeword``: The specific wakeword phrase to listen for.
Default: ``hey mycroft``

``--wakeword-threshold``: Wakeword detection sensitivity (0.0 to 1.0). Higher is less sensitive.
Default: ``0.5``

``--vad-aggressiveness``: Voice Activity Detection aggressiveness (0=least, 3=most aggressive).
Default: ``2``

``--silence-seconds``: Seconds of silence to wait before stopping recording.
Default: ``2.0``
