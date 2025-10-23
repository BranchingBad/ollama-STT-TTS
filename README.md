# 🤖 Ollama Voice Assistant (STT, LLM, TTS)

A simple, hands-free Python voice assistant that runs 100% locally. This script uses openwakeword for wakeword detection, webrtcvad for silence detection, OpenAI's Whisper for transcription, and Ollama for generative AI responses.

## 🧩 How It Works

The assistant operates in a continuous loop with the following flow:

```mermaid
flowchart LR
    A[Microphone] --> B(openwakeword);
    B -- "hey mycroft" --> C(webrtcvad);
    C -- "Records until silence" --> D[Whisper STT];
    D -- "Transcribes audio" --> E[Ollama LLM];
    E -- "Generates response" --> F[pyttsx3 TTS];
    F -- "Speaks response" --> G[Speaker];
```

## 💡 Features
- **100% Local**: No cloud services are required for STT, TTS, or the LLM.
- **Hands-Free**: Uses openwakeword for wakeword detection.
- **Smart Recording**: Uses webrtcvad (Voice Activity Detection) to automatically stop recording when you finish speaking.
- **High-Quality Transcription**: Leverages OpenAI's Whisper model for accurate speech-to-text.
- **Flexible LLM**: Easily configurable to use any model supported by your local Ollama instance (e.g., llama3, mistral, phi3).

## 🔩 1. Prerequisites

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

On Fedora/RHEL Linux:

_Note: The portaudio-devel package is in the standard Fedora repositories, but ffmpeg is not. You must first enable the RPM Fusion repository to install ffmpeg._

```bash
sudo dnf install portaudio-devel gcc python3-devel ffmpeg
```

## 🔧 2. Installation

Clone this repository to your local machine:
```bash
git clone https://github.com/BranchingBad/ollama-STT-TTS.git
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

## ⌨️ 3. Usage

Make sure your Ollama application is running in the background.

Run the main script:
```bash
python ollama_voice_chat.py
```

On the first run, the script will automatically download the Whisper (``base.en``) and openwakeword models.

You will see the message: ``Ready! Listening for 'hey glados'...``

Say the wakeword (e.g., "Hey glados").

The assistant will respond "Yes?" and begin listening for your command.

Speak your prompt (e.g., "What's the capital of France?"). The script will listen until you stop talking based on the silence detection settings.

The script will transcribe your audio, send it to Ollama, and speak the response back to you. It will then automatically return to listening for the wakeword.

**Note:** When running Whisper on a CPU, you may see a warning like UserWarning: FP16 is not supported on CPU; using FP32 instead. This is expected behavior and does not indicate an error.

You can stop the script at any time with Ctrl+C. Special voice commands like "goodbye" or "exit" will also stop the script, and "new chat" or "reset chat" will clear the conversation history.

## 🎛️ 4. Configuration

You can customize the assistant's behavior using command-line arguments or by editing the ``config.ini`` file. Command-line arguments override settings in ``config.ini``.

Run with defaults (set in ``ollama_voice_chat.py``):
```Bash
python ollama_voice_chat.py
```
Example: Run with different models and settings:
```Bash
python ollama_voice_chat.py --wakeword-model "hey_glados" --wakeword "hey glados" --vad-aggressiveness 1
```

All Arguments:

``--ollama-model``: The Ollama model to use (e.g., "llama3", "mistral", "phi3").

Default: ``llama3``

``--whisper-model``: The Whisper model to use (e.g., "tiny.en", "base.en", "small.en"). Models are downloaded automatically.

Default: ``base.en``

``--wakeword-model``: The base name of the openwakeword model file (e.g., "hey_glados", "hey_mycroft"). Assumes the .onnx file is in the same directory or will be downloaded.

Default: ``hey_glados``

``--wakeword``: The specific wakeword phrase to listen for (must match the loaded model's phrase).

Default: ``hey glados``

``--wakeword-threshold``: Wakeword detection sensitivity (0.0 to 1.0). Higher values are less sensitive.

Default: ``0.6``

``--vad-aggressiveness``: Voice Activity Detection aggressiveness (0=least aggressive, 3=most aggressive). Higher values detect silence more readily.

Default: ``2``

``--silence-seconds``: Seconds of silence to wait before stopping recording after speech is detected.

Default: ``0.7``

``--listen-timeout``: Seconds to wait for speech to start after the wakeword before timing out.

Default: ``6.0``

``--pre-buffer-ms``: Milliseconds of audio to capture before speech is detected, helps prevent cutting off the start of words.

Default: ``400``

``--system-prompt``: The system prompt for the Ollama model.

Default: ``You are a helpful, concise voice assistant.``

Configuration File (`config.ini`)

You can also set default values by editing the `config.ini` file in the same directory as the script. This file allows you to configure most of the same options as the command-line arguments, plus the system prompt.

ini
```bash
[Models]
ollama_model = llama3       # Default Ollama model
whisper_model = base.en     # Default Whisper model
wakeword_model = hey_glados # Default wakeword model base name

[Functionality]
wakeword = hey glados       # Wakeword phrase
wakeword_threshold = 0.6    # Wakeword sensitivity
vad_aggressiveness = 2      # VAD aggressiveness
silence_seconds = 0.7       # Silence duration to end recording
listen_timeout = 6.0        # Timeout waiting for command speech
pre_buffer_ms = 400         # Audio pre-buffering duration
system_prompt = You are a helpful, concise voice assistant. # The initial prompt for Ollama
```

**Note**: The ``ollama_voice_chat.py`` script reads default settings from ``config.ini``. Any command-line arguments provided when running the script will override the values set in this file.
