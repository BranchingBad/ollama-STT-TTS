# 🤖 Ollama Voice Assistant (STT, LLM, TTS)

A simple, hands-free Python voice assistant that runs 100% locally. This script uses openwakeword for wakeword detection, webrtcvad for silence detection, OpenAI's Whisper for transcription, and Ollama for generative AI responses.

## 🧩 How It Works

The assistant operates in a continuous loop with the following flow:

```mermaid
flowchart LR
    A[Microphone] --> B(openwakeword);
    B -- "hey jarvis" --> C(webrtcvad);
    C -- "Records until silence" --> D[faster-whisper STT];
    D -- "Transcribes audio" --> E[Ollama LLM];
    E -- "Generates streaming response" --> F[Piper TTS];
    F -- "Speaks response" --> G[Speaker];
```

## 💡 Features
- **100% Local**: No cloud services are required for STT, TTS, or the LLM.
- **Hands-Free**: Uses openwakeword for wakeword detection.
- **Low-Latency TTS**: Uses the Piper TTS engine for fast, high-quality voice output.
- **Optimized STT**: Leverages faster-whisper models for efficient and accurate speech-to-text.
- **Smart Recording**: Uses webrtcvad (Voice Activity Detection) to automatically stop recording when you finish speaking.
- **Flexible LLM**: Easily configurable to use any model supported by your local Ollama instance (e.g., llama3, mistral, phi3).
- **Cross-Platform Audio**: Uses sounddevice for audio input/output.
- **Configurable**: Settings adjustable via ``config.ini`` and command-line arguments.

## 🔩 1. Prerequisites
Before you begin, ensure you have the following installed and running:

### 🦙 A. Ollama
You must have the Ollama application installed and running.

### 📦 B. Pull an Ollama Model
You need at least one model downloaded for Ollama to use.

```Bash
# The default model is llama3
ollama pull llama3
```
### ⚙️ C. System Dependencies
The sounddevice library requires portaudio. faster-whisper may require ffmpeg for audio handling.

On Fedora/RHEL Linux:
```Bash
# Enable RPM Fusion (if not already done) - see https://rpmfusion.org/Configuration
sudo dnf install portaudio-devel gcc python3-devel ffmpeg  pulseaudio-libs-devel
```

On Debian/Ubuntu Linux:
```Bash
sudo apt-get update && sudo apt-get install portaudio19-dev ffmpeg
```

### 🗣️ D. Piper TTS Model
The assistant requires the Piper ONNX model and its corresponding JSON config file. By default, these files must be in the ``models/`` directory, matching the paths specified in ``config.ini``.

## 🔧 2. Installation
Clone this repository to your local machine:

```Bash
git clone https://github.com/BranchingBad/ollama-STT-TTS.git
cd ollama-STT-TTS
```

(Recommended) Create a Python virtual environment:
```Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required Python libraries using the requirements.txt file:
```Bash
pip install -r requirements.txt
```

## ⌨️ 3. Usage
You can run the assistant either via Docker or directly with Python.

### 🐋 A. Run with Docker (Recommended)
A Docker image is automatically built and published to the GitHub Container Registry (ghcr) via a GitHub Action.

Pull the latest image:

```Bash
docker pull ghcr.io/branchingbad/ollama-stt-tts:latest
```

Prepare your ``config.ini``: Copy the ``config.ini`` file from this repository to your host machine. You will likely need to edit it to set your device_index. You can find the correct index by running python ``assistant.py --list-devices`` from a local (non-Docker) installation, or by checking your system's audio settings.

Run the container (Linux): The following command runs the container, connects it to your host's network (to access Ollama), mounts your host's audio devices, and mounts your local ``config.ini``.

```Bash
docker run --rm -it \
  --network=host \
  --device /dev/snd \
  -v ./config.ini:/app/config.ini:ro \
  ghcr.io/branchingbad/ollama-stt-tts:latest
```
--network=host: Required for the container to access your Ollama server at http://localhost:11434.

--device /dev/snd: Grants the container access to your host's sound devices (this is Linux-specific).

-v ./config.ini:/app/config.ini:ro: Mounts your local configuration file as read-only into the container's /app directory.

Run the container (macOS/Windows): Audio device mapping on macOS and Windows is more complex. You may need to adjust the docker run command to correctly share your microphone and audio output. If --network=host is not available, remove it and set ollama_host in your ``config.ini`` to http://host.docker.internal:11434.

### 🐍 B. Run Locally with Python
Run the main script: Make sure your Ollama application is running in the background.
```Bash
python assistant.py
```

On the first run, the script will automatically download the faster-whisper (base.en by default) and openwakeword (hey_jarvis_v2.onnx by default) models.

List devices (Optional): To find the index for your microphone or speaker, use these commands:
```Bash
# Lists available audio input devices (microphones)
python assistant.py --list-devices
# Lists available audio output devices (speakers for TTS)
python assistant.py --list-output-devices
```

Use the index provided by these commands with the ``device_index`` and ``piper_output_device_index`` arguments in ``config.ini``.

Interact: When ready, you will see the message: Ready! Listening for 'hey jarvis'....

Say the wakeword (e.g., "``Hey jarvis``").

The assistant will respond "Yes?" and begin listening.

Speak your prompt (e.g., "Who won the war of 1812?").

The script will transcribe your audio, send it to Ollama, and speak the response. It will then return to listening for the wakeword.

_Note: Special voice commands like "``goodbye``" or "``exit``" will stop the script, and "``new chat``" or "``reset chat``" will clear the conversation history._

## 🎛️ 4. Configuration
You can customize the assistant's behavior using command-line arguments or by editing the ``config.ini`` file. Command-line arguments override settings in ``config.ini``.

Run with defaults (loaded from ``config.ini``):
```Bash
python assistant.py
```

Example: Run with different settings:
```Bash
python assistant.py --wakeword-threshold 0.5 --vad-aggressiveness 1 --ollama-model llama3 --device-index 2
```

### All Arguments (Defaults from ``config.ini`` and ``audio_utils.py`` shown):

``--list-devices``: List available audio input devices and exit.

``--list-voices``: List available TTS voices and exit.

``--debug``: Enable debug logging.

### Models Group:

``--ollama-model``: Name of the Ollama model to use. (Default: ``phi3:mini``)

``--whisper-model``: Name of the faster-whisper model (e.g., tiny.en, base.en). (Default: ``tiny.en``)

``--wakeword-model-path``: Path to the .onnx wakeword model file. (Default: ``hey_jarvis.onnx``)

``--ollama-host``: URL of the Ollama server. (Default: ``http://localhost:11434``)

### Functionality Group:

``--wakeword``: The wakeword phrase. (Default: ``hey jarvis``)

``--wakeword-threshold``: Wakeword detection threshold (0.0 to 1.0). (Default: ``0.45``)

``--vad-aggressiveness``: VAD aggressiveness (0=least, 3=most aggressive). (Default: ``2``)

``--silence-seconds``: Seconds of silence to detect end of speech. (Default: ``0.5``)

``--listen-timeout``: Seconds to wait for speech before timing out. (Default: ``4.0``)

``--pre-buffer-ms``: Milliseconds of audio to keep before speech starts. (Default: ``400``)

``--system-prompt``: The system prompt for the assistant, or a path to a .txt file containing the prompt. (Default: You are a friendly, concise, and intelligent voice assistant named jarvis. Keep your responses short and witty.)

``--device-index``: Index of the audio input device (use --list-devices). Set to None for default. (Default: ``13 in config.ini, but None in code defaults``)

``--max-words-per-command``: Maximum words allowed in a single transcribed command. (Default: ``60``)

``--whisper-device``: Device for Whisper ('cpu', 'cuda'). (Default: ``cpu``)

``--whisper-compute-type``: Compute type for Whisper (e.g., 'int8', 'float16'). (Default: ``int8``)

``--max-history-tokens``: Maximum token context for chat history (used for pruning). (Default: ``2048``)