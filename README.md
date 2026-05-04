# 🤖 Ollama Voice Assistant (STT, LLM, TTS)

A simple, hands-free Python voice assistant that runs 100% locally. This script uses openwakeword for wakeword detection, webrtcvad for silence detection, OpenAI's Whisper for transcription, and Ollama for generative AI responses.

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
- **Configurable**: Settings are adjustable via `config.ini` and command-line arguments.

## 🔩 1. Prerequisites
Before you begin, ensure you have the following installed and running:

### 🦙 A. Ollama
You must have the Ollama application installed and running.

### 📦 B. Pull an Ollama Model
You need at least one model downloaded for Ollama to use.
```bash
# The default model is llama3
ollama pull llama3
```

### ⚙️ C. System Dependencies
The underlying audio libraries require system packages to be installed.

**On Debian/Ubuntu Linux:**
```bash
sudo apt-get update && sudo apt-get install -y portaudio19-dev ffmpeg
```
**On Fedora/RHEL Linux:**
```bash
# Enable RPM Fusion if you haven't already (see https://rpmfusion.org/Configuration)
sudo dnf install -y portaudio-devel gcc python3-devel ffmpeg pulseaudio-libs-devel
```

### 🗣️ D. Wake-Word and TTS Models
This project comes with pre-packaged wake-word (`hey_jarvis`) and TTS models in the `models/` directory. No download is required unless you wish to use different ones.

## 🔧 2. Installation
Clone this repository to your local machine and navigate into the project directory.
```bash
git clone https://github.com/BranchingBad/ollama-STT-TTS.git
cd ollama-STT-TTS
```

Create and activate a Python virtual environment (recommended).
```bash
# Create the environment
python3 -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

Install the project. For regular use:
```bash
pip install .
```

For development (including testing dependencies):
```bash
pip install -e .[test]
```
On the first run, the application will automatically download the required `faster-whisper` model.

## ⌨️ 3. Running the Assistant
You can run the assistant either locally with Python or via Docker. **All commands should be run from the root of the project directory.**

### 🐍 A. Run Locally with Python
Make sure your Ollama application is running. Then, start the assistant:
```bash
python run.py
```
Or, if you have installed the package, you can use the entry point:
```bash
ollama-voice-assistant
```

When ready, you will see the message: `Ready! Listening for 'hey jarvis'...
`

**How to Interact:**
1.  **Say the wakeword** (e.g., "Hey jarvis").
2.  The assistant will respond, "Yes?" and begin listening.
3.  **Speak your command** (e.g., "Who won the war of 1812?").
4.  The assistant will transcribe your audio, send it to Ollama, and speak the response. It will then return to listening for the wakeword.

**Special Commands:**
- `"goodbye"` or `"exit"`: Stops the script.
- `"new chat"` or `"reset chat"`: Clears the conversation history for the LLM.

### 🐋 B. Run with Docker
A pre-built Docker image is available on the GitHub Container Registry.

**1. Pull the Image:**
```bash
docker pull ghcr.io/branchingbad/ollama-stt-tts:latest
```

**2. Prepare Configuration:**
You will likely need to find the correct audio device index for the container to use. You can list the devices from your local (non-Docker) installation:
```bash
python run.py --list-devices
```
Copy the `config.ini` file from the repository to a local directory and edit the `device_index` with the correct value from the command above.

**3. Run the Container (Linux):**
This command connects the container to your host's network (to access Ollama), mounts your sound devices, and mounts your local `config.ini`.
```bash
docker run --rm -it \
  --network=host \
  --device /dev/snd \
  -v ./config.ini:/app/config.ini:ro \
  ghcr.io/branchingbad/ollama-stt-tts:latest
```
- `--network=host`: Required for the container to access Ollama at `http://localhost:11434`.
- `--device /dev/snd`: Grants the container access to your host's sound devices (Linux-specific).
- `-v ./config.ini...`: Mounts your local configuration file as read-only.

**Note for macOS/Windows users:** Audio device mapping is more complex. You may need to adjust the `docker run` command. If `--network=host` is unavailable, remove it and set `ollama_host` in your `config.ini` to `http://host.docker.internal:11434`.

## 🎛️ 4. Configuration
Customize the assistant by editing `config.ini` or by providing command-line arguments. Arguments always override settings from the config file.

**Example Commands:**
```bash
# Run with a different wakeword threshold and VAD aggressiveness
python run.py --wakeword-threshold 0.5 --vad-aggressiveness 1

# Run using a different Ollama model and input device
python run.py --ollama-model mistral --device-index 2
```

**Common Arguments:**
- `--list-devices`: List available audio input devices and exit.
- `--list-output-devices`: List available audio output devices and exit.
- `--debug`: Enable verbose debug logging.
- `--ollama-model`: Name of the Ollama model to use (e.g., `llama3`, `mistral`).
- `--whisper-model`: Name of the `faster-whisper` model to use (e.g., `tiny.en`, `base.en`).
- `--wakeword`: The wakeword phrase to listen for.
- `--device-index`: The integer index of your microphone.
- `--piper-output-device-index`: The integer index of your speaker.
- `--system-prompt`: A custom system prompt or a path to a `.txt` file containing one.

For a full list of configurable options, see the `[Models]` and `[Functionality]` sections in the `config.ini` file.

## 🧪 5. Testing
This project includes a suite of unit tests to ensure the reliability of its core components. The tests cover:
-   Ollama connection
-   Configuration management
-   Audio utilities
-   LLM handling
-   Audio transcription
-   Speech synthesis
-   Wake-word debouncing logic

### Running the Tests
To run the tests, first ensure you have installed the development dependencies:
```bash
pip install -e .[test]
```
Then, run `pytest` and `ruff` from the root of the project directory:
```bash
python3 -m pytest
ruff check src tests
```

A sample debug run is captured in [`docs/sample-conversation.log`](docs/sample-conversation.log).

## 🚑 6. Troubleshooting

### "OSError: PortAudio library not found" / "cannot find -lportaudio"
You're missing the system-level audio library. Install it as described in the *Prerequisites* section (`portaudio19-dev` on Debian/Ubuntu, `portaudio-devel` on Fedora, `brew install portaudio` on macOS).

### "Failed to connect to Ollama at http://localhost:11434"
- Make sure `ollama serve` is running in another terminal (or the Ollama desktop app is open).
- Verify the model is pulled: `ollama list` should include the model named in `config.ini`.
- If you run inside Docker without `--network=host`, set `ollama_host = http://host.docker.internal:11434` in `config.ini`.

### Microphone permission denied (macOS / Windows)
- **macOS:** Go to *System Settings → Privacy & Security → Microphone* and enable access for your terminal / IDE.
- **Windows:** *Settings → Privacy → Microphone* → allow desktop apps.
- Inside WSL the host microphone is not exposed by default; run from native Windows Python instead.

### "Wakeword model missing" on startup
Either the path in `config.ini` is wrong or the file under `models/` was not pulled. Check that `wakeword_model_path` resolves to one of the `.onnx` files actually present in `models/`. Relative paths are resolved against the project root.

### Wake word never triggers / triggers all the time
- Run `python run.py --debug` and watch the live `score`/`avg` values in the log.
- Lower `wakeword_threshold` if your real wake-word scores are below it; raise it if you see false positives.
- A headset with a close-mic gives noticeably better results than a laptop's built-in mic.

### "Audio may be clipping" warnings
Reduce `gain` in `config.ini` (e.g. from `1.0` to `0.7`).

### Whisper transcription is empty or wrong
- Try a larger `whisper_model` (e.g. `base.en` → `small.en`).
- If Whisper's `no_speech_prob` is consistently near 1.0 in `--debug` logs, your VAD is keeping non-speech audio — increase `vad_aggressiveness` (0–3) or trim `pre_buffer_ms`.

### `pip install .` fails on `webrtcvad`
You need a C compiler. On Debian/Ubuntu: `sudo apt-get install build-essential`. On Windows, install the *Visual C++ Build Tools*.

### Check version
```bash
python run.py --version
```
