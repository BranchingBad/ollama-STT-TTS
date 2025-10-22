# Ollama Voice Chat Setup (with Wakeword)

This script lets you talk to an Ollama model using your voice and hear its responses. It runs in a hands-free mode, waiting for a "wakeword" before it starts listening for your command.

## How It Works

Wait: It uses pvporcupine to listen for a wakeword (e.g., "Picovoice", "Hey Computer").

Listen: When the wakeword is detected, it uses the SpeechRecognition library (with OpenAI's Whisper model locally) to capture audio from your microphone and convert it to text.

Think: It sends this text to your running Ollama instance (using the ollama library).

Speak: It takes the text response from Ollama and uses the pyttsx3 library to convert it back into speech.

## 1. Prerequisites (Must Do First!)

A. Install Ollama

You must have the Ollama application installed and running on your computer.

B. Pull an Ollama Model

You need at least one model downloaded. If you don't have one, run this in your terminal:

ollama pull llama3


(You can replace llama3 with another model like mistral if you prefer).

C. Get a Porcupine Access Key

The wakeword engine requires a free Access Key from Picovoice.

Go to the Picovoice Console.

Sign up for a free account.

On the main page, you will see your AccessKey. Copy this long string.

You will need to paste this key into the ollama_voice_chat.py script.

D. Install Python Libraries

Open your terminal and install the following Python libraries:

# For the Ollama API
pip install ollama

# For text-to-speech
pip install pyttsx3

# For speech-to-text
pip install SpeechRecognition

# For local, high-quality transcription
pip install openai-whisper

# For microphone access
pip install PyAudio

# For wakeword detection
pip install pvporcupine


Note on Dependencies:

On macOS or Linux, you may need to install portaudio first:

macOS: brew install portaudio

Linux: sudo apt-get install portaudio19-dev

## 2. Run the Script

Save the ollama_voice_chat.py file.

IMPORTANT: Open the .py file and paste your AccessKey from Step 1C into the PORCUPINE_ACCESS_KEY variable.

Make sure your Ollama application is running.

Run the script from your terminal:

python ollama_voice_chat.py


The first time you run it, Whisper will download its 'base' model (a one-time download).

You will see "Listening for wakeword ('picovoice')...".

Say "Picovoice".

The script will say "Yes?" and you will see "Listening...". Now, say your command.

When you see "Thinking...", Ollama is generating a response.

You'll hear the response, and the script will go back to listening for the wakeword.

You can stop the script with Ctrl+C in your terminal.
