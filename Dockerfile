# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for
# pyaudio (portaudio19-dev) and pyttsx3 (espeak) on Linux.
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the modularized application files into the container
# The old 'ollama_voice_chat.py' is replaced by these three:
COPY assistant.py .
COPY config_manager.py .
COPY audio_utils.py .
COPY config.ini .
COPY hey_glados.onnx .
# Command to run the application when the container starts
CMD ["python3", "assistant.py"]