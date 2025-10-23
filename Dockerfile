# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for
# pyaudio (portaudio19-dev) and pyttsx3 (espeak) on Linux.
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# NOTE: The build may fail here if the versions in requirements.txt
# (e.g., openai-whisper==20250625) are not available on PyPI.
# You may need to fix those versions first.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY ollama_voice_chat.py .
COPY config.ini .
COPY hey_glados.onnx .
# The .tflite file is not copied as the script only seems to use .onnx

# Command to run the application when the container starts
CMD ["python3", "ollama_voice_chat.py"]