# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for
# sounddevice (portaudio19-dev).
# 'espeak' is no longer required as we've moved to Piper TTS.
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the modularized application files
COPY assistant.py .
COPY config_manager.py .
COPY audio_utils.py .
COPY config.ini .

# Create a models directory and copy all models into it
# This includes the wakeword and the Piper TTS models
RUN mkdir models
COPY models/ ./models/

# Command to run the application when the container starts
CMD ["python3", "assistant.py"]