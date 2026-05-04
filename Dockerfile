# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system-level dependencies required for
# sounddevice (portaudio19-dev).
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging metadata and source first so the package is installable.
COPY pyproject.toml setup.py README.md ./
COPY src/ ./src/

# Install Python dependencies (the package itself + its deps)
RUN pip install --no-cache-dir .

# Copy runtime assets (config + models) after the install layer
COPY config.ini .
COPY models/ ./models/

# Command to run the application when the container starts
CMD ["ollama-voice-assistant"]
