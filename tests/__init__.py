"""
tests/__init__.py

Test package initialization for ollama-STT-TTS tests.
This file makes the tests/ directory a Python package and can contain
shared test configuration, fixtures, and utilities.
"""

import os
import sys
import logging

# Add the parent directory to the Python path so tests can import from app/
# This allows: from app.audio_utils import DEFAULT_SETTINGS
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Configure logging for tests
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during tests
    format='%(levelname)s - %(name)s - %(message)s'
)

# ============================================================================
# Shared Test Configuration
# ============================================================================

# Default test configuration values
TEST_CONFIG = {
    'ollama_host': os.getenv('TEST_OLLAMA_HOST', 'http://localhost:11434'),
    'test_timeout': 10,  # seconds
    'skip_slow_tests': os.getenv('SKIP_SLOW_TESTS', 'false').lower() == 'true',
}

# ============================================================================
# Shared Test Fixtures and Utilities
# ============================================================================

def get_test_ollama_host():
    """Get the Ollama host URL for testing."""
    return TEST_CONFIG['ollama_host']


def skip_if_ollama_unavailable():
    """
    Decorator to skip tests if Ollama is not available.
    
    Usage:
        @skip_if_ollama_unavailable()
        def test_something_requiring_ollama(self):
            pass
    """
    import unittest
    import requests
    
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            try:
                response = requests.get(
                    f"{TEST_CONFIG['ollama_host']}/api/tags",
                    timeout=2
                )
                if response.status_code != 200:
                    raise unittest.SkipTest("Ollama server not responding")
            except Exception as e:
                raise unittest.SkipTest(f"Ollama unavailable: {e}")
            
            return test_func(*args, **kwargs)
        
        return wrapper
    return decorator


def get_test_audio_device():
    """
    Get a test audio device index (returns None for default device).
    Can be overridden with TEST_AUDIO_DEVICE environment variable.
    """
    device_str = os.getenv('TEST_AUDIO_DEVICE', 'none')
    if device_str.lower() == 'none':
        return None
    try:
        return int(device_str)
    except ValueError:
        logging.warning(f"Invalid TEST_AUDIO_DEVICE value: {device_str}, using default")
        return None


def get_test_models_path():
    """Get the path to the models directory for testing."""
    models_path = os.path.join(PROJECT_ROOT, 'models')
    if not os.path.exists(models_path):
        logging.warning(f"Models directory not found: {models_path}")
    return models_path


# ============================================================================
# Test Data Helpers
# ============================================================================

def get_test_data_dir():
    """Get the path to test data directory."""
    test_data_dir = os.path.join(TESTS_DIR, 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    return test_data_dir


def create_mock_audio_data(duration_seconds=1.0, sample_rate=16000):
    """
    Create mock audio data for testing (sine wave).
    
    Args:
        duration_seconds: Length of audio in seconds
        sample_rate: Sample rate in Hz
    
    Returns:
        numpy array of float32 audio data
    """
    try:
        import numpy as np
        num_samples = int(duration_seconds * sample_rate)
        # Generate a 440Hz sine wave (A note)
        t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        return audio
    except ImportError:
        logging.warning("numpy not available for creating mock audio")
        return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'TEST_CONFIG',
    'get_test_ollama_host',
    'skip_if_ollama_unavailable',
    'get_test_audio_device',
    'get_test_models_path',
    'get_test_data_dir',
    'create_mock_audio_data',
]