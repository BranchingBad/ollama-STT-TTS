import argparse
import os
import sys
import unittest
from unittest.mock import patch

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant.audio_utils import DEFAULT_SETTINGS  # noqa: E402
from voice_assistant.config_manager import load_config_and_args  # noqa: E402


def _build_default_namespace():
    """Build an argparse.Namespace pre-populated with every attribute
    `load_config_and_args()` reads after parse_args() returns. Mirrors
    the parser.set_defaults() call inside that function."""
    ns = argparse.Namespace(
        list_devices=False,
        list_output_devices=False,
        debug=False,
    )
    # Mirror everything that goes through parser.set_defaults().
    for key in (
        'ollama_model', 'whisper_model', 'wakeword_model_path',
        'piper_model_path', 'ollama_host', 'wakeword',
        'wakeword_threshold', 'vad_aggressiveness', 'silence_seconds',
        'listen_timeout', 'pre_buffer_ms', 'gain', 'system_prompt',
        'device_index', 'piper_output_device_index',
        'max_words_per_command', 'max_phrase_duration', 'whisper_device',
        'whisper_compute_type', 'whisper_avg_logprob',
        'whisper_no_speech_prob', 'max_history_tokens', 'audio_buffer_size',
        'trim_wake_word', 'gc_interval', 'memory_profiling',
    ):
        setattr(ns, key, DEFAULT_SETTINGS[key])
    return ns


class TestConfigManager(unittest.TestCase):

    @patch('voice_assistant.config_manager.sanitize_file_path', side_effect=lambda x, y: x)
    @patch('voice_assistant.config_manager.os.path.exists', return_value=False)
    @patch('argparse.ArgumentParser.parse_args')
    def test_load_config_and_args_defaults(self, mock_parse_args, mock_exists, mock_sanitize):
        """
        Test that load_config_and_args returns default settings when
        config.ini does not exist and no command-line arguments are provided.
        """
        mock_parse_args.return_value = _build_default_namespace()

        args, config, should_exit = load_config_and_args()

        self.assertFalse(should_exit)
        self.assertEqual(args.ollama_model, DEFAULT_SETTINGS['ollama_model'])
        self.assertEqual(args.wakeword, DEFAULT_SETTINGS['wakeword'])
        self.assertEqual(args.device_index, DEFAULT_SETTINGS['device_index'])


if __name__ == '__main__':
    unittest.main()
