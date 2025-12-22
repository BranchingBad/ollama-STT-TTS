import unittest
from unittest.mock import patch
import argparse
import sys
import os

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant.config_manager import load_config_and_args
from voice_assistant.audio_utils import DEFAULT_SETTINGS

class TestConfigManager(unittest.TestCase):

    @patch('voice_assistant.config_manager.sanitize_file_path', side_effect=lambda x, y: x)
    @patch('voice_assistant.config_manager.os.path.exists', return_value=False)
    @patch('argparse.ArgumentParser.parse_args')
    def test_load_config_and_args_defaults(self, mock_parse_args, mock_exists, mock_sanitize):
        """
        Test that load_config_and_args returns default settings when
        config.ini does not exist and no command-line arguments are provided.
        """
        # Mock command-line arguments to be an empty Namespace
        mock_parse_args.return_value = argparse.Namespace(
            list_devices=False, 
            list_output_devices=False,
            debug=False,
            system_prompt=DEFAULT_SETTINGS['system_prompt'] # Set default to avoid file read
        )

        # Call the function
        args, config, should_exit = load_config_and_args()

        # Assert that should_exit is False
        self.assertFalse(should_exit)

        # Assert a few key default values
        self.assertEqual(args.ollama_model, DEFAULT_SETTINGS['ollama_model'])
        self.assertEqual(args.wakeword, DEFAULT_SETTINGS['wakeword'])
        self.assertEqual(args.device_index, DEFAULT_SETTINGS['device_index'])

if __name__ == '__main__':
    unittest.main()
