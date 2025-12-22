import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from io import StringIO

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant.audio_utils import list_audio_input_devices, list_audio_output_devices, monitor_memory

class TestAudioUtils(unittest.TestCase):

    @patch('sounddevice.query_devices')
    def test_list_audio_input_devices(self, mock_query_devices):
        """
        Test that list_audio_input_devices correctly lists input devices.
        """
        # Mock the return value of sounddevice.query_devices
        mock_query_devices.return_value = [
            {'name': 'mic1', 'max_input_channels': 2},
            {'name': 'mic2', 'max_input_channels': 1},
            {'name': 'speaker1', 'max_input_channels': 0}
        ]

        # Redirect stdout to capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        list_audio_input_devices()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output and assert
        output = captured_output.getvalue()
        self.assertIn("Index 0: mic1", output)
        self.assertIn("Index 1: mic2", output)
        self.assertNotIn("speaker1", output)

    @patch('sounddevice.query_devices')
    def test_list_audio_output_devices(self, mock_query_devices):
        """
        Test that list_audio_output_devices correctly lists output devices.
        """
        # Mock the return value of sounddevice.query_devices
        mock_query_devices.return_value = [
            {'name': 'speaker1', 'max_output_channels': 2},
            {'name': 'speaker2', 'max_output_channels': 1},
            {'name': 'mic1', 'max_output_channels': 0}
        ]

        # Redirect stdout to capture the output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Call the function
        list_audio_output_devices()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the output and assert
        output = captured_output.getvalue()
        self.assertIn("Index 0: speaker1", output)
        self.assertIn("Index 1: speaker2", output)
        self.assertNotIn("mic1", output)

    @patch('voice_assistant.audio_utils.psutil.Process')
    def test_monitor_memory(self, mock_psutil_process):
        """
        Test that monitor_memory correctly returns memory usage in MB.
        """
        # Mock the memory_info method to return a specific RSS value in bytes
        mock_process_instance = mock_psutil_process.return_value
        mock_process_instance.memory_info.return_value = MagicMock(rss=1024 * 1024 * 50)  # 50 MB

        # Call the function
        memory_usage = monitor_memory()

        # Assert the result
        self.assertEqual(memory_usage, 50.0)

if __name__ == '__main__':
    unittest.main()
