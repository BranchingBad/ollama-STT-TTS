import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import argparse
import threading
import time
import numpy as np

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Mock piper and sounddevice before they are imported by the synthesizer
sys.modules['piper'] = MagicMock()
sys.modules['sounddevice'] = MagicMock()

from voice_assistant.synthesizer import Synthesizer

class TestSynthesizer(unittest.TestCase):

    def setUp(self):
        """Set up a mock voice and args for each test."""
        self.args = argparse.Namespace(
            piper_model_path="dummy_model.onnx",
            piper_output_device_index=None
        )
        self.interrupt_event = threading.Event()
        
        # Mock the piper voice and its config
        self.mock_voice = MagicMock()
        
        # Mock open for reading the model's json config
        self.mock_config_data = '{"audio": {"sample_rate": 16000}}'
        self.m_open = mock_open(read_data=self.mock_config_data)

    @patch('os.path.exists', return_value=True)
    def test_initialization(self, mock_exists):
        """Test successful initialization of the Synthesizer."""
        with patch('builtins.open', self.m_open):
            with patch('piper.PiperVoice.load', return_value=self.mock_voice) as mock_load:
                synthesizer = Synthesizer(self.args, self.interrupt_event)
                
                mock_load.assert_called_once_with("dummy_model.onnx", "dummy_model.onnx.json")
                self.assertIsNotNone(synthesizer.voice)
                self.assertEqual(synthesizer.sample_rate, 16000)
                self.assertTrue(synthesizer.thread.is_alive())
                
                synthesizer.stop()

    @patch('os.path.exists', return_value=True)
    @patch('sounddevice.OutputStream')
    def test_speak_and_worker(self, mock_output_stream, mock_exists):
        """Test the speak method and worker processing."""
        with patch('builtins.open', self.m_open):
            with patch('piper.PiperVoice.load', return_value=self.mock_voice):
                
                # Mock the audio synthesized by the voice
                mock_audio_chunk = MagicMock()
                mock_audio_chunk.audio_int16_bytes = np.random.randint(-32768, 32767, size=160, dtype=np.int16).tobytes()
                self.mock_voice.synthesize.return_value = [mock_audio_chunk]

                synthesizer = Synthesizer(self.args, self.interrupt_event)
                
                mock_stream_instance = mock_output_stream.return_value.__enter__.return_value

                synthesizer.speak("Hello world")
                time.sleep(0.2) # Give the worker thread time to process

                self.mock_voice.synthesize.assert_called_once_with("Hello world")
                mock_stream_instance.write.assert_called()
                
                synthesizer.stop()

    @patch('os.path.exists', return_value=True)
    @patch('sounddevice.OutputStream')
    def test_interrupt(self, mock_output_stream, mock_exists):
        """Test that the synthesis can be interrupted."""
        with patch('builtins.open', self.m_open):
            with patch('piper.PiperVoice.load', return_value=self.mock_voice):
                
                # Simulate a long synthesis process with multiple chunks
                long_synthesis = [MagicMock() for _ in range(10)]
                for chunk in long_synthesis:
                    chunk.audio_int16_bytes = b''
                
                # The side effect will set the interrupt event after the first chunk
                def a_side_effect(*args):
                    yield long_synthesis[0]
                    self.interrupt_event.set()
                    yield long_synthesis[1]

                self.mock_voice.synthesize.side_effect = a_side_effect
                
                synthesizer = Synthesizer(self.args, self.interrupt_event)
                mock_stream_instance = mock_output_stream.return_value.__enter__.return_value

                synthesizer.speak("This is a long message")
                time.sleep(0.2)

                # The stream should have been written to only once before interrupt
                mock_stream_instance.write.assert_called_once()
                
                synthesizer.stop()

    @patch('os.path.exists', return_value=True)
    def test_stop(self, mock_exists):
        """Test that the stop method cleans up resources."""
        with patch('builtins.open', self.m_open):
            with patch('piper.PiperVoice.load', return_value=self.mock_voice):
                synthesizer = Synthesizer(self.args, self.interrupt_event)
                thread = synthesizer.thread
                self.assertTrue(thread.is_alive())
                
                synthesizer.stop()

                self.assertFalse(thread.is_alive())
                self.assertIsNone(synthesizer.voice)

if __name__ == '__main__':
    unittest.main()
