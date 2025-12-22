import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import argparse
import numpy as np
import time

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# We need to mock torch before it's imported by the transcriber
MOCK_TORCH = MagicMock()
sys.modules['torch'] = MOCK_TORCH

from voice_assistant.transcriber import Transcriber, TRANSCRIPTION_TIMEOUT_SECONDS

class TestTranscriber(unittest.TestCase):

    def setUp(self):
        """Set up a mock model and args for each test."""
        self.args = argparse.Namespace(
            whisper_model="tiny.en",
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_avg_logprob=-0.5,
            whisper_no_speech_prob=0.7
        )

    @patch('faster_whisper.WhisperModel')
    def test_initialization_cpu(self, mock_whisper_model):
        """Test successful initialization on CPU."""
        transcriber = Transcriber(self.args)
        mock_whisper_model.assert_called_once_with(
            "tiny.en",
            device="cpu",
            compute_type="int8"
        )
        self.assertIsNotNone(transcriber.model)

    @patch('faster_whisper.WhisperModel')
    def test_initialization_cuda_fallback(self, mock_whisper_model):
        """Test fallback to CPU when CUDA is requested but not available."""
        self.args.whisper_device = "cuda"
        MOCK_TORCH.cuda.is_available.return_value = False

        transcriber = Transcriber(self.args)
        
        # Check that it fell back to CPU
        mock_whisper_model.assert_called_once_with(
            "tiny.en",
            device="cpu",
            compute_type="int8"
        )
        self.assertEqual(transcriber.device, "cpu")

    @patch('faster_whisper.WhisperModel')
    def test_transcribe_success(self, mock_whisper_model):
        """Test a successful transcription call."""
        # Mock the return value of model.transcribe
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.avg_logprob = -0.2
        mock_segment.no_speech_prob = 0.1
        
        mock_info = MagicMock(language="en", language_probability=0.99)
        
        mock_model_instance = mock_whisper_model.return_value
        mock_model_instance.transcribe.return_value = ([mock_segment], mock_info)

        transcriber = Transcriber(self.args)
        audio_np = np.random.rand(16000).astype(np.float32)
        
        result = transcriber.transcribe(audio_np)
        
        self.assertEqual(result, "Hello world")
        mock_model_instance.transcribe.assert_called_once()

    @patch('faster_whisper.WhisperModel')
    @patch('voice_assistant.transcriber.ThreadPoolExecutor')
    def test_transcribe_timeout(self, mock_executor, mock_whisper_model):
        """Test the transcription timeout mechanism."""
        # Make the future.result() raise a TimeoutError
        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError
        mock_executor.return_value.submit.return_value = mock_future

        transcriber = Transcriber(self.args)
        audio_np = np.random.rand(16000).astype(np.float32)
        
        result = transcriber.transcribe(audio_np)
        
        self.assertEqual(result, "")
        mock_executor.return_value.submit.assert_called_once()
        mock_future.result.assert_called_once_with(timeout=TRANSCRIPTION_TIMEOUT_SECONDS)


    @patch('faster_whisper.WhisperModel')
    def test_transcribe_filtering(self, mock_whisper_model):
        """Test that segments are correctly filtered based on confidence."""
        # Segment that should be accepted
        good_segment = MagicMock()
        good_segment.text = "This is a good segment. "
        good_segment.avg_logprob = -0.1
        good_segment.no_speech_prob = 0.1

        # Segment to be rejected for low log probability
        bad_logprob_segment = MagicMock()
        bad_logprob_segment.text = "This should be filtered. "
        bad_logprob_segment.avg_logprob = -0.8
        bad_logprob_segment.no_speech_prob = 0.1

        # Segment to be rejected for high no-speech probability
        bad_nospeech_segment = MagicMock()
        bad_nospeech_segment.text = "This also filtered. "
        bad_nospeech_segment.avg_logprob = -0.2
        bad_nospeech_segment.no_speech_prob = 0.9

        mock_info = MagicMock(language="en", language_probability=0.99)
        
        mock_model_instance = mock_whisper_model.return_value
        mock_model_instance.transcribe.return_value = (
            [good_segment, bad_logprob_.pysegment, bad_nospeech_segment], 
            mock_info
        )

        transcriber = Transcriber(self.args)
        audio_np = np.random.rand(16000).astype(np.float32)
        
        # We test the internal method directly to avoid threading complexities
        result = transcriber._internal_transcribe(audio_np)
        
        self.assertEqual(result, "This is a good segment.")

if __name__ == '__main__':
    unittest.main()
