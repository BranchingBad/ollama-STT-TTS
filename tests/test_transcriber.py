import argparse
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant import transcriber as transcriber_module  # noqa: E402
from voice_assistant.transcriber import TRANSCRIPTION_TIMEOUT_SECONDS, Transcriber  # noqa: E402


class TestTranscriber(unittest.TestCase):
    """Tests for Transcriber.

    We patch `voice_assistant.transcriber.torch` and `.TORCH_AVAILABLE` per-test
    rather than stuffing a MagicMock into `sys.modules['torch']`, because
    transformers (pulled in transitively by faster-whisper) inspects
    `torch.__spec__` at import time and rejects naive MagicMocks.
    """

    def setUp(self):
        self.args = argparse.Namespace(
            whisper_model="tiny.en",
            whisper_device="cpu",
            whisper_compute_type="int8",
            whisper_avg_logprob=-0.5,
            whisper_no_speech_prob=0.7,
        )

    @patch.object(transcriber_module, 'WhisperModel')
    def test_initialization_cpu(self, mock_whisper_model):
        """Test successful initialization on CPU."""
        transcriber = Transcriber(self.args)
        mock_whisper_model.assert_called_once_with(
            "tiny.en",
            device="cpu",
            compute_type="int8",
        )
        self.assertIsNotNone(transcriber.model)

    @patch.object(transcriber_module, 'WhisperModel')
    def test_initialization_cuda_fallback(self, mock_whisper_model):
        """Test fallback to CPU when CUDA is requested but not available."""
        self.args.whisper_device = "cuda"

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.object(transcriber_module, 'torch', fake_torch), \
             patch.object(transcriber_module, 'TORCH_AVAILABLE', True):
            transcriber = Transcriber(self.args)

        mock_whisper_model.assert_called_once_with(
            "tiny.en",
            device="cpu",
            compute_type="int8",
        )
        self.assertEqual(transcriber.device, "cpu")

    @patch.object(transcriber_module, 'WhisperModel')
    def test_transcribe_success(self, mock_whisper_model):
        """Test a successful transcription call."""
        mock_segment = MagicMock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 1.0
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
        transcriber.close()

    @patch.object(transcriber_module, 'WhisperModel')
    @patch.object(transcriber_module, 'ThreadPoolExecutor')
    def test_transcribe_timeout(self, mock_executor, mock_whisper_model):
        """Test the transcription timeout mechanism."""
        mock_future = MagicMock()
        mock_future.result.side_effect = TimeoutError
        mock_executor.return_value.submit.return_value = mock_future

        transcriber = Transcriber(self.args)
        audio_np = np.random.rand(16000).astype(np.float32)

        result = transcriber.transcribe(audio_np)

        self.assertEqual(result, "")
        mock_executor.return_value.submit.assert_called_once()
        mock_future.result.assert_called_once_with(timeout=TRANSCRIPTION_TIMEOUT_SECONDS)

    @patch.object(transcriber_module, 'WhisperModel')
    def test_transcribe_filtering(self, mock_whisper_model):
        """Test that segments are correctly filtered based on confidence."""
        good_segment = MagicMock()
        good_segment.text = "This is a good segment. "
        good_segment.start = 0.0
        good_segment.end = 1.0
        good_segment.avg_logprob = -0.1
        good_segment.no_speech_prob = 0.1

        bad_logprob_segment = MagicMock()
        bad_logprob_segment.text = "This should be filtered. "
        bad_logprob_segment.start = 1.0
        bad_logprob_segment.end = 2.0
        bad_logprob_segment.avg_logprob = -0.8
        bad_logprob_segment.no_speech_prob = 0.1

        bad_nospeech_segment = MagicMock()
        bad_nospeech_segment.text = "This also filtered. "
        bad_nospeech_segment.start = 2.0
        bad_nospeech_segment.end = 3.0
        bad_nospeech_segment.avg_logprob = -0.2
        bad_nospeech_segment.no_speech_prob = 0.9

        mock_info = MagicMock(language="en", language_probability=0.99)

        mock_model_instance = mock_whisper_model.return_value
        mock_model_instance.transcribe.return_value = (
            [good_segment, bad_logprob_segment, bad_nospeech_segment],
            mock_info,
        )

        transcriber = Transcriber(self.args)
        audio_np = np.random.rand(16000).astype(np.float32)

        result = transcriber._internal_transcribe(
            audio_np,
            avg_logprob_threshold=self.args.whisper_avg_logprob,
            no_speech_threshold=self.args.whisper_no_speech_prob,
        )

        self.assertEqual(result, "This is a good segment.")
        transcriber.close()


if __name__ == '__main__':
    unittest.main()
