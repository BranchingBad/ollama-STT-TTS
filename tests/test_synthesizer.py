import argparse
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

# Make voice_assistant importable from src/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Stub out optional native dependencies BEFORE importing the synthesizer so the
# import succeeds in environments that do not have piper/sounddevice installed.
# Tests below patch attributes on these stubs directly (no string lookup).
_piper_stub = MagicMock()
_sd_stub = MagicMock()
sys.modules.setdefault('piper', _piper_stub)
sys.modules.setdefault('sounddevice', _sd_stub)

from voice_assistant import synthesizer as synth_module  # noqa: E402
from voice_assistant.synthesizer import Synthesizer  # noqa: E402


class TestSynthesizer(unittest.TestCase):

    def setUp(self):
        self.args = argparse.Namespace(
            piper_model_path="dummy_model.onnx",
            piper_output_device_index=None,
        )
        self.interrupt_event = threading.Event()
        self.mock_voice = MagicMock()
        self.mock_config_data = '{"audio": {"sample_rate": 16000}}'

    def _build_synth(self, output_stream_mock=None):
        """Construct a Synthesizer with PiperVoice.load and config-file IO patched."""
        m_open = mock_open(read_data=self.mock_config_data)
        patches = [
            patch('os.path.exists', return_value=True),
            patch('builtins.open', m_open),
            patch.object(synth_module, 'PiperVoice', create=True,
                         **{'load.return_value': self.mock_voice}),
        ]
        if output_stream_mock is not None:
            patches.append(patch.object(synth_module.sd, 'OutputStream', output_stream_mock))
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return Synthesizer(self.args, self.interrupt_event)

    def test_initialization(self):
        synthesizer = self._build_synth()
        self.assertIsNotNone(synthesizer.voice)
        self.assertEqual(synthesizer.sample_rate, 16000)
        self.assertTrue(synthesizer.thread.is_alive())
        synthesizer.stop()

    def test_speak_and_worker(self):
        mock_audio_chunk = MagicMock()
        mock_audio_chunk.audio_int16_bytes = np.random.randint(
            -32768, 32767, size=160, dtype=np.int16
        ).tobytes()
        self.mock_voice.synthesize.return_value = [mock_audio_chunk]

        output_stream_mock = MagicMock()
        synthesizer = self._build_synth(output_stream_mock=output_stream_mock)
        stream_instance = output_stream_mock.return_value.__enter__.return_value

        synthesizer.speak("Hello world")
        # Wait for worker to drain the queue.
        synthesizer.queue.join()

        self.mock_voice.synthesize.assert_called_once_with("Hello world")
        stream_instance.write.assert_called()
        synthesizer.stop()

    def test_interrupt(self):
        long_synthesis = [MagicMock() for _ in range(10)]
        for chunk in long_synthesis:
            chunk.audio_int16_bytes = b''

        def synth_side_effect(*_args):
            yield long_synthesis[0]
            self.interrupt_event.set()
            yield long_synthesis[1]

        self.mock_voice.synthesize.side_effect = synth_side_effect

        output_stream_mock = MagicMock()
        synthesizer = self._build_synth(output_stream_mock=output_stream_mock)
        stream_instance = output_stream_mock.return_value.__enter__.return_value

        synthesizer.speak("This is a long message")
        synthesizer.queue.join()

        # Only the first chunk is written; the second is skipped after interrupt.
        stream_instance.write.assert_called_once()
        synthesizer.stop()

    def test_stop(self):
        synthesizer = self._build_synth()
        thread = synthesizer.thread
        self.assertTrue(thread.is_alive())

        synthesizer.stop()

        # Give the worker a moment to exit on slow CI.
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(synthesizer.voice)


if __name__ == '__main__':
    unittest.main()
