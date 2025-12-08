import logging
import numpy as np
import numpy.typing as npt
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from faster_whisper import WhisperModel

# Import torch for CUDA check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRANSCRIPTION_TIMEOUT_SECONDS = 10.0

class Transcriber:
    def __init__(self, args):
        self.args = args
        self.device = args.whisper_device
        self.compute_type = args.whisper_compute_type
        
        # Auto-detect CUDA
        if self.device == 'cuda' and (not TORCH_AVAILABLE or not torch.cuda.is_available()):
            logging.warning("CUDA not available. Falling back to CPU for Whisper.")
            self.device = 'cpu'
        elif self.device == 'cuda':
            logging.info("CUDA device found. Using 'cuda' for Whisper.")

        logging.info(f"Loading faster-whisper model: {args.whisper_model} on device '{self.device}'...")
        try:
            self.model = WhisperModel(
                args.whisper_model,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            logging.critical(f"Error loading faster-whisper model: {e}")
            raise

    def _internal_transcribe(self, audio_np: npt.NDArray[np.float32]) -> str:
        segments, _ = self.model.transcribe(audio_np, language="en")
        
        # Filter segments by confidence
        transcription = []
        for segment in segments:
            # The log probability is a negative number, so a higher value (closer to 0) is better.
            # no_speech_prob is the probability of the segment being silence, so lower is better.
            if segment.avg_logprob > self.args.whisper_avg_logprob and segment.no_speech_prob < self.args.whisper_no_speech_prob:
                transcription.append(segment.text)
            else:
                logging.debug(f"Segment discarded: avg_logprob={segment.avg_logprob:.2f}, no_speech_prob={segment.no_speech_prob:.2f}, text='{segment.text.strip()}'")

        if not transcription:
            logging.warning("No speech detected or transcription confidence too low.")
            return ""

        full_text = "".join(transcription)
        return full_text.strip()

    def transcribe(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Runs transcription with a timeout and memory cleanup."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._internal_transcribe, audio_np)
        
        try:
            result = future.result(timeout=TRANSCRIPTION_TIMEOUT_SECONDS)
            return result
        except TimeoutError:
            logging.error("Transcription timed out.")
            return ""
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return ""
        finally:
            executor.shutdown(wait=False)
            # Clear CUDA cache if using GPU
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()

    def close(self):
        if hasattr(self, 'model'):
            del self.model
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()