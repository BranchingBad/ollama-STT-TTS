import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel

# Import torch for CUDA check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

TRANSCRIPTION_TIMEOUT_SECONDS = 15.0  # Increased from 10

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
            logging.debug("Whisper model loaded successfully")
        except Exception as e:
            logging.critical(f"Error loading faster-whisper model: {e}")
            raise

        # Re-use a single worker thread across calls so we don't leak threads
        # on long sessions. Whisper itself is not safe to call concurrently
        # from multiple threads, so max_workers=1 is intentional.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="whisper"
        )

    def _internal_transcribe(
        self,
        audio_np: npt.NDArray[np.float32],
        avg_logprob_threshold: float,
        no_speech_threshold: float,
    ) -> str:
        """Internal transcription with detailed logging."""
        logging.debug(f"Starting Whisper transcription (audio length: {len(audio_np)} samples, {len(audio_np)/16000:.2f}s)")

        try:
            segments, info = self.model.transcribe(
                audio_np,
                language="en",
                vad_filter=False,  # We've already done VAD
                condition_on_previous_text=True,
                log_prob_threshold=None,
                compression_ratio_threshold=None
            )

            logging.debug(f"Transcription info - language: {info.language}, language_probability: {info.language_probability:.2f}")

            transcription = []
            segment_count = 0
            discarded_count = 0

            for segment in segments:
                segment_count += 1

                logging.debug(
                    f"Segment {segment_count}: [{segment.start:.2f}s-{segment.end:.2f}s] "
                    f"avg_logprob={segment.avg_logprob:.3f}, "
                    f"no_speech_prob={segment.no_speech_prob:.3f}, "
                    f"text='{segment.text.strip()}'"
                )

                if segment.avg_logprob > avg_logprob_threshold and \
                   segment.no_speech_prob < no_speech_threshold:
                    transcription.append(segment.text)
                    logging.debug("  ✓ Segment accepted")
                else:
                    discarded_count += 1
                    reasons = []
                    if segment.avg_logprob <= avg_logprob_threshold:
                        reasons.append(f"low_logprob({segment.avg_logprob:.3f}<={avg_logprob_threshold})")
                    if segment.no_speech_prob >= no_speech_threshold:
                        reasons.append(f"high_nospeech({segment.no_speech_prob:.3f}>={no_speech_threshold})")
                    logging.debug(f"  ✗ Segment discarded: {', '.join(reasons)}")

            if not transcription:
                logging.warning(f"No valid segments found ({segment_count} total, {discarded_count} discarded)")
                return ""

            full_text = "".join(transcription)
            logging.debug(f"Transcription result: '{full_text.strip()}' ({len(transcription)}/{segment_count} segments used)")

            return full_text.strip()

        except Exception as e:
            logging.error(f"Whisper transcription error: {e}", exc_info=True)
            return ""

    def transcribe(
        self,
        audio_np: npt.NDArray[np.float32],
        avg_logprob_threshold: float | None = None,
        no_speech_threshold: float | None = None,
    ) -> str:
        """Runs transcription with a timeout. Thresholds default to args."""
        if avg_logprob_threshold is None:
            avg_logprob_threshold = self.args.whisper_avg_logprob
        if no_speech_threshold is None:
            no_speech_threshold = self.args.whisper_no_speech_prob

        future = self._executor.submit(
            self._internal_transcribe,
            audio_np,
            avg_logprob_threshold,
            no_speech_threshold,
        )

        try:
            return future.result(timeout=TRANSCRIPTION_TIMEOUT_SECONDS)
        except TimeoutError:
            logging.error(f"Transcription timed out after {TRANSCRIPTION_TIMEOUT_SECONDS}s")
            future.cancel()
            return ""
        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            return ""
        finally:
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared")

    def close(self):
        """Clean up model resources."""
        logging.debug("Closing Whisper transcriber")
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self, 'model'):
            del self.model
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared on close")
