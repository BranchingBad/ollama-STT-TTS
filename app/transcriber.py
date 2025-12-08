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
            logging.debug(f"Whisper model loaded successfully")
        except Exception as e:
            logging.critical(f"Error loading faster-whisper model: {e}")
            raise

    def _internal_transcribe(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Internal transcription with detailed logging."""
        logging.debug(f"Starting Whisper transcription (audio length: {len(audio_np)} samples, "
                     f"{len(audio_np)/16000:.2f}s)")
        
        try:
            segments, info = self.model.transcribe(
                audio_np, 
                language="en",
                vad_filter=False,  # We've already done VAD
                condition_on_previous_text=True,
                log_prob_threshold=None,
                compression_ratio_threshold=None
            )
            
            logging.debug(f"Transcription info - language: {info.language}, "
                         f"language_probability: {info.language_probability:.2f}")
            
            # Process segments with detailed logging
            transcription = []
            segment_count = 0
            discarded_count = 0
            
            for segment in segments:
                segment_count += 1
                
                # Log each segment for debugging
                logging.debug(f"Segment {segment_count}: "
                            f"[{segment.start:.2f}s-{segment.end:.2f}s] "
                            f"avg_logprob={segment.avg_logprob:.3f}, "
                            f"no_speech_prob={segment.no_speech_prob:.3f}, "
                            f"text='{segment.text.strip()}'")
                
                # Check confidence thresholds
                if segment.avg_logprob > self.args.whisper_avg_logprob and \
                   segment.no_speech_prob < self.args.whisper_no_speech_prob:
                    transcription.append(segment.text)
                    logging.debug(f"  ✓ Segment accepted")
                else:
                    discarded_count += 1
                    reasons = []
                    if segment.avg_logprob <= self.args.whisper_avg_logprob:
                        reasons.append(f"low_logprob({segment.avg_logprob:.3f}<={self.args.whisper_avg_logprob})")
                    if segment.no_speech_prob >= self.args.whisper_no_speech_prob:
                        reasons.append(f"high_nospeech({segment.no_speech_prob:.3f}>={self.args.whisper_no_speech_prob})")
                    logging.debug(f"  ✗ Segment discarded: {', '.join(reasons)}")

            if not transcription:
                logging.warning(f"No valid segments found ({segment_count} total, {discarded_count} discarded)")
                return ""

            full_text = "".join(transcription)
            logging.debug(f"Transcription result: '{full_text.strip()}' "
                         f"({len(transcription)}/{segment_count} segments used)")
            
            return full_text.strip()
            
        except Exception as e:
            logging.error(f"Whisper transcription error: {e}", exc_info=True)
            return ""

    def transcribe(self, audio_np: npt.NDArray[np.float32]) -> str:
        """Runs transcription with a timeout and memory cleanup."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._internal_transcribe, audio_np)
        
        try:
            result = future.result(timeout=TRANSCRIPTION_TIMEOUT_SECONDS)
            return result
        except TimeoutError:
            logging.error(f"Transcription timed out after {TRANSCRIPTION_TIMEOUT_SECONDS}s")
            return ""
        except Exception as e:
            logging.error(f"Transcription error: {e}", exc_info=True)
            return ""
        finally:
            executor.shutdown(wait=False)
            # Clear CUDA cache if using GPU
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared")

    def close(self):
        """Clean up model resources."""
        logging.debug("Closing Whisper transcriber")
        if hasattr(self, 'model'):
            del self.model
            if self.device == 'cuda' and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared on close")