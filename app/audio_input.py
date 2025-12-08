# audio_input.py - Improvements for better silence detection

import logging
import queue
import time
import numpy as np
import sounddevice as sd
import webrtcvad
from audio_utils import (
    FORMAT_NP, CHANNELS, RATE, CHUNK_SIZE, INT16_MAX
)

class AudioInput:
    def __init__(self, args):
        self.args = args
        self.vad = webrtcvad.Vad(args.vad_aggressiveness)
        self.stream_buffer = queue.Queue(maxsize=args.audio_buffer_size)
        self.stream = None
        
        # Derived settings
        self.silence_chunks = int(args.silence_seconds * 1000 / 30)
        self.pre_buffer_chunks = int(args.pre_buffer_ms / 30)
        
        # IMPROVEMENT: Track speech confidence for better detection
        self.speech_confidence_history = []
        self.speech_confidence_window = 10  # Track last 10 chunks

    def start(self):
        if self.stream: return
        try:
            self.stream = sd.InputStream(
                samplerate=RATE,
                blocksize=CHUNK_SIZE,
                device=self.args.device_index,
                channels=CHANNELS,
                dtype=FORMAT_NP,
                callback=self._callback
            )
            self.stream.start()
            logging.debug("Audio stream started.")
        except Exception as e:
            logging.critical(f"Failed to start audio stream: {e}")
            raise

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, frames, time, status):
        if status: logging.warning(f"Audio status: {status}")
        try:
            # Apply gain
            if self.args.gain != 1.0:
                # Ensure gain is not negative
                gain = max(0, self.args.gain)
                indata = (indata.astype(np.float32) * gain).astype(np.int16)

            self.stream_buffer.put_nowait(indata.tobytes())
        except queue.Full:
            pass # Buffer full, drop chunk

    def get_chunk(self, timeout=0.01):
        try:
            chunk = self.stream_buffer.get(timeout=timeout)
            self.stream_buffer.task_done()
            return chunk
        except queue.Empty:
            return None

    def clear_buffer(self):
        """Clears all items from the audio buffer."""
        with self.stream_buffer.mutex:
            self.stream_buffer.queue.clear()
        self.speech_confidence_history.clear()

    def record_phrase(self, interrupt_event, timeout_seconds):
        """Records until silence or interruption with improved detection."""
        frames = []
        silent_chunks = 0
        is_speaking = False
        pre_buffer = []
        start_time = time.time()
        speech_start_time = None
        max_phrase_seconds = self.args.max_phrase_duration
        
        # IMPROVEMENT: Track speech energy for better silence detection
        speech_energy_history = []
        energy_window = 5
        
        # IMPROVEMENT: Minimum speech duration before allowing stop (prevent cutting off too early)
        min_speech_duration = 0.3  # seconds
        
        self.clear_buffer()

        while True:
            if interrupt_event.is_set(): 
                logging.debug("Recording interrupted by event")
                return None
            
            if not is_speaking and (time.time() - start_time > timeout_seconds):
                logging.debug(f"No speech detected within {timeout_seconds}s timeout")
                return None

            data = self.get_chunk(timeout=0.1)
            if not data: 
                continue

            # Calculate audio energy for this chunk
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / INT16_MAX
            chunk_energy = np.sqrt(np.mean(audio_chunk**2))
            
            is_speech = self.vad.is_speech(data, RATE)

            if is_speaking:
                # Check for max phrase duration timeout
                elapsed = time.time() - speech_start_time
                if elapsed > max_phrase_seconds:
                    logging.debug(f"Max phrase duration of {max_phrase_seconds}s exceeded. Forcing stop.")
                    break

                frames.append(data)
                
                # IMPROVEMENT: Track energy history
                speech_energy_history.append(chunk_energy)
                if len(speech_energy_history) > energy_window:
                    speech_energy_history.pop(0)
                
                if not is_speech:
                    silent_chunks += 1
                    
                    # IMPROVEMENT: Use energy trend to decide when to stop
                    # Only stop if we've been speaking long enough AND energy is consistently low
                    if silent_chunks > self.silence_chunks:
                        # Check if we've spoken for minimum duration
                        if elapsed >= min_speech_duration:
                            # Check recent energy trend
                            if len(speech_energy_history) >= energy_window:
                                avg_recent_energy = sum(speech_energy_history[-energy_window:]) / energy_window
                                
                                # Stop if energy has dropped significantly
                                if avg_recent_energy < 0.015:  # Very low energy threshold
                                    logging.debug(f"Silence detected after {elapsed:.2f}s "
                                                f"(energy: {avg_recent_energy:.4f})")
                                    break
                                else:
                                    logging.debug(f"VAD says silence but energy still present "
                                                f"({avg_recent_energy:.4f}), continuing...")
                                    silent_chunks = max(0, silent_chunks - 2)  # Reset counter partially
                            else:
                                break  # Not enough history, trust VAD
                        else:
                            # Too early to stop, keep recording
                            logging.debug(f"Silence detected but only {elapsed:.2f}s spoken, continuing...")
                            silent_chunks = max(0, silent_chunks - 1)
                else:
                    # Speech detected, reset silence counter
                    silent_chunks = 0
                    
            elif is_speech:
                # Speech just started
                is_speaking = True
                speech_start_time = time.time()
                logging.debug(f"Speech started, using {len(pre_buffer)} pre-buffer chunks")
                frames.extend(pre_buffer)
                frames.append(data)
                speech_energy_history = [chunk_energy]
            else:
                # No speech yet, accumulate pre-buffer
                pre_buffer.append(data)
                if len(pre_buffer) > self.pre_buffer_chunks:
                    pre_buffer.pop(0)

        if not frames:
            logging.debug("No frames recorded")
            return None
        
        # IMPROVEMENT: Check if we got meaningful audio
        total_duration = len(frames) * 0.03  # Each chunk is 30ms
        logging.debug(f"Recorded {len(frames)} chunks ({total_duration:.2f}s)")
        
        if total_duration < 0.2:  # Less than 200ms
            logging.debug(f"Recording too short ({total_duration:.2f}s), discarding")
            return None
        
        # Convert to float32 for Whisper
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_MAX
        
        # Final quality check
        final_rms = np.sqrt(np.mean(audio_np**2))
        if final_rms < 0.005:
            logging.debug(f"Final audio too quiet (RMS: {final_rms:.4f}), discarding")
            return None
        
        return audio_np