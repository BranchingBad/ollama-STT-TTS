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

    def record_phrase(self, interrupt_event, timeout_seconds):
        """Records until silence or interruption."""
        frames = []
        silent_chunks = 0
        is_speaking = False
        pre_buffer = []
        start_time = time.time()
        speech_start_time = None
        max_phrase_seconds = self.args.max_phrase_duration
        
        self.clear_buffer()

        while True:
            if interrupt_event.is_set(): return None
            
            # Timeout if no speech detected initially
            if not is_speaking and (time.time() - start_time > timeout_seconds):
                return None

            data = self.get_chunk(timeout=0.1)
            if not data: continue

            is_speech = self.vad.is_speech(data, RATE)

            if is_speaking:
                # SAFETY: Stop recording if the phrase is too long (e.g., constant noise)
                if time.time() - speech_start_time > max_phrase_seconds:
                    logging.debug(f"Max phrase duration of {max_phrase_seconds}s exceeded. Forcing stop.")
                    break

                frames.append(data)
                if not is_speech:
                    silent_chunks += 1
                    if silent_chunks > self.silence_chunks:
                        break # End of sentence
                else:
                    silent_chunks = 0
            elif is_speech:
                is_speaking = True
                speech_start_time = time.time()
                logging.debug(f"Speech started, using {len(pre_buffer)} pre-buffer chunks")
                frames.extend(pre_buffer)
                frames.append(data)
            else:
                pre_buffer.append(data)
                if len(pre_buffer) > self.pre_buffer_chunks:
                    pre_buffer.pop(0)

        if not frames: return None
        
        # Convert to float32 for Whisper
        audio_data = b''.join(frames)
        return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / INT16_MAX