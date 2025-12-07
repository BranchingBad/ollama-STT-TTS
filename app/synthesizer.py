import logging
import json
import os
import queue
import threading
import time
import numpy as np
import sounddevice as sd
from piper import PiperVoice
from audio_utils import RATE, MAX_TTS_ERRORS

class Synthesizer:
    def __init__(self, args, interrupt_event: threading.Event):
        self.args = args
        self.interrupt_event = interrupt_event
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_speaking_event = threading.Event()
        self.has_failed = threading.Event()
        
        self.voice = None
        self.sample_rate = 16000

        self._load_model()
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _load_model(self):
        logging.info("Initializing Piper TTS...")
        try:
            config_path = self.args.piper_model_path + ".json"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found: {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)
                self.sample_rate = int(config['audio']['sample_rate'])

            self.voice = PiperVoice.load(self.args.piper_model_path, config_path)
            logging.info(f"Loaded Piper voice. Rate: {self.sample_rate}Hz")
        except Exception as e:
            logging.critical(f"TTS Init Failed: {e}")
            self.has_failed.set()

    def _worker(self):
        consecutive_errors = 0
        while not self.stop_event.is_set():
            # FIX: Explicitly reset text to None at the start of every iteration
            # to prevent variable leakage from previous turns.
            text = None
            
            try:
                text = self.queue.get(timeout=0.1)
                if text is None: break

                self.is_speaking_event.set()
                with sd.OutputStream(
                    samplerate=self.sample_rate,
                    device=self.args.piper_output_device_index,
                    channels=1,
                    dtype='int16'
                ) as stream:
                    for audio_chunk in self.voice.synthesize(text):
                        if self.interrupt_event.is_set(): break
                        stream.write(np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16))
                
                consecutive_errors = 0
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS Error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_TTS_ERRORS:
                    self.has_failed.set()
                    break
            finally:
                if text is not None:
                    self.queue.task_done()
                
                if self.queue.empty():
                    self.is_speaking_event.clear()

    def speak(self, text: str):
        if not self.has_failed.is_set():
            self.queue.put(text)

    def stop(self):
        self.stop_event.set()
        self.queue.put(None)
        self.thread.join()

    def clear_queue(self):
        with self.queue.mutex:
            self.queue.queue.clear()