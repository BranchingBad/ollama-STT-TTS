#!/usr/bin/env python3

import logging
import threading
import time
import numpy as np
from openwakeword.model import Model
import os

# Import our new modules
from audio_input import AudioInput
from transcriber import Transcriber
from synthesizer import Synthesizer
from llm_handler import LLMHandler
from audio_utils import SENTENCE_END_PUNCTUATION

class VoiceAssistant:
    def __init__(self, args, client):
        self.args = args
        self.interrupt_event = threading.Event()
        
        # Initialize Subsystems
        self.audio = AudioInput(args)
        self.transcriber = Transcriber(args)
        self.tts = Synthesizer(args, self.interrupt_event)
        self.llm = LLMHandler(client, args)

        # Wakeword Setup
        if not os.path.exists(args.wakeword_model_path):
            raise FileNotFoundError(f"Wakeword model missing: {args.wakeword_model_path}")
        self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
        self.wakeword_key = list(self.oww_model.models.keys())[0]

    def run(self):
        logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
        self.audio.start()
        
        try:
            while True:
                # 1. Get audio for Wakeword Detection
                chunk = self.audio.get_chunk()
                if not chunk:
                    time.sleep(0.001)
                    continue

                # 2. Check Wakeword
                int16_audio = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(int16_audio)
                
                if prediction.get(self.wakeword_key, 0) > self.args.wakeword_threshold:
                    logging.info("Wakeword detected!")
                    self.oww_model.reset()
                    self._handle_conversation()
                    logging.info(f"\nReady! Listening for '{self.args.wakeword}'...")

        except KeyboardInterrupt:
            logging.info("Stopping...")
        finally:
            self.cleanup()

    def _handle_conversation(self):
        self.tts.speak("Yes?")
        self.interrupt_event.clear()
        
        # 3. Listen for Command
        audio_np = self.audio.record_phrase(self.interrupt_event, self.args.listen_timeout)
        if audio_np is None:
            return

        # 4. Transcribe
        user_text = self.transcriber.transcribe(audio_np)
        logging.info(f"You: {user_text}")
        
        if not user_text.strip(): return

        # Check for exit commands
        if "exit" in user_text.lower():
            self.tts.speak("Goodbye.")
            self.tts.stop()
            exit(0)

        # 5. Get LLM Response & Speak
        sentence_buffer = ""
        for token in self.llm.chat_stream(user_text):
            if token is None: break # Error
            if self.interrupt_event.is_set():
                self.tts.clear_queue()
                break
                
            sentence_buffer += token
            # Stream sentences to TTS
            if any(p in token for p in SENTENCE_END_PUNCTUATION):
                self.tts.speak(sentence_buffer.strip())
                sentence_buffer = ""
        
        # Speak remaining buffer
        if sentence_buffer.strip() and not self.interrupt_event.is_set():
            self.tts.speak(sentence_buffer.strip())
            
        self.tts.queue.join() # Wait for speech to finish

    def cleanup(self):
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()