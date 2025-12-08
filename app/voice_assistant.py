import logging
import threading
import time
import numpy as np
from openwakeword.model import Model
import os
import gc

# Import our new modules
from audio_input import AudioInput
from transcriber import Transcriber
from synthesizer import Synthesizer
from llm_handler import LLMHandler
from audio_utils import SENTENCE_END_PUNCTUATION, monitor_memory

class VoiceAssistant:
    def __init__(self, args, client):
        self.args = args
        self.interrupt_event = threading.Event()
        self.conversation_count = 0
        
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
                    logging.info(f"Ready! Listening for '{self.args.wakeword}'...")

        except KeyboardInterrupt:
            logging.info("Stopping...")
        finally:
            self.cleanup()

    def _handle_conversation(self):
        # Optional memory profiling
        mem_before = 0
        if self.args.debug and self.args.memory_profiling:
            mem_before = monitor_memory()

        self.audio.stop()
        self.tts.speak("Yes?")
        self.tts.queue.join()
        
        self.interrupt_event.clear()
        
        # Start listening for command
        self.audio.start()
        audio_np = self.audio.record_phrase(self.interrupt_event, self.args.listen_timeout)
        
        # Stop listening and process
        self.audio.stop()
        
        if audio_np is None:
            self.audio.start() # Ensure we start listening again before exiting
            return

        # Transcribe
        user_text = self.transcriber.transcribe(audio_np)
        
        # Explicitly release audio data from memory
        del audio_np
        
        if not user_text or not user_text.strip():
            self.audio.start() # Ensure we start listening again before exiting
            return

        # Trim wake word if enabled
        if self.args.trim_wake_word:
            wakeword = self.args.wakeword.lower()
            user_text_stripped = user_text.strip()
            if user_text_stripped.lower().startswith(wakeword):
                # Trim the wake word and any following punctuation/space
                user_text = user_text_stripped[len(wakeword):].lstrip(' ,.').strip()

        # If the command is now empty, do nothing
        if not user_text:
            self.audio.start()
            return

        logging.info(f"You: {user_text}")

        # Check for exit commands
        if "exit" in user_text.lower() or "goodbye" in user_text.lower():
            self.tts.speak("Goodbye.")
            self.tts.stop()
            exit(0)

        # Get LLM Response & Speak
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
        
        # After conversation completes
        self.conversation_count += 1
        
        # Periodic aggressive cleanup
        if self.args.gc_interval > 0 and self.conversation_count % self.args.gc_interval == 0:
            gc.collect()
            logging.debug("Periodic garbage collection triggered")

        # Optional memory profiling
        if self.args.debug and self.args.memory_profiling and mem_before > 0:
            mem_after = monitor_memory()
            logging.debug(f"Memory delta for conversation: {mem_after - mem_before:.2f} MB")
            
        self.audio.start()

    def cleanup(self):
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()