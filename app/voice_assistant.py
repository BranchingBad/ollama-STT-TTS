import logging
import threading
import time
import numpy as np
from openwakeword.model import Model
import os
import gc
import re

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
        
        # Wake word detection improvements
        self.last_wakeword_time = 0
        self.wakeword_cooldown = 1.0  # Reduced from 1.5s
        self.consecutive_detection_count = 0
        self.required_consecutive = 2  # Confirmations needed
        
        logging.debug(f"VoiceAssistant init - cooldown: {self.wakeword_cooldown}s, "
                     f"required consecutive: {self.required_consecutive}")
        
        # Initialize Subsystems
        self.audio = AudioInput(args)
        self.transcriber = Transcriber(args)
        self.tts = Synthesizer(args, self.interrupt_event)
        self.llm = LLMHandler(client, args)

        # Wakeword Setup
        if not os.path.exists(args.wakeword_model_path):
            raise FileNotFoundError(f"Wakeword model missing: {args.wakeword_model_path}")
        
        logging.debug(f"Loading wakeword model from: {args.wakeword_model_path}")
        self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
        self.wakeword_key = list(self.oww_model.models.keys())[0]
        logging.debug(f"Wakeword model loaded with key: {self.wakeword_key}")

    def run(self):
        logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
        self.audio.start()
        
        # Track wake word scores for debugging
        score_history = []
        
        try:
            while True:
                # 1. Get audio for Wakeword Detection
                chunk = self.audio.get_chunk()
                if not chunk:
                    time.sleep(0.001)
                    continue

                # 2. Check Wakeword with improved logic
                int16_audio = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(int16_audio)
                score = prediction.get(self.wakeword_key, 0)
                
                # Track scores for debugging (keep last 100)
                score_history.append(score)
                if len(score_history) > 100:
                    score_history.pop(0)
                
                current_time = time.time()
                
                # Enhanced wake word detection
                if score > self.args.wakeword_threshold:
                    # Check cooldown period to prevent rapid re-triggers
                    if current_time - self.last_wakeword_time > self.wakeword_cooldown:
                        # Require consistent detection to reduce false positives
                        self.consecutive_detection_count += 1
                        
                        logging.debug(f"Wakeword candidate detected (score: {score:.2f}, "
                                    f"consecutive: {self.consecutive_detection_count}/{self.required_consecutive})")
                        
                        # Trigger after required consecutive detections
                        if self.consecutive_detection_count >= self.required_consecutive:
                            # Log recent score history
                            recent_scores = [f"{s:.2f}" for s in score_history[-10:]]
                            logging.info(f"Wakeword detected! (score: {score:.2f}, recent: {', '.join(recent_scores)})")
                            
                            self.last_wakeword_time = current_time
                            self.consecutive_detection_count = 0
                            self.oww_model.reset()
                            
                            self._handle_conversation()
                            
                            # Clear score history after conversation
                            score_history.clear()
                            logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
                    else:
                        time_since_last = current_time - self.last_wakeword_time
                        logging.debug(f"Wakeword in cooldown period (score: {score:.2f}, "
                                    f"time since last: {time_since_last:.2f}s)")
                else:
                    # Reset consecutive count if score drops below threshold
                    if self.consecutive_detection_count > 0:
                        logging.debug(f"Wakeword detection sequence broken (score: {score:.2f})")
                        self.consecutive_detection_count = 0

        except KeyboardInterrupt:
            logging.info("Stopping...")
        self.cleanup()

    def _handle_conversation(self):
        conversation_start = time.time()
        
        # Optional memory profiling
        mem_before = 0
        if self.args.debug and self.args.memory_profiling:
            mem_before = monitor_memory()
            logging.debug(f"Memory at conversation start: {mem_before:.2f} MB")

        self.audio.stop()
        self.audio.clear_buffer()
        
        logging.debug("Playing acknowledgment")
        self.tts.speak("Yes?")
        self.tts.queue.join()
        
        self.interrupt_event.clear()
        
        # Start listening for command
        logging.debug("Starting audio recording for command")
        self.audio.start()
        
        # Longer delay to allow TTS audio to fade completely
        time.sleep(0.3)
        
        recording_start = time.time()
        audio_np = self.audio.record_phrase(self.interrupt_event, self.args.listen_timeout)
        recording_duration = time.time() - recording_start
        
        # Stop listening and process
        self.audio.stop()
        
        if audio_np is None:
            logging.debug(f"No audio recorded (recording took {recording_duration:.2f}s)")
            self.audio.start()
            return

        logging.debug(f"Audio recording completed in {recording_duration:.2f}s")

        # Validate audio quality before transcription
        audio_rms = np.sqrt(np.mean(audio_np**2))
        audio_peak = np.max(np.abs(audio_np))
        logging.debug(f"Audio quality check - RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}")
        
        if audio_rms < 0.005:  # Very quiet threshold
            logging.warning(f"Audio too quiet (RMS: {audio_rms:.4f}), likely silence")
            self.audio.start()
            return

        # Transcribe with retry logic
        transcription_start = time.time()
        user_text = self._transcribe_with_retry(audio_np)
        transcription_duration = time.time() - transcription_start
        
        logging.debug(f"Transcription completed in {transcription_duration:.2f}s")
        
        # Explicitly release audio data from memory
        del audio_np
        
        if not user_text or not user_text.strip():
            logging.debug("Transcription was empty or whitespace only")
            self.audio.start()
            return

        # Trim wake word if enabled
        original_text = user_text
        if self.args.trim_wake_word:
            user_text = self._trim_wakeword(user_text)
            if user_text != original_text:
                logging.debug(f"Wake word trimmed: '{original_text}' -> '{user_text}'")

        # If the command is now empty, do nothing
        if not user_text or not user_text.strip():
            logging.debug("Command empty after wake word trimming")
            self.audio.start()
            return

        # Take only the first sentence
        sentences = re.split(r'(?<=[.?!])\s+', user_text)
        if sentences:
            first_sentence = sentences[0]
            if first_sentence != user_text:
                logging.debug(f"Using first sentence only: '{first_sentence}'")
                user_text = first_sentence

        logging.info(f"You: {user_text}")

        # Check for exit commands
        user_text_lower = user_text.lower()
        if "exit" in user_text_lower or "goodbye" in user_text_lower:
            logging.debug("Exit command detected")
            self.tts.speak("Goodbye.")
            self.tts.queue.join()
            exit(0)

        # Check for history reset commands
        if "new chat" in user_text_lower or "reset chat" in user_text_lower:
            logging.debug("Chat reset command detected")
            self.llm.reset_history()
            self.tts.speak("Chat history cleared.")
            self.tts.queue.join()
            self.audio.start()
            return

        # Get LLM Response & Speak
        logging.debug("Sending to LLM")
        llm_start = time.time()
        sentence_buffer = ""
        token_count = 0
        
        for token in self.llm.chat_stream(user_text):
            if token is None: 
                logging.error("LLM returned None token")
                break
            if self.interrupt_event.is_set():
                logging.debug("Conversation interrupted")
                self.tts.clear_queue()
                break
            
            token_count += 1
            sentence_buffer += token
            
            # Stream sentences to TTS
            if any(p in token for p in SENTENCE_END_PUNCTUATION):
                sentence = sentence_buffer.strip()
                if sentence:
                    logging.debug(f"Queuing sentence for TTS: '{sentence[:50]}...'")
                    self.tts.speak(sentence)
                sentence_buffer = ""
        
        llm_duration = time.time() - llm_start
        logging.debug(f"LLM streaming completed in {llm_duration:.2f}s ({token_count} tokens)")
        
        # Speak remaining buffer
        if sentence_buffer.strip() and not self.interrupt_event.is_set():
            logging.debug(f"Queuing final buffer for TTS: '{sentence_buffer.strip()}'")
            self.tts.speak(sentence_buffer.strip())
        
        logging.debug("Waiting for TTS to complete")
        self.tts.queue.join()
        
        # After conversation completes
        self.conversation_count += 1
        conversation_duration = time.time() - conversation_start
        
        logging.debug(f"Conversation #{self.conversation_count} completed in {conversation_duration:.2f}s")
        
        # Periodic aggressive cleanup
        if self.args.gc_interval > 0 and self.conversation_count % self.args.gc_interval == 0:
            gc.collect()
            logging.debug(f"Periodic garbage collection triggered (every {self.args.gc_interval} conversations)")

        # Optional memory profiling
        if self.args.debug and self.args.memory_profiling and mem_before > 0:
            mem_after = monitor_memory()
            mem_delta = mem_after - mem_before
            logging.debug(f"Memory at conversation end: {mem_after:.2f} MB (delta: {mem_delta:+.2f} MB)")
            
        self.audio.start()

    def _transcribe_with_retry(self, audio_np: np.ndarray, max_retries: int = 2) -> str:
        """Transcribe with retry logic and progressive threshold relaxation."""
        original_logprob = self.args.whisper_avg_logprob
        original_nospeech = self.args.whisper_no_speech_prob
        
        logging.debug(f"Starting transcription (thresholds: logprob={original_logprob}, "
                     f"no_speech={original_nospeech})")
        
        for attempt in range(max_retries):
            logging.debug(f"Transcription attempt {attempt + 1}/{max_retries}")
            
            user_text = self.transcriber.transcribe(audio_np)
            
            if user_text and user_text.strip():
                # Success!
                logging.debug(f"Transcription successful on attempt {attempt + 1}: '{user_text}'")
                # Restore original thresholds
                self.args.whisper_avg_logprob = original_logprob
                self.args.whisper_no_speech_prob = original_nospeech
                return user_text
            
            # On retry, relax thresholds slightly
            if attempt < max_retries - 1:
                self.args.whisper_avg_logprob -= 0.2
                self.args.whisper_no_speech_prob += 0.15
                logging.debug(f"Transcription attempt {attempt + 1} failed, relaxing thresholds to "
                            f"logprob={self.args.whisper_avg_logprob}, "
                            f"no_speech={self.args.whisper_no_speech_prob}")
        
        # Restore original thresholds
        self.args.whisper_avg_logprob = original_logprob
        self.args.whisper_no_speech_prob = original_nospeech
        
        logging.warning(f"All {max_retries} transcription attempts failed")
        return ""

    def _trim_wakeword(self, text: str) -> str:
        """Intelligently trim wake word from transcription."""
        wakeword = self.args.wakeword.lower()
        text_lower = text.lower().strip()
        
        # Try exact match at start
        if text_lower.startswith(wakeword):
            trimmed = text[len(wakeword):].lstrip(' ,.?!').strip()
            logging.debug(f"Exact wake word match trimmed")
            return trimmed
        
        # Try fuzzy match for common mishearings
        common_variants = [
            "hey jarvis",
            "a jarvis", 
            "hey jarvas",
            "hey jarvs",
            "a jarvus",
            "hey drivers",  # Common mishearing
            "hey jarves",
            "hey jarvis,",
            "hayjarvis",
            "hey jar",
            "jarvis"  # Sometimes just the name
        ]
        
        for variant in common_variants:
            if text_lower.startswith(variant):
                trimmed = text[len(variant):].lstrip(' ,.?!').strip()
                logging.debug(f"Fuzzy wake word match ('{variant}') trimmed")
                return trimmed
        
        # Check if wake word appears mid-sentence (sometimes transcribed oddly)
        for variant in [wakeword, "jarvis"]:
            if variant in text_lower and text_lower.index(variant) < 15:
                # Wake word appears early in text
                idx = text_lower.index(variant)
                trimmed = text[idx + len(variant):].lstrip(' ,.?!').strip()
                if trimmed:  # Only trim if there's text after
                    logging.debug(f"Wake word found at position {idx}, trimmed")
                    return trimmed
        
        # If no match found, return original
        logging.debug("No wake word pattern found, keeping original text")
        return text

    def cleanup(self):
        logging.debug("Starting cleanup")
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()
        logging.debug("Cleanup complete")