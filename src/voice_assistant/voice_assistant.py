import logging
import threading
import time
import numpy as np
from openwakeword.model import Model
import os
import gc
import re

# Import our new modules
from .audio_input import AudioInput
from .transcriber import Transcriber
from .synthesizer import Synthesizer
from .llm_handler import LLMHandler
from .audio_utils import SENTENCE_END_PUNCTUATION, monitor_memory
from .wakeword import WakewordDetector

class VoiceAssistant:
    def __init__(self, args, client):
        self.args = args
        self.interrupt_event = threading.Event()
        self.conversation_count = 0
        self.is_handling_conversation = False

        self.wakeword_detector = WakewordDetector(
            threshold=args.wakeword_threshold,
            cooldown_seconds=1.0,
            required_consecutive=2,
            window=5,
        )

        logging.debug(
            f"VoiceAssistant init - cooldown: {self.wakeword_detector.cooldown_seconds}s, "
            f"required consecutive: {self.wakeword_detector.required_consecutive}"
        )
        
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
        model_keys = list(self.oww_model.models.keys())
        if not model_keys:
            raise RuntimeError(
                f"openWakeWord loaded no models from '{args.wakeword_model_path}'. "
                "The file may be corrupt or incompatible."
            )
        self.wakeword_key = model_keys[0]
        logging.debug(f"Wakeword model loaded with key: {self.wakeword_key}")

    def run(self):
        logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
        self.audio.start()

        try:
            while True:
                if self.is_handling_conversation:
                    time.sleep(0.01)
                    continue

                chunk = self.audio.get_chunk()
                if not chunk:
                    time.sleep(0.001)
                    continue

                int16_audio = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(int16_audio)
                score = prediction.get(self.wakeword_key, 0)

                decision = self.wakeword_detector.feed(score, time.time())

                if decision.in_cooldown:
                    logging.debug(f"Wakeword in cooldown period (score: {score:.2f})")
                    continue

                if decision.triggered:
                    logging.info(
                        f"Wakeword detected! (score: {decision.score:.2f}, "
                        f"avg: {decision.avg_score:.2f})"
                    )
                    self.oww_model.reset()
                    self.is_handling_conversation = True
                    self._handle_conversation()
                    self.wakeword_detector.reset()
                    logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
                elif decision.consecutive > 0:
                    logging.debug(
                        f"Wakeword candidate (score: {decision.score:.2f}, "
                        f"avg: {decision.avg_score:.2f}, "
                        f"consecutive: {decision.consecutive}/"
                        f"{self.wakeword_detector.required_consecutive})"
                    )

        except KeyboardInterrupt:
            logging.info("Stopping...")

    def _process_plugins(self, text: str) -> str:
        """Processes simple plugins like [current time]."""
        if "[current time]" in text.lower():
            current_time = time.strftime("%I:%M %p")
            logging.debug(f"Plugin found: [current time] -> {current_time}")
            # Use regex for case-insensitive replacement
            text = re.sub(r'\[current time\]', current_time, text, flags=re.IGNORECASE)
        return text

    def _handle_conversation(self):
        try:
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
            time.sleep(0.4)
            
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
    
            # IMPROVEMENT: More sophisticated audio quality validation
            audio_rms = np.sqrt(np.mean(audio_np**2))
            audio_peak = np.max(np.abs(audio_np))
            audio_std = np.std(audio_np)
            
            logging.debug(f"Audio quality - RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}, StdDev: {audio_std:.4f}")
            
            # Check for multiple quality indicators
            if audio_rms < 0.01:
                logging.warning(f"Audio too quiet (RMS: {audio_rms:.4f}), proceeding to transcription")
            
            if audio_std < 0.005:
                logging.warning(f"Audio lacks variation (StdDev: {audio_std:.4f}), likely silence, proceeding to transcription")

            
            # Check if audio is clipping (saturated)
            if audio_peak > 0.98:
                logging.warning(f"Audio may be clipping (Peak: {audio_peak:.4f})")
                # Don't return - just warn, as clipped audio can still be transcribed
    
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

            # Process any plugins
            user_text = self._process_plugins(user_text)

            logging.info(f"You: {user_text}")

            # Check for exit commands
            user_text_lower = user_text.lower()
            if "exit" in user_text_lower or "goodbye" in user_text_lower:
                logging.debug("Exit command detected")
                self.tts.speak("Goodbye.")
                self.tts.queue.join()
                # Raise to let assistant.py's finally block run cleanup.
                raise KeyboardInterrupt("User requested exit")
    
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
                        logging.debug(f"Queuing sentence for TTS: '{sentence[:50]}...'" )
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
        finally:
            self.is_handling_conversation = False

    def _transcribe_with_retry(self, audio_np: np.ndarray, max_retries: int = 3) -> str:
        """Transcribe with progressive threshold relaxation. Pure-functional: no
        mutation of self.args, so it's safe under concurrent / interrupted use."""
        base_logprob = self.args.whisper_avg_logprob
        base_nospeech = self.args.whisper_no_speech_prob

        threshold_steps = [
            (base_logprob, base_nospeech),
            (base_logprob - 0.15, base_nospeech + 0.1),
            (base_logprob - 0.3, base_nospeech + 0.2),
        ]

        logging.debug(f"Starting transcription (initial thresholds: logprob={base_logprob}, no_speech={base_nospeech})")

        for attempt in range(min(max_retries, len(threshold_steps))):
            logprob_threshold, nospeech_threshold = threshold_steps[attempt]
            logging.debug(
                f"Transcription attempt {attempt + 1}/{max_retries} "
                f"(logprob={logprob_threshold:.2f}, no_speech={nospeech_threshold:.2f})"
            )

            user_text = self.transcriber.transcribe(
                audio_np,
                avg_logprob_threshold=logprob_threshold,
                no_speech_threshold=nospeech_threshold,
            )

            if user_text and user_text.strip():
                logging.debug(f"Transcription successful on attempt {attempt + 1}: '{user_text}'")
                return user_text

            if attempt < max_retries - 1:
                logging.debug(f"Attempt {attempt + 1} failed, trying with relaxed thresholds")

        logging.warning(f"All {max_retries} transcription attempts failed")
        return ""

    def _trim_wakeword(self, text: str) -> str:
        """Trims the wake word from the transcription using regex for robustness."""
        wakeword = self.args.wakeword.lower()
        text_lower = text.lower().strip()

        # Generate a list of core wake word names (e.g., "jarvis" from "hey jarvis")
        # and common misspellings/pronunciations that should be trimmed.
        # This list should NOT contain partial words that could lead to over-trimming.
        core_wakeword_names = ["jarvis", "jarlis", "jarvas", "jarves", "jarvys", "jarvois"]
        
        # Add the exact configured wake word to the patterns, in case it's a multi-word phrase
        patterns_to_match = [re.escape(wakeword)] # Escape for regex safety
        
        # Add patterns for common prefixes/suffixes around the core names
        for name in core_wakeword_names:
            patterns_to_match.append(r"(?:hey\s*)?" + re.escape(name)) # Optional "hey "
            patterns_to_match.append(re.escape(name)) # Just the name

        # Create a single regex pattern to match any of these at the start or end,
        # with optional punctuation and spaces. Use word boundaries where appropriate.
        # This regex will look for the pattern either at the beginning (^) or the end ($)
        # of the string, allowing for flexible matching.
        
        # Example: if wakeword is "hey jarvis"
        # patterns_to_match could be: ["hey\\s*jarvis", "hey\\s*jarlis", ..., "jarvis", "jarlis", ...]
        
        # Construct the full regex:
        # 1. Match at the beginning: (?:<pattern>)\b[.,!?]*\s*
        # 2. Match at the end: \s*\b(?:<pattern>)[.,!?]*$
        
        # To avoid over-trimming, ensure word boundaries (\b) are used where logical.
        # Also, make sure the most specific patterns are tried first if using an OR separated list.
        
        # For simplicity and to avoid complex lookarounds, we'll try to find the longest match first
        # and then trim. A single regex can be structured to capture the matched wake word part.
        
        # Let's build a regex that captures the wake word part we want to remove.
        # We need to ensure we don't accidentally trim valid speech that happens to contain
        # a wake word component.
        
        # Pattern for matching at the beginning (case-insensitive)
        # e.g., "hey jarvis, what time" -> "what time"
        # (?:^|\s) ensures we match at the start or after a space, \b for word boundary
        combined_start_pattern = r"^(?:" + "|".join(patterns_to_match) + r")\b[.,!?]*\s*"
        match_start = re.match(combined_start_pattern, text_lower, re.IGNORECASE)
        if match_start:
            trimmed_text = text[len(match_start.group(0)):].strip()
            logging.debug(f"Wake word trimmed from start: '{match_start.group(0)}' removed. Result: '{trimmed_text}'")
            return trimmed_text
            
        # Pattern for matching at the end (case-insensitive)
        # e.g., "what time hey jarvis" -> "what time"
        combined_end_pattern = r"\s*\b(?:" + "|".join(patterns_to_match) + r")[.,!?]*$"
        match_end = re.search(combined_end_pattern, text_lower, re.IGNORECASE)
        if match_end:
            trimmed_text = text[:match_end.start()].strip()
            logging.debug(f"Wake word trimmed from end: '{match_end.group(0)}' removed. Result: '{trimmed_text}'")
            return trimmed_text

        logging.debug("No wake word pattern found, keeping original text")
        return text

    def cleanup(self):
        logging.debug("Starting cleanup")
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()
        logging.debug("Cleanup complete")
