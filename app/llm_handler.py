import logging
import ollama
import gc
from audio_utils import MAX_HISTORY_MESSAGES, SENTENCE_END_PUNCTUATION

class LLMHandler:
    def __init__(self, client: ollama.Client, args):
        self.client = client
        self.args = args
        self.messages = []
        self.reset_history()

    def reset_history(self):
        self.messages = [{'role': 'system', 'content': self.args.system_prompt}]

    def chat_stream(self, user_text: str):
        """Yields tokens from the LLM response."""
        self.messages.append({'role': 'user', 'content': user_text})
        self._prune_history()
        
        full_response = ""
        try:
            stream = self.client.chat(
                model=self.args.ollama_model,
                messages=self.messages,
                stream=True
            )
            for chunk in stream:
                token = chunk.get('message', {}).get('content', '')
                full_response += token
                yield token
                
            self.messages.append({'role': 'assistant', 'content': full_response})
            
        except Exception as e:
            logging.error(f"Ollama Error: {e}")
            # Rollback user message on failure
            if self.messages and self.messages[-1]['role'] == 'user':
                self.messages.pop()
            yield None

    def _prune_history(self):
        """More aggressive history management"""
        # Keep system prompt + recent exchanges
        max_pairs = MAX_HISTORY_MESSAGES
        
        if len(self.messages) > (max_pairs * 2 + 1):
            # Keep system prompt (index 0)
            system_prompt = self.messages[0]
            # Keep only recent messages
            recent = self.messages[-(max_pairs * 2):]
            self.messages = [system_prompt] + recent
            
            # Force garbage collection after major prune
            gc.collect()