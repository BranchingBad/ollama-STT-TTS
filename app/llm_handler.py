import logging
import ollama
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
            if self.messages[-1]['role'] == 'user':
                self.messages.pop()
            yield None

    def _prune_history(self):
        # Simple turn-based pruning for brevity
        if len(self.messages) > MAX_HISTORY_MESSAGES * 2:
            # Keep system prompt (index 0), remove oldest pair
            del self.messages[1:3]