import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import argparse

# To import voice_assistant modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant.llm_handler import LLMHandler
from voice_assistant.audio_utils import MAX_HISTORY_MESSAGES

class TestLLMHandler(unittest.TestCase):

    def setUp(self):
        """Set up a mock client and args for each test."""
        self.mock_client = MagicMock()
        self.args = argparse.Namespace(
            system_prompt="You are a helpful assistant.",
            ollama_model="test_model"
        )
        self.handler = LLMHandler(self.mock_client, self.args)

    def test_initialization(self):
        """Test that the handler initializes with the correct system prompt."""
        self.assertEqual(len(self.handler.messages), 1)
        self.assertEqual(self.handler.messages[0]['role'], 'system')
        self.assertEqual(self.handler.messages[0]['content'], self.args.system_prompt)

    def test_reset_history(self):
        """Test that reset_history correctly resets the message history."""
        self.handler.messages.append({'role': 'user', 'content': 'Hello'})
        self.handler.messages.append({'role': 'assistant', 'content': 'Hi there!'})
        self.assertEqual(len(self.handler.messages), 3)

        self.handler.reset_history()
        self.assertEqual(len(self.handler.messages), 1)
        self.assertEqual(self.handler.messages[0]['role'], 'system')

    def test_chat_stream_success(self):
        """Test a successful chat stream response."""
        user_text = "Tell me a joke."
        mock_response_chunks = [
            {'message': {'content': 'Why'}},
            {'message': {'content': ' did'}},
            {'message': {'content': ' the'}},
            {'message': {'content': ' scarecrow'}},
            {'message': {'content': ' win'}},
            {'message': {'content': ' an'}},
            {'message': {'content': ' award?'}},
        ]
        self.mock_client.chat.return_value = mock_response_chunks

        response_generator = self.handler.chat_stream(user_text)
        
        full_response = "".join(list(response_generator))

        # Check the streamed response
        self.assertEqual(full_response, "Why did the scarecrow win an award?")

        # Check that history is updated correctly
        self.assertEqual(len(self.handler.messages), 3)
        self.assertEqual(self.handler.messages[1]['role'], 'user')
        self.assertEqual(self.handler.messages[1]['content'], user_text)
        self.assertEqual(self.handler.messages[2]['role'], 'assistant')
        self.assertEqual(self.handler.messages[2]['content'], "Why did the scarecrow win an award?")

    def test_chat_stream_error(self):
        """Test that the chat stream handles errors gracefully."""
        self.mock_client.chat.side_effect = Exception("Ollama connection failed")

        response_generator = self.handler.chat_stream("What's up?")
        
        # The generator should yield None on error
        result = list(response_generator)
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0])

        # Check that the user message was rolled back from history
        self.assertEqual(len(self.handler.messages), 1)
        self.assertEqual(self.handler.messages[0]['role'], 'system')

    @patch('voice_assistant.llm_handler.gc.collect')
    def test_prune_history(self, mock_gc):
        """Test that the history pruning logic works correctly."""
        # Fill history beyond the max limit
        # MAX_HISTORY_MESSAGES is the number of pairs, so total messages = pairs * 2 + 1 (system)
        for i in range(MAX_HISTORY_MESSAGES + 5):
            self.handler.messages.append({'role': 'user', 'content': f'User message {i}'})
            self.handler.messages.append({'role': 'assistant', 'content': f'Assistant response {i}'})

        initial_length = len(self.handler.messages)
        
        # This will trigger the prune
        self.handler.chat_stream("This is the final message.")

        # Check that the history has been pruned
        final_length = len(self.handler.messages)
        expected_length = (MAX_HISTORY_MESSAGES * 2) + 1 # System prompt + max pairs
        
        # After the new message is added, the history is pruned, then the new user message is added
        # And after the response, the assistant message is added.
        # So we expect the length to be expected_length + 2 (user + assistant)
        # But for this test, we only care that it's pruned.
        # Let's call _prune_history directly for a more direct test.
        
        self.handler.messages = [{'role': 'system', 'content': self.args.system_prompt}]
        for i in range(MAX_HISTORY_MESSAGES + 5):
            self.handler.messages.append({'role': 'user', 'content': f'User message {i}'})
            self.handler.messages.append({'role': 'assistant', 'content': f'Assistant response {i}'})
        
        self.handler._prune_history()

        self.assertEqual(len(self.handler.messages), expected_length)
        # Check that the system prompt is still the first message
        self.assertEqual(self.handler.messages[0]['role'], 'system')
        # Check that the last message is the most recent one
        self.assertEqual(self.handler.messages[-1]['content'], f'Assistant response {MAX_HISTORY_MESSAGES + 4}')
        
        # Check if garbage collection was called
        mock_gc.assert_called_once()


if __name__ == '__main__':
    unittest.main()
