#!/usr/bin/env python3
"""
test_ollama_connection.py

Standalone test script and unit tests for checking Ollama service availability.
Can be run directly or with pytest.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
import ollama
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


class OllamaConnectionTester:
    """Test Ollama service connectivity with detailed diagnostics."""
    
    def __init__(self, host: Optional[str] = None):
        """
        Initialize the tester.
        
        Args:
            host (str, optional): Ollama host URL. If None, defaults to 
                                  OLLAMA_HOST environment variable or 
                                  'http://localhost:11434'.
        """
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = self.host.rstrip('/')
        
    def test_raw_http(self) -> Dict[str, Any]:
        """Test raw HTTP connection without ollama client."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "method": "raw_http"
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "error": f"Connection refused: {e}",
                "method": "raw_http"
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out after 5 seconds",
                "method": "raw_http"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "raw_http"
            }
    
    def test_ollama_client(self) -> Dict[str, Any]:
        """Test using the ollama Python client."""
        try:
            client = ollama.Client(host=self.host)
            models = client.list()
            
            model_names = [m.get('name', 'unknown') for m in models.get('models', [])]
            
            return {
                "success": True,
                "models": model_names,
                "model_count": len(model_names),
                "method": "ollama_client"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "ollama_client"
            }
    
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the /api/health endpoint if available."""
        try:
            response = requests.get(
                f"{self.base_url}/",
                timeout=5
            )
            return {
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.text[:100] if response.text else "",
                "method": "health_endpoint"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": "health_endpoint"
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all connection tests and return comprehensive results."""
        logging.info(f"Testing Ollama connection at: {self.host}")
        logging.info("=" * 60)
        
        results = {
            "host": self.host,
            "tests": {}
        }
        
        # Test 1: Raw HTTP
        logging.info("Test 1: Raw HTTP connection...")
        http_result = self.test_raw_http()
        results["tests"]["raw_http"] = http_result
        if http_result["success"]:
            logging.info("✓ Raw HTTP connection successful")
        else:
            logging.error(f"✗ Raw HTTP failed: {http_result.get('error', 'Unknown error')}")
        
        # Test 2: Health endpoint
        logging.info("\nTest 2: Health endpoint check...")
        health_result = self.test_health_endpoint()
        results["tests"]["health"] = health_result
        if health_result["success"]:
            logging.info("✓ Health endpoint responding")
        else:
            logging.warning(f"⚠ Health endpoint check failed: {health_result.get('error', 'Unknown')}")
        
        # Test 3: Ollama client
        logging.info("\nTest 3: Ollama Python client...")
        client_result = self.test_ollama_client()
        results["tests"]["ollama_client"] = client_result
        if client_result["success"]:
            logging.info(f"✓ Ollama client connected successfully")
            logging.info(f"  Available models: {client_result['model_count']}")
            if client_result.get('models'):
                for model in client_result['models']:
                    logging.info(f"    - {model}")
        else:
            logging.error(f"✗ Ollama client failed: {client_result.get('error', 'Unknown error')}")
        
        # Overall result
        logging.info("\n" + "=" * 60)
        all_passed = all(test["success"] for test in results["tests"].values())
        results["overall_success"] = all_passed
        
        if all_passed:
            logging.info("✓ All tests PASSED - Ollama is running and accessible")
        else:
            logging.error("✗ Some tests FAILED - Check Ollama installation and configuration")
            logging.info("\nTroubleshooting tips:")
            logging.info("1. Ensure Ollama is installed: https://ollama.ai/download")
            logging.info("2. Start Ollama: Run 'ollama serve' in a terminal")
            logging.info("3. Check if another instance is using port 11434")
            logging.info("4. Verify firewall settings aren't blocking the connection")
        
        return results


# ============================================================================
# Unit Tests (compatible with pytest and unittest)
# ============================================================================

import unittest

class TestOllamaConnection(unittest.TestCase):
    """Unit tests for Ollama connectivity."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        cls.tester = OllamaConnectionTester(cls.default_host)
    
    def test_raw_http_connection(self):
        """Test raw HTTP connection to Ollama."""
        result = self.tester.test_raw_http()
        self.assertTrue(
            result["success"],
            f"Raw HTTP connection failed: {result.get('error', 'Unknown')}"
        )
    
    def test_ollama_client_connection(self):
        """Test Ollama client can connect and list models."""
        result = self.tester.test_ollama_client()
        self.assertTrue(
            result["success"],
            f"Ollama client connection failed: {result.get('error', 'Unknown')}"
        )
        self.assertIsInstance(result.get("models", []), list)
    
    def test_models_available(self):
        """Test that at least one model is available."""
        result = self.tester.test_ollama_client()
        if result["success"]:
            self.assertGreater(
                result.get("model_count", 0),
                0,
                "No models found. Please run 'ollama pull llama3' to download a model."
            )
    
    def test_health_endpoint(self):
        """Test the health/root endpoint responds."""
        result = self.tester.test_health_endpoint()
        # This test is informational - we don't fail if it doesn't exist
        if not result["success"]:
            self.skipTest(f"Health endpoint not available: {result.get('error')}")


# ============================================================================
# Standalone Script Execution
# ============================================================================

def main():
    """Main entry point for standalone execution."""
    import argparse
    
    # Determine default host
    default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    
    parser = argparse.ArgumentParser(
        description="Test Ollama web service connectivity"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help=f"Ollama host URL (default: {default_host})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Pre-test warning for local users
    if "localhost" in args.host or "127.0.0.1" in args.host:
        logging.info("=" * 60)
        logging.info("🚀 Starting tests for local Ollama instance.")
        logging.info("Quick Tip: Make sure the 'ollama serve' command is running in another terminal.")
        logging.info("=" * 60)

    # Run tests
    tester = OllamaConnectionTester(args.host)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["overall_success"] else 1)


if __name__ == "__main__":
    main()