# Simple test to verify test setup
import os
import pytest
from dotenv import load_dotenv

load_dotenv()

def test_environment_variables():
    """Test that essential environment variables are available."""
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_KEY"]
    
    for var in required_vars:
        assert os.getenv(var) is not None, f"Missing required environment variable: {var}"

def test_simple_passing():
    """A simple passing test."""
    assert True, "This test should always pass" 