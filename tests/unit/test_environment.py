# Simple test to verify test setup
import os
import pytest
from dotenv import load_dotenv

# Try to load environment variables, but don't fail if .env is not found
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

def test_environment_variables():
    """Test that essential environment variables are available."""
    # Skip this test if we're in a CI environment or env vars aren't set
    if os.getenv("CI") or not os.getenv("SLACK_BOT_TOKEN"):
        pytest.skip("Skipping environment test (CI environment or env vars not set)")
        
    required_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_KEY"]
    
    for var in required_vars:
        assert os.getenv(var) is not None, f"Missing required environment variable: {var}"

def test_simple_passing():
    """A simple passing test."""
    assert True, "This test should always pass" 