"""
Configuration for pytest.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment variables at test time
from dotenv import load_dotenv
load_dotenv()

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and before tests are collected.
    """
    print(f"Running tests with Python {sys.version}")
    print(f"Project root: {project_root}")
    print(f"Test directory: {os.getcwd()}")

def pytest_collect_file(parent, path):
    """
    Skip files in the v1 directory to prevent running legacy tests.
    """
    if "v1" in str(path):
        return None 