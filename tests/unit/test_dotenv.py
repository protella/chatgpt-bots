#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

print("Starting dotenv test...")
print(f"Current directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")
print(f"Python path: {sys.path}")

# Try to find .env file
dotenv_path = find_dotenv()
print(f"Found .env at: {dotenv_path if dotenv_path else 'Not found'}")

print(f"Before load_dotenv: SLACK_BOT_TOKEN = {os.getenv('SLACK_BOT_TOKEN')}")

# Load environment variables
load_dotenv()

print(f"After load_dotenv: SLACK_BOT_TOKEN = {os.getenv('SLACK_BOT_TOKEN')}")
print("Dotenv test completed.") 