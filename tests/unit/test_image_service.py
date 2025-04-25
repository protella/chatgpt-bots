"""Unit tests for the image_service module.

This module tests the image generation functionality from the image_service module.
"""

import unittest
from unittest.mock import patch, MagicMock
import base64
import os
import pytest
from io import BytesIO
from PIL import Image
import prompts

from app.core.image_service import generate_image, create_optimized_prompt


class TestImageService(unittest.TestCase):
    """Test cases for the image_service module."""
    
    def setUp(self):
        """Set up tests."""
        # Create a small test image in memory as bytes
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        self.test_image = img_bytes.getvalue()
        self.test_image_b64 = base64.b64encode(self.test_image).decode('utf-8')
    
    @patch('openai.OpenAI')
    def test_create_optimized_prompt(self, mock_openai):
        """Test the create_optimized_prompt function."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response from the API
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        
        # Set the mock response content
        expected_prompt = "A beautiful sunset over snow-capped mountains with golden light filtering through clouds, dramatic sky with purple and orange hues, alpine forest in the foreground, crystal clear lake reflecting the mountains"
        mock_message.content = expected_prompt
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call the function
        input_text = "Create an image of a sunset in the mountains"
        config = {"temperature": 0.7}
        result = create_optimized_prompt(input_text, "test_thread_123", config)
        
        # Verify the API was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        
        # Check the model and settings
        self.assertEqual(call_args["model"], "gpt-4.1-mini-2025-04-14")
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["store"], False)
        
        # Verify the messages format
        messages = call_args["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], prompts.IMAGE_GEN_SYSTEM_PROMPT)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], input_text)
        
        # Verify the result
        self.assertEqual(result, expected_prompt)
    
    @patch('openai.OpenAI')
    def test_create_optimized_prompt_error(self, mock_openai):
        """Test error handling in create_optimized_prompt function."""
        # Mock the OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        
        # Call the function
        input_text = "Create an image of a sunset in the mountains"
        config = {"temperature": 0.7}
        result = create_optimized_prompt(input_text, "test_thread_123", config)
        
        # Verify it falls back to the original text
        self.assertEqual(result, input_text)
    
    @patch('openai.OpenAI')
    def test_generate_image_with_gpt_image_1(self, mock_openai):
        """Test generate_image function with gpt-image-1 model."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response from the API
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.b64_json = self.test_image_b64
        mock_response.data = [mock_data]
        mock_client.images.generate.return_value = mock_response
        
        # Call the function with gpt-image-1 config
        config = {
            "image_model": "gpt-image-1",
            "size": "1024x1024"
        }
        image_bytes, revised_prompt, is_error = generate_image(
            "A beautiful sunset over mountains", 
            "test_thread_123", 
            config
        )
        
        # Verify the API was called with correct parameters
        mock_client.images.generate.assert_called_once()
        call_args = mock_client.images.generate.call_args[1]
        self.assertEqual(call_args["model"], "gpt-image-1")
        self.assertEqual(call_args["size"], "1024x1024")
        self.assertEqual(call_args["n"], 1)
        self.assertEqual(call_args["response_format"], "b64_json")
        
        # Verify the result
        self.assertEqual(image_bytes, self.test_image)
        self.assertIsNone(revised_prompt)  # No revised prompt for gpt-image-1
        self.assertFalse(is_error)
    
    @patch('openai.OpenAI')
    def test_generate_image_with_dalle_3(self, mock_openai):
        """Test generate_image function with dall-e-3 model."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response from the API
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.b64_json = self.test_image_b64
        mock_data.revised_prompt = "A stunning sunset over majestic mountains with golden light"
        mock_response.data = [mock_data]
        mock_client.images.generate.return_value = mock_response
        
        # Call the function with dall-e-3 config
        config = {
            "image_model": "dall-e-3",
            "size": "1024x1792",
            "quality": "hd",
            "style": "natural",
            "d3_revised_prompt": True
        }
        image_bytes, revised_prompt, is_error = generate_image(
            "A beautiful sunset over mountains", 
            "test_thread_123", 
            config
        )
        
        # Verify the API was called with correct parameters
        mock_client.images.generate.assert_called_once()
        call_args = mock_client.images.generate.call_args[1]
        self.assertEqual(call_args["model"], "dall-e-3")
        self.assertEqual(call_args["size"], "1024x1792")
        self.assertEqual(call_args["quality"], "hd")
        self.assertEqual(call_args["style"], "natural")
        
        # Verify the result
        self.assertEqual(image_bytes, self.test_image)
        self.assertEqual(revised_prompt, "A stunning sunset over majestic mountains with golden light")
        self.assertFalse(is_error)
    
    @patch('openai.OpenAI')
    def test_generate_image_with_default_config(self, mock_openai):
        """Test generate_image function with default configuration."""
        # Mock the OpenAI client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Mock response from the API
        mock_response = MagicMock()
        mock_data = MagicMock()
        mock_data.b64_json = self.test_image_b64
        mock_response.data = [mock_data]
        mock_client.images.generate.return_value = mock_response
        
        # Call the function with empty config (should use defaults)
        config = {}
        image_bytes, revised_prompt, is_error = generate_image(
            "A beautiful sunset over mountains", 
            "test_thread_123", 
            config
        )
        
        # Verify the API was called with default parameters
        mock_client.images.generate.assert_called_once()
        call_args = mock_client.images.generate.call_args[1]
        self.assertEqual(call_args["model"], "gpt-image-1")  # Default model
        self.assertEqual(call_args["size"], "1024x1024")  # Default size
        
        # Verify the result
        self.assertEqual(image_bytes, self.test_image)
        self.assertIsNone(revised_prompt)
        self.assertFalse(is_error)
    
    @patch('openai.OpenAI')
    def test_generate_image_error_handling(self, mock_openai):
        """Test error handling in generate_image function."""
        # Mock the OpenAI client to raise an exception
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.images.generate.side_effect = Exception("API error")
        
        # Call the function
        config = {"image_model": "gpt-image-1"}
        image_bytes, revised_prompt, is_error = generate_image(
            "A beautiful sunset over mountains", 
            "test_thread_123", 
            config
        )
        
        # Verify error handling
        self.assertEqual(image_bytes, bytes())  # Empty bytes
        self.assertIsNone(revised_prompt)
        self.assertTrue(is_error)


if __name__ == '__main__':
    unittest.main() 