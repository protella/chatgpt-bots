"""Unit tests for the image_service module.

This module tests the image generation functionality from the image_service module.
"""

import unittest
from unittest.mock import patch, MagicMock
import base64
import os
from io import BytesIO
from PIL import Image

from app.core.image_service import generate_image


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