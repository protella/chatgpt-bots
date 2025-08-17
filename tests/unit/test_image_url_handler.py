"""Unit tests for image_url_handler.py"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import base64
import requests
from image_url_handler import ImageURLHandler, SUPPORTED_IMAGE_MIMETYPES, IMAGE_EXTENSIONS


class TestImageURLHandler:
    """Test ImageURLHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Create an ImageURLHandler instance"""
        return ImageURLHandler(max_image_size=10*1024*1024, timeout=5)
    
    def test_initialization(self):
        """Test handler initialization with default and custom values"""
        # Default values
        handler = ImageURLHandler()
        assert handler.max_image_size == 20 * 1024 * 1024
        assert handler.timeout == 10
        
        # Custom values
        handler = ImageURLHandler(max_image_size=5*1024*1024, timeout=30)
        assert handler.max_image_size == 5 * 1024 * 1024
        assert handler.timeout == 30
    
    def test_extract_image_urls_basic(self, handler):
        """Test extracting basic image URLs from text"""
        text = "Check out this image: https://example.com/image.jpg"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg"
    
    def test_extract_image_urls_multiple(self, handler):
        """Test extracting multiple image URLs"""
        text = """Here are some images:
        https://example.com/image1.jpg
        https://example.com/image2.png
        https://example.com/image3.gif"""
        urls = handler.extract_image_urls(text)
        assert len(urls) == 3
        assert "https://example.com/image1.jpg" in urls
        assert "https://example.com/image2.png" in urls
        assert "https://example.com/image3.gif" in urls
    
    def test_extract_image_urls_slack_format(self, handler):
        """Test extracting URLs from Slack's angle bracket format"""
        text = "Image: <https://example.com/image.jpg>"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg"
    
    def test_extract_image_urls_with_query_params(self, handler):
        """Test extracting URLs with query parameters"""
        text = "Image: https://example.com/image.jpg?size=large&quality=high"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg?size=large&quality=high"
    
    def test_extract_image_urls_html_entities(self, handler):
        """Test extracting URLs with HTML entities"""
        text = "Image: https://example.com/image.jpg?param=value&amp;other=test"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg?param=value&other=test"
    
    def test_extract_image_urls_case_insensitive(self, handler):
        """Test extracting URLs with mixed case extensions"""
        text = "Images: https://example.com/image.JPG and https://example.com/photo.PNG"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 2
    
    def test_extract_image_urls_duplicates(self, handler):
        """Test duplicate URL removal"""
        text = """Image: https://example.com/image.jpg
        Same image: https://example.com/image.jpg"""
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg"
    
    def test_extract_image_urls_hosting_patterns(self, handler):
        """Test detection of image hosting service URLs"""
        text = """Images from hosts:
        https://imgur.com/abc123
        https://cdn.discordapp.com/attachments/123/456/image
        https://files.slack.com/files-pri/T123/F456/image"""
        urls = handler.extract_image_urls(text)
        assert len(urls) == 3
    
    def test_extract_image_urls_no_images(self, handler):
        """Test with text containing no image URLs"""
        text = "This is just text with no URLs"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 0
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_success(self, mock_head, handler):
        """Test successful URL validation"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1024'}
        mock_head.return_value = mock_response
        
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/image.jpg")
        assert is_valid is True
        assert mimetype == 'image/jpeg'
        assert error is None
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_invalid_status(self, mock_head, handler):
        """Test URL validation with non-200 status"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response
        
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/image.jpg")
        assert is_valid is False
        assert mimetype is None
        assert "status code 404" in error
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_unsupported_type(self, mock_head, handler):
        """Test URL validation with unsupported content type"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_head.return_value = mock_response
        
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/page.html")
        assert is_valid is False
        assert mimetype is None
        assert "Not an image URL" in error
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_size_limit(self, mock_head, handler):
        """Test URL validation with size exceeding limit"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {
            'content-type': 'image/jpeg',
            'content-length': str(30 * 1024 * 1024)  # 30MB
        }
        mock_head.return_value = mock_response
        
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/large.jpg")
        assert is_valid is False
        assert mimetype is None
        assert "Image too large" in error
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_with_auth(self, mock_head, handler):
        """Test URL validation with authentication token"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'image/png'}
        mock_head.return_value = mock_response
        
        is_valid, mimetype, error = handler.validate_image_url(
            "https://files.slack.com/image.png",
            auth_token="xoxb-123456"
        )
        
        assert is_valid is True
        assert mimetype == 'image/png'
        mock_head.assert_called_once()
        call_args = mock_head.call_args
        assert 'Authorization' in call_args[1]['headers']
        assert call_args[1]['headers']['Authorization'] == "Bearer xoxb-123456"
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_type_from_extension(self, mock_head, handler):
        """Test inferring content type from file extension"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'application/octet-stream'}
        mock_head.return_value = mock_response
        
        # Should infer from .png extension
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/image.png")
        assert is_valid is True
        assert mimetype == 'image/png'
    
    @patch('image_url_handler.requests.head')
    def test_validate_image_url_exception(self, mock_head, handler):
        """Test URL validation with request exception"""
        mock_head.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        is_valid, mimetype, error = handler.validate_image_url("https://example.com/image.jpg")
        assert is_valid is False
        assert mimetype is None
        assert "Failed to validate URL" in error
    
    @patch('image_url_handler.requests.get')
    def test_download_image_success(self, mock_get, handler):
        """Test successful image download"""
        # Create fake image data (PNG header)
        image_data = b'\x89PNG\r\n\x1a\n' + b'fake image data'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/png'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/image.png", mimetype='image/png')
        
        assert result is not None
        assert result['url'] == "https://example.com/image.png"
        assert result['mimetype'] == 'image/png'
        assert result['size'] == len(image_data)
        assert result['data'] == image_data
        assert result['base64_data'] == base64.b64encode(image_data).decode('utf-8')
    
    @patch('image_url_handler.requests.get')
    def test_download_image_jpeg(self, mock_get, handler):
        """Test downloading JPEG image"""
        # JPEG header
        image_data = b'\xff\xd8\xff\xe0' + b'fake jpeg data'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/photo.jpg")
        
        assert result is not None
        assert result['mimetype'] == 'image/jpeg'
    
    @patch('image_url_handler.requests.get')
    def test_download_image_gif(self, mock_get, handler):
        """Test downloading GIF image"""
        # GIF header
        image_data = b'GIF89a' + b'fake gif data'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/gif'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/anim.gif")
        
        assert result is not None
        assert result['mimetype'] == 'image/gif'
    
    @patch('image_url_handler.requests.get')
    def test_download_image_webp(self, mock_get, handler):
        """Test downloading WebP image"""
        # WebP header
        image_data = b'RIFF\x00\x00\x00\x00WEBP' + b'fake webp data'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/webp'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/modern.webp")
        
        assert result is not None
        assert result['mimetype'] == 'image/webp'
    
    @patch('image_url_handler.requests.get')
    def test_download_image_failed_status(self, mock_get, handler):
        """Test download with non-200 status"""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/image.png")
        assert result is None
    
    @patch('image_url_handler.requests.get')
    def test_download_image_html_response(self, mock_get, handler):
        """Test download returning HTML (auth failure)"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = "<html>Login required</html>"
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/image.png")
        assert result is None
    
    @patch('image_url_handler.requests.get')
    def test_download_image_too_large(self, mock_get, handler):
        """Test download of image exceeding size limit"""
        image_data = b'x' * (handler.max_image_size + 1)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/huge.jpg")
        assert result is None
    
    @patch('image_url_handler.requests.get')
    def test_download_image_invalid_data(self, mock_get, handler):
        """Test download of non-image data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'This is not image data'
        mock_response.headers = {'content-type': 'image/jpeg'}
        mock_get.return_value = mock_response
        
        result = handler.download_image("https://example.com/fake.jpg")
        assert result is None
    
    @patch('image_url_handler.requests.get')
    def test_download_image_with_auth(self, mock_get, handler):
        """Test download with authentication token"""
        image_data = b'\x89PNG\r\n\x1a\n' + b'fake image data'
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = image_data
        mock_response.headers = {'content-type': 'image/png'}
        mock_get.return_value = mock_response
        
        result = handler.download_image(
            "https://files.slack.com/image.png",
            auth_token="xoxb-123456"
        )
        
        assert result is not None
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert 'Authorization' in call_args[1]['headers']
        assert call_args[1]['headers']['Authorization'] == "Bearer xoxb-123456"
    
    @patch('image_url_handler.requests.get')
    def test_download_image_exception(self, mock_get, handler):
        """Test download with request exception"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        result = handler.download_image("https://example.com/image.png")
        assert result is None
    
    @patch.object(ImageURLHandler, 'download_image')
    @patch.object(ImageURLHandler, 'validate_image_url')
    def test_process_urls_from_text_success(self, mock_validate, mock_download, handler):
        """Test processing URLs from text successfully"""
        text = "Check out https://example.com/image1.jpg and https://example.com/image2.png"
        
        # Mock validation
        mock_validate.side_effect = [
            (True, 'image/jpeg', None),
            (True, 'image/png', None)
        ]
        
        # Mock downloads
        mock_download.side_effect = [
            {'url': 'https://example.com/image1.jpg', 'mimetype': 'image/jpeg', 'size': 1024},
            {'url': 'https://example.com/image2.png', 'mimetype': 'image/png', 'size': 2048}
        ]
        
        images, failed = handler.process_urls_from_text(text)
        
        assert len(images) == 2
        assert len(failed) == 0
        assert images[0]['url'] == 'https://example.com/image1.jpg'
        assert images[1]['url'] == 'https://example.com/image2.png'
    
    @patch.object(ImageURLHandler, 'validate_image_url')
    def test_process_urls_from_text_validation_failure(self, mock_validate, handler):
        """Test processing URLs with validation failure"""
        text = "Check out https://example.com/fake-image.jpg"
        
        mock_validate.return_value = (False, None, "Not an image")
        
        images, failed = handler.process_urls_from_text(text)
        
        assert len(images) == 0
        assert len(failed) == 1
        assert failed[0] == 'https://example.com/fake-image.jpg'
    
    def test_process_urls_from_text_slack_without_auth(self, handler):
        """Test processing Slack URLs without auth token"""
        text = "Slack image: https://slack.com/files/T123/F456/image.png"
        
        images, failed = handler.process_urls_from_text(text)
        
        assert len(images) == 0
        assert len(failed) == 1
        assert 'slack.com' in failed[0]
    
    @patch.object(ImageURLHandler, 'download_image')
    @patch.object(ImageURLHandler, 'validate_image_url')
    def test_process_urls_from_text_slack_with_auth(self, mock_validate, mock_download, handler):
        """Test processing Slack URLs with auth token"""
        text = "Slack image: https://slack.com/files/T123/F456/image.png"
        
        mock_validate.return_value = (True, 'image/png', None)
        mock_download.return_value = {
            'url': 'https://slack.com/files/T123/F456/image.png',
            'mimetype': 'image/png',
            'size': 1024
        }
        
        images, failed = handler.process_urls_from_text(text, auth_token="xoxb-123456")
        
        assert len(images) == 1
        assert len(failed) == 0
        # Verify auth token was passed to validate and download
        mock_validate.assert_called_with(
            'https://slack.com/files/T123/F456/image.png',
            "xoxb-123456"
        )
    
    @patch.object(ImageURLHandler, 'download_image')
    @patch.object(ImageURLHandler, 'validate_image_url')
    def test_process_urls_from_text_mixed_results(self, mock_validate, mock_download, handler):
        """Test processing URLs with mixed success/failure"""
        text = "Images: https://example.com/good.jpg and https://example.com/bad.png"
        
        mock_validate.side_effect = [
            (True, 'image/jpeg', None),
            (True, 'image/png', None)
        ]
        
        mock_download.side_effect = [
            {'url': 'https://example.com/good.jpg', 'mimetype': 'image/jpeg', 'size': 1024},
            None  # Download fails
        ]
        
        images, failed = handler.process_urls_from_text(text)
        
        assert len(images) == 1
        assert len(failed) == 1
        assert images[0]['url'] == 'https://example.com/good.jpg'
        assert failed[0] == 'https://example.com/bad.png'
    
    def test_process_urls_from_text_no_urls(self, handler):
        """Test processing text with no URLs"""
        text = "This is just plain text"
        
        images, failed = handler.process_urls_from_text(text)
        
        assert len(images) == 0
        assert len(failed) == 0


class TestImageURLHandlerIntegration:
    """Integration tests for ImageURLHandler"""
    
    @pytest.mark.integration
    def test_smoke_basic_url_extraction(self):
        """Smoke test for basic URL extraction"""
        handler = ImageURLHandler()
        text = "Image at https://example.com/test.jpg"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/test.jpg"
    
    @pytest.mark.critical
    def test_critical_image_format_detection(self):
        """Critical test for detecting supported image formats"""
        handler = ImageURLHandler()
        
        # Test all supported formats
        for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            text = f"https://example.com/image{ext}"
            urls = handler.extract_image_urls(text)
            assert len(urls) == 1
            assert urls[0].endswith(ext)
    
    def test_contract_handler_interface(self):
        """Test that ImageURLHandler maintains expected interface"""
        handler = ImageURLHandler()
        
        # Required methods exist
        assert hasattr(handler, 'extract_image_urls')
        assert hasattr(handler, 'validate_image_url')
        assert hasattr(handler, 'download_image')
        assert hasattr(handler, 'process_urls_from_text')
        
        # Required attributes exist
        assert hasattr(handler, 'max_image_size')
        assert hasattr(handler, 'timeout')
    
    def test_constants_defined(self):
        """Test that required constants are defined"""
        assert SUPPORTED_IMAGE_MIMETYPES is not None
        assert isinstance(SUPPORTED_IMAGE_MIMETYPES, set)
        assert 'image/jpeg' in SUPPORTED_IMAGE_MIMETYPES
        assert 'image/png' in SUPPORTED_IMAGE_MIMETYPES
        
        assert IMAGE_EXTENSIONS is not None
        assert isinstance(IMAGE_EXTENSIONS, set)
        assert '.jpg' in IMAGE_EXTENSIONS
        assert '.png' in IMAGE_EXTENSIONS