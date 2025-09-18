"""Unit tests for image_url_handler.py (Async Version)"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import base64
import aiohttp
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
        """Test basic image URL extraction"""
        text = "Check out this image: https://example.com/image.jpg"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/image.jpg"

        text = "Multiple images: https://site.com/pic.png and http://other.com/photo.jpeg"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 2
        assert "https://site.com/pic.png" in urls
        assert "http://other.com/photo.jpeg" in urls

    def test_extract_image_urls_extensions(self, handler):
        """Test detection of various image extensions"""
        text = """Images:
        https://example.com/image.jpg
        https://example.com/photo.jpeg
        https://example.com/pic.png
        https://example.com/animation.gif
        https://example.com/modern.webp
        """
        urls = handler.extract_image_urls(text)
        assert len(urls) == 5

    def test_extract_image_urls_case_insensitive(self, handler):
        """Test case-insensitive extension detection"""
        text = """Images:
        https://example.com/image.JPG
        https://example.com/photo.PNG
        https://example.com/pic.Jpeg
        """
        urls = handler.extract_image_urls(text)
        assert len(urls) == 3

    def test_extract_image_urls_query_params(self, handler):
        """Test URL extraction with query parameters"""
        text = "Image: https://cdn.example.com/image.jpg?size=large&quality=high"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert "https://cdn.example.com/image.jpg?size=large&quality=high" in urls

    def test_extract_image_urls_encoded(self, handler):
        """Test extraction of encoded URLs"""
        text = "Encoded: https://example.com/images%2Fphoto.jpg"
        urls = handler.extract_image_urls(text)
        assert len(urls) == 1
        assert "https://example.com/images%2Fphoto.jpg" in urls

    def test_extract_image_urls_angle_brackets(self, handler):
        """Test URLs wrapped in angle brackets (Slack format)"""
        text = "Image: <https://example.com/image.jpg>"
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

    @pytest.mark.asyncio
    async def test_validate_image_url_success(self, handler):
        """Test successful URL validation"""
        # Create a mock session
        mock_session = MagicMock()

        # Create a proper mock response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1024'}

        # Make the context manager work
        mock_session.head.return_value.__aenter__.return_value = mock_response
        mock_session.head.return_value.__aexit__.return_value = None

        # Patch the _get_session method
        with patch.object(handler, '_get_session', return_value=mock_session):
            is_valid, mimetype, error = await handler.validate_image_url("https://example.com/image.jpg")
            assert is_valid is True
            assert mimetype == 'image/jpeg'
            assert error is None

    @pytest.mark.asyncio
    async def test_validate_image_url_invalid_status(self, handler):
        """Test URL validation with non-200 status"""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 404
        mock_session.head.return_value.__aenter__.return_value = mock_response
        mock_session.head.return_value.__aexit__.return_value = None

        with patch.object(handler, '_get_session', return_value=mock_session):
            is_valid, mimetype, error = await handler.validate_image_url("https://example.com/image.jpg")
            assert is_valid is False
            assert mimetype is None
            assert "status code 404" in error

    @pytest.mark.asyncio
    async def test_download_image_success(self, handler):
        """Test successful image download"""
        # Create fake image data (PNG header)
        image_data = b'\x89PNG\r\n\x1a\n' + b'fake image data'

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'image/png'}
        mock_response.read = AsyncMock(return_value=image_data)
        mock_session.get.return_value.__aenter__.return_value = mock_response
        mock_session.get.return_value.__aexit__.return_value = None

        with patch.object(handler, '_get_session', return_value=mock_session):
            result = await handler.download_image("https://example.com/image.png", mimetype='image/png')

            assert result is not None
            assert result['url'] == "https://example.com/image.png"
            assert result['mimetype'] == 'image/png'
            assert result['size'] == len(image_data)
            assert result['data'] == image_data
            assert result['base64_data'] == base64.b64encode(image_data).decode('utf-8')

    @pytest.mark.asyncio
    async def test_process_urls_from_text_success(self, handler):
        """Test processing multiple URLs from text"""
        text = "Check these: https://example.com/image1.jpg and https://example.com/image2.png"

        with patch.object(handler, 'validate_image_url', new_callable=AsyncMock) as mock_validate:
            with patch.object(handler, 'download_image', new_callable=AsyncMock) as mock_download:
                # Mock validation results
                mock_validate.side_effect = [
                    (True, 'image/jpeg', None),
                    (True, 'image/png', None)
                ]

                # Mock download results - needs to include all expected fields
                mock_download.side_effect = [
                    {'url': 'https://example.com/image1.jpg', 'base64_data': 'data1', 'mimetype': 'image/jpeg', 'size': 100},
                    {'url': 'https://example.com/image2.png', 'base64_data': 'data2', 'mimetype': 'image/png', 'size': 200}
                ]

                downloaded, failed = await handler.process_urls_from_text(text)

                assert len(downloaded) == 2
                assert len(failed) == 0
                assert downloaded[0]['url'] == 'https://example.com/image1.jpg'
                assert downloaded[1]['url'] == 'https://example.com/image2.png'

    def test_critical_url_extraction(self, handler):
        """Critical test for URL extraction functionality"""
        # This is a core functionality that must work
        text = "Image at https://example.com/test.jpg"
        urls = handler.extract_image_urls(text)
        assert urls == ["https://example.com/test.jpg"]

    def test_smoke_basic_functionality(self, handler):
        """Smoke test for basic functionality"""
        # Basic sanity check
        assert handler is not None
        assert handler.max_image_size > 0
        assert handler.timeout > 0

        # Can extract URLs
        urls = handler.extract_image_urls("https://example.com/image.png")
        assert len(urls) == 1