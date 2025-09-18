"""
Image URL Detection and Download Handler

This module handles detection of image URLs in text messages,
validates them, downloads the images, and prepares them for processing.
"""

import re
import aiohttp
import base64
from typing import List, Tuple, Optional, Dict
from urllib.parse import urlparse, unquote
import logging

# Supported image MIME types (matching OpenAI vision API requirements)
SUPPORTED_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/gif",
    "image/webp"
}

# Common image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.webp',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.WEBP'
}

logger = logging.getLogger(__name__)


class ImageURLHandler:
    """Handles detection and downloading of images from URLs"""

    def __init__(self, max_image_size: int = 20 * 1024 * 1024, timeout: int = 10):
        """
        Initialize the handler

        Args:
            max_image_size: Maximum image size in bytes (default 20MB)
            timeout: Download timeout in seconds (default 10)
        """
        self.max_image_size = max_image_size
        self.timeout = timeout
        self._session = None  # Reusable session for better resource management

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures cleanup"""
        await self.cleanup()

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def cleanup(self):
        """Clean up resources and close aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("ImageURLHandler aiohttp session closed")

    def extract_image_urls(self, text: str) -> List[str]:
        """
        Extract potential image URLs from text
        
        Args:
            text: The message text to scan for URLs
            
        Returns:
            List of potential image URLs (excluding Slack file URLs)
        """
        import html
        
        # First, handle Slack's angle bracket format <URL>
        # Replace <URL> with just URL to normalize
        text_normalized = re.sub(r'<(https?://[^>]+)>', r'\1', text)
        
        # Decode HTML entities (e.g., &amp; to &)
        text_normalized = html.unescape(text_normalized)
        
        # Regex pattern to match URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"{}|\\^`\[\]]*)?'
        
        # Find all URLs that look like image URLs
        urls = re.findall(url_pattern, text_normalized, re.IGNORECASE)
        
        # Also look for general URLs and check if they might be images
        general_url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        all_urls = re.findall(general_url_pattern, text_normalized)
        
        # Check each URL to see if it might be an image
        for url in all_urls:
            if url not in urls:
                # Parse URL and check path
                parsed = urlparse(url)
                path = unquote(parsed.path.lower())
                
                # Check if the path ends with an image extension
                if any(path.endswith(ext.lower()) for ext in IMAGE_EXTENSIONS):
                    urls.append(url)
                # Check for common image hosting patterns (including Slack)
                elif any(host in parsed.netloc for host in ['imgur.com', 'cloudinary.com', 'cdn.discordapp.com', 'slack.com', 'slack-files.com']):
                    urls.append(url)
        
        # Include all URLs for processing (Slack URLs will use auth token)
        filtered_urls = urls
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in filtered_urls:
            if url not in seen:
                seen.add(url)
                # Make sure URL is properly unescaped
                import html
                url_cleaned = html.unescape(url)
                unique_urls.append(url_cleaned)
        
        return unique_urls
    
    async def validate_image_url(self, url: str, auth_token: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate if a URL points to a supported image

        Args:
            url: The URL to validate
            auth_token: Optional authentication token for private URLs

        Returns:
            Tuple of (is_valid, mimetype, error_message)
        """
        try:
            # Set up headers with auth if provided
            headers = {}
            if auth_token:
                headers['Authorization'] = f"Bearer {auth_token}"

            # Make a HEAD request to check content type without downloading
            session = self._get_session()
            async with session.head(url, headers=headers, allow_redirects=True) as response:

                    # Check status code
                    if response.status != 200:
                        return False, None, f"URL returned status code {response.status}"

                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()

                    # Extract base MIME type (remove charset etc)
                    if ';' in content_type:
                        content_type = content_type.split(';')[0].strip()

                    # Check if it's a supported image type
                    if content_type not in SUPPORTED_IMAGE_MIMETYPES:
                        # Sometimes servers don't set correct content-type, check extension
                        parsed = urlparse(url)
                        path = unquote(parsed.path.lower())

                        if any(path.endswith(ext.lower()) for ext in IMAGE_EXTENSIONS):
                            # Extension suggests it's an image, try to determine type
                            if path.endswith(('.jpg', '.jpeg')):
                                content_type = 'image/jpeg'
                            elif path.endswith('.png'):
                                content_type = 'image/png'
                            elif path.endswith('.gif'):
                                content_type = 'image/gif'
                            elif path.endswith('.webp'):
                                content_type = 'image/webp'
                            else:
                                return False, None, f"Unsupported content type: {content_type}"
                        else:
                            return False, None, f"Not an image URL (content-type: {content_type})"

                    # Check content length if available
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size = int(content_length)
                        if size > self.max_image_size:
                            return False, None, f"Image too large: {size / 1024 / 1024:.1f}MB (max: {self.max_image_size / 1024 / 1024:.1f}MB)"

                    return True, content_type, None

        except aiohttp.ClientError as e:
            return False, None, f"Failed to validate URL: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error validating URL: {str(e)}"
    
    async def download_image(self, url: str, mimetype: Optional[str] = None, auth_token: Optional[str] = None) -> Optional[Dict]:
        """
        Download an image from a URL

        Args:
            url: The image URL to download
            mimetype: Optional MIME type if already known
            auth_token: Optional authentication token for private URLs

        Returns:
            Dict with image data or None if download failed
        """
        try:
            # Set up headers with auth if provided
            headers = {}
            if auth_token:
                headers['Authorization'] = f"Bearer {auth_token}"
                logger.debug(f"Using auth token for {url}: Bearer {auth_token[:10]}...")
            else:
                logger.debug(f"No auth token for {url}")

            # Download the image
            session = self._get_session()
            async with session.get(url, headers=headers, allow_redirects=True) as response:

                    if response.status != 200:
                        logger.error(f"Failed to download image from {url}: Status {response.status}")
                        return None

                    # Check if we got HTML instead of an image (common with auth failures)
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        logger.error(f"Got HTML instead of image from {url} - likely authentication required")
                        # Log first 200 chars to help debug
                        text_preview = await response.text()
                        logger.debug(f"Response preview: {text_preview[:200]}")
                        return None

                    # Read content
                    content = await response.read()

                    # Check size
                    if len(content) > self.max_image_size:
                        logger.error(f"Image from {url} too large: {len(content) / 1024 / 1024:.1f}MB")
                        return None

                    # Verify it's actually image data by checking magic bytes
                    if len(content) > 4:
                        # Check for common image format signatures
                        header = content[:4]
                        is_png = header[:4] == b'\x89PNG'
                        is_jpeg = header[:2] == b'\xff\xd8'
                        is_gif = header[:3] == b'GIF'
                        is_webp = header[:4] == b'RIFF' and len(content) > 12 and content[8:12] == b'WEBP'

                        if not any([is_png, is_jpeg, is_gif, is_webp]):
                            logger.error(f"Downloaded content from {url} does not appear to be a valid image")
                            logger.debug(f"First 20 bytes: {content[:20]}")
                            return None

                    # Determine MIME type if not provided
                    if not mimetype:
                        if ';' in content_type:
                            content_type = content_type.split(';')[0].strip()

                        if content_type in SUPPORTED_IMAGE_MIMETYPES:
                            mimetype = content_type
                        else:
                            # Guess from URL extension or magic bytes
                            parsed = urlparse(url)
                            path = unquote(parsed.path.lower())

                            if path.endswith(('.jpg', '.jpeg')) or header[:2] == b'\xff\xd8':
                                mimetype = 'image/jpeg'
                            elif path.endswith('.png') or header[:4] == b'\x89PNG':
                                mimetype = 'image/png'
                            elif path.endswith('.gif') or header[:3] == b'GIF':
                                mimetype = 'image/gif'
                            elif path.endswith('.webp') or (header[:4] == b'RIFF' and content[8:12] == b'WEBP'):
                                mimetype = 'image/webp'
                            else:
                                logger.error(f"Could not determine MIME type for {url}")
                                return None

                    # Convert to base64
                    base64_data = base64.b64encode(content).decode('utf-8')

                    return {
                        "url": url,
                        "mimetype": mimetype,
                        "base64_data": base64_data,
                        "size": len(content),
                        "data": content  # Raw bytes for upload if needed
                    }

        except aiohttp.ClientError as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {url}: {str(e)}")
            return None
    
    async def process_urls_from_text(self, text: str, auth_token: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
        """
        Extract and download images from URLs in text

        Args:
            text: The message text containing URLs
            auth_token: Optional authentication token for private URLs (e.g., Slack)

        Returns:
            Tuple of (downloaded_images, failed_urls)
        """
        # Extract potential image URLs
        urls = self.extract_image_urls(text)

        if not urls:
            return [], []

        downloaded_images = []
        failed_urls = []

        for url in urls:
            # Check if this is a Slack file URL that needs auth
            is_slack_url = 'slack.com/files/' in url or 'slack-files.com' in url
            token_to_use = auth_token if is_slack_url else None

            # Debug logging
            if is_slack_url:
                logger.info(f"Processing Slack URL: {url}")
                logger.info(f"Auth token available: {bool(auth_token)}")

            # For Slack URLs, we need the auth token
            if is_slack_url and not auth_token:
                logger.warning(f"Slack file URL requires authentication token: {url}")
                failed_urls.append(url)
                continue

            # Validate the URL
            is_valid, mimetype, error = await self.validate_image_url(url, token_to_use)

            if not is_valid:
                logger.warning(f"Invalid image URL {url}: {error}")
                failed_urls.append(url)
                continue

            # Download the image
            image_data = await self.download_image(url, mimetype, token_to_use)

            if image_data:
                downloaded_images.append(image_data)
                logger.info(f"Successfully downloaded image from {url} (size: {image_data['size']} bytes)")
            else:
                logger.error(f"Failed to download image from {url}")
                failed_urls.append(url)

        return downloaded_images, failed_urls