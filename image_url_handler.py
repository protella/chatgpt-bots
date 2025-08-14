"""
Image URL Detection and Download Handler

This module handles detection of image URLs in text messages,
validates them, downloads the images, and prepares them for processing.
"""

import re
import requests
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
                
                # Skip Slack file URLs - these are handled separately
                if 'slack.com/files/' in url or 'slack-files.com' in url:
                    continue
                
                # Check if the path ends with an image extension
                if any(path.endswith(ext.lower()) for ext in IMAGE_EXTENSIONS):
                    urls.append(url)
                # Check for common image hosting patterns (but not Slack)
                elif any(host in parsed.netloc for host in ['imgur.com', 'cloudinary.com', 'cdn.discordapp.com']):
                    urls.append(url)
        
        # Filter out any Slack file URLs that might have been caught
        filtered_urls = []
        for url in urls:
            if 'slack.com/files/' in url or 'slack-files.com' in url:
                continue
            filtered_urls.append(url)
        
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
    
    def validate_image_url(self, url: str, auth_token: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
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
            response = requests.head(url, headers=headers, timeout=self.timeout, allow_redirects=True)
            
            # Check status code
            if response.status_code != 200:
                return False, None, f"URL returned status code {response.status_code}"
            
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
            
        except requests.exceptions.RequestException as e:
            return False, None, f"Failed to validate URL: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error validating URL: {str(e)}"
    
    def download_image(self, url: str, mimetype: Optional[str] = None, auth_token: Optional[str] = None) -> Optional[Dict]:
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
            
            # Download the image
            response = requests.get(url, headers=headers, timeout=self.timeout, allow_redirects=True)
            
            if response.status_code != 200:
                logger.error(f"Failed to download image from {url}: Status {response.status_code}")
                return None
            
            # Check if we got HTML instead of an image (common with auth failures)
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type:
                logger.error(f"Got HTML instead of image from {url} - likely authentication required")
                # Log first 200 chars to help debug
                logger.debug(f"Response preview: {response.text[:200]}")
                return None
            
            # Check size
            if len(response.content) > self.max_image_size:
                logger.error(f"Image from {url} too large: {len(response.content) / 1024 / 1024:.1f}MB")
                return None
            
            # Verify it's actually image data by checking magic bytes
            if len(response.content) > 4:
                # Check for common image format signatures
                header = response.content[:4]
                is_png = header[:4] == b'\x89PNG'
                is_jpeg = header[:2] == b'\xff\xd8'
                is_gif = header[:3] == b'GIF'
                is_webp = header[:4] == b'RIFF' and len(response.content) > 12 and response.content[8:12] == b'WEBP'
                
                if not any([is_png, is_jpeg, is_gif, is_webp]):
                    logger.error(f"Downloaded content from {url} does not appear to be a valid image")
                    logger.debug(f"First 20 bytes: {response.content[:20]}")
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
                    elif path.endswith('.webp') or (header[:4] == b'RIFF' and response.content[8:12] == b'WEBP'):
                        mimetype = 'image/webp'
                    else:
                        logger.error(f"Could not determine MIME type for {url}")
                        return None
            
            # Convert to base64
            base64_data = base64.b64encode(response.content).decode('utf-8')
            
            return {
                "url": url,
                "mimetype": mimetype,
                "base64_data": base64_data,
                "size": len(response.content),
                "data": response.content  # Raw bytes for upload if needed
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {url}: {str(e)}")
            return None
    
    def process_urls_from_text(self, text: str, auth_token: Optional[str] = None) -> Tuple[List[Dict], List[str]]:
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
            
            # This shouldn't happen anymore since we filter Slack URLs earlier
            if is_slack_url:
                logger.debug(f"Skipping Slack file URL (should be handled separately): {url}")
                continue
            
            # Validate the URL
            is_valid, mimetype, error = self.validate_image_url(url, token_to_use)
            
            if not is_valid:
                logger.warning(f"Invalid image URL {url}: {error}")
                failed_urls.append(url)
                continue
            
            # Download the image
            image_data = self.download_image(url, mimetype, token_to_use)
            
            if image_data:
                downloaded_images.append(image_data)
                logger.info(f"Successfully downloaded image from {url} (size: {image_data['size']} bytes)")
            else:
                logger.error(f"Failed to download image from {url}")
                failed_urls.append(url)
        
        return downloaded_images, failed_urls