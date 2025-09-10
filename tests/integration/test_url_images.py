#!/usr/bin/env python3
"""
Test script for URL image detection and downloading functionality
"""

import sys
import logging
from image_url_handler import ImageURLHandler

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_url_extraction():
    """Test URL extraction from text"""
    handler = ImageURLHandler()
    
    test_cases = [
        # Simple image URLs
        ("Check out this image: https://example.com/image.jpg", 1),
        ("Multiple images: https://site.com/pic1.png and https://site.com/pic2.jpeg", 2),
        
        # Mixed content
        ("Here's a link https://google.com and an image https://example.com/photo.webp", 1),
        
        # URLs with query parameters
        ("Image with params: https://cdn.example.com/image.jpg?width=800&height=600", 1),
        
        # Case insensitive extensions
        ("Uppercase ext: https://example.com/IMAGE.JPG", 1),
        
        # No images
        ("Just text with no URLs", 0),
        ("Non-image URL: https://example.com/document.pdf", 0),
        
        # Common image hosting patterns
        ("Imgur link: https://imgur.com/a/xyz123", 1),
        ("Discord CDN: https://cdn.discordapp.com/attachments/123/456/image.png", 1),
    ]
    
    print("Testing URL extraction:")
    print("-" * 50)
    
    for text, expected_count in test_cases:
        urls = handler.extract_image_urls(text)
        status = "✓" if len(urls) == expected_count else "✗"
        print(f"{status} Input: {text[:50]}...")
        print(f"  Expected: {expected_count}, Found: {len(urls)}")
        if urls:
            print(f"  URLs: {urls}")
        print()
    
    return True

def test_url_validation():
    """Test URL validation (requires internet connection)"""
    handler = ImageURLHandler()
    
    # Note: These are example URLs - in production you'd use real URLs
    test_urls = [
        # Valid image URLs (placeholders - replace with real URLs for testing)
        "https://via.placeholder.com/150",  # Returns a PNG
        "https://via.placeholder.com/300.jpg",  # Returns a JPEG
        
        # Invalid URLs
        "https://example.com/not-found-404.jpg",  # 404 error
        "https://example.com/document.pdf",  # Not an image
    ]
    
    print("\nTesting URL validation:")
    print("-" * 50)
    
    for url in test_urls:
        is_valid, mimetype, error = handler.validate_image_url(url)
        status = "✓" if is_valid else "✗"
        print(f"{status} URL: {url}")
        if is_valid:
            print(f"  MIME type: {mimetype}")
        else:
            print(f"  Error: {error}")
        print()
    
    return True

def test_integration():
    """Test the full integration"""
    handler = ImageURLHandler()
    
    # Test text with mixed content
    test_text = """
    Here's a regular link: https://google.com
    And here's an image: https://via.placeholder.com/200.png
    Another image: https://via.placeholder.com/300.jpg
    And a broken link: https://example.com/broken.jpg
    """
    
    print("\nTesting full integration:")
    print("-" * 50)
    print(f"Input text: {test_text[:100]}...")
    
    downloaded_images, failed_urls = handler.process_urls_from_text(test_text)
    
    print(f"\nResults:")
    print(f"  Successfully downloaded: {len(downloaded_images)} images")
    for img in downloaded_images:
        print(f"    - {img['url']} ({img['mimetype']}, {img['size']} bytes)")
    
    print(f"  Failed URLs: {len(failed_urls)}")
    for url in failed_urls:
        print(f"    - {url}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("URL Image Handler Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_url_extraction()
        
        # These tests require internet connection
        print("\n" + "=" * 60)
        print("Tests requiring internet connection:")
        print("=" * 60)
        
        try:
            test_url_validation()
            test_integration()
        except Exception as e:
            print(f"Network tests failed (this is expected if offline): {e}")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())