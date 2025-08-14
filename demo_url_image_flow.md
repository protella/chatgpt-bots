# URL Image Processing Flow

## Overview
The bot now automatically detects and processes image URLs in messages, allowing users to paste image links instead of uploading files directly.

## Implementation Details

### 1. **Image URL Detection** (`image_url_handler.py`)
- Scans message text for potential image URLs using regex patterns
- Supports common image extensions: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`
- Handles URLs with query parameters
- Recognizes common image hosting services (imgur, discord CDN, etc.)

### 2. **URL Validation & Download**
- Validates URLs by checking HTTP headers (HEAD request)
- Verifies MIME type is a supported image format
- Downloads images with size limits (20MB default)
- Converts downloaded images to base64 for processing

### 3. **Integration with Message Processing** (`message_processor.py`)
- `_process_attachments()` now checks for image URLs in message text
- Downloaded images are treated similarly to uploaded attachments
- Images are marked with source type: `"attachment"`, `"url"`, or `"generated"`

### 4. **AssetLedger Tracking** (`thread_manager.py`)
- Extended to track URL-sourced images
- Stores original URL along with image data
- New `add_url_image()` method for URL-specific tracking

### 5. **Vision Analysis Updates**
- URL images are included in vision analysis requests
- Breadcrumb messages distinguish between uploaded files and URL images
- Analysis results include all image sources

## Usage Examples

### Basic URL Image Analysis
```
User: Check out this image: https://example.com/photo.jpg
Bot: [Downloads image from URL]
     [Analyzes image]
     "This image shows..."
```

### Multiple Image Sources
```
User: [Uploads file.png] Also analyze https://site.com/image.jpg
Bot: [Processes both uploaded file and URL image]
     "Analyzing 2 images..."
     "Image 1 (uploaded): ..."
     "Image 2 (from URL): ..."
```

### URL-Only Vision Request
```
User: What's in this image? https://cdn.example.com/pic.webp
Bot: [Downloads and analyzes image]
     "The image shows..."
```

## Technical Flow

1. **Message Received** → Slack/Discord client receives message
2. **Text Processing** → `_process_attachments()` called
3. **URL Detection** → `ImageURLHandler.extract_image_urls()` scans text
4. **Validation** → Each URL validated with HEAD request
5. **Download** → Valid image URLs downloaded
6. **Conversion** → Images converted to base64
7. **Processing** → Images added to `image_inputs` list
8. **Vision API** → All images sent to OpenAI vision API
9. **AssetLedger** → URL images tracked with original URLs
10. **Response** → Analysis results sent back to user

## Benefits

1. **User Convenience**: No need to download and re-upload images
2. **Direct Link Support**: Can analyze images from any accessible URL
3. **Mixed Input**: Supports both uploaded files and URLs in same message
4. **Tracking**: Full history of URL-sourced images maintained
5. **Edit Support**: URL images can be edited like uploaded images

## Configuration

- Max image size: 20MB (configurable)
- Download timeout: 10 seconds (configurable)
- Max images per message: 10 (existing limit)
- Supported formats: JPEG, PNG, GIF, WebP

## Error Handling

- Invalid URLs logged but don't block processing
- Failed downloads reported in logs
- Network errors handled gracefully
- Users notified if no valid images found

## Future Enhancements

1. **Caching**: Cache downloaded images to avoid re-downloading
2. **Preview Generation**: Generate thumbnails for large images
3. **URL Shortening**: Display shortened URLs in responses
4. **Batch Downloads**: Parallel downloading for multiple URLs
5. **Smart Detection**: ML-based image URL detection
6. **Video Frame Extraction**: Support video URLs with frame extraction