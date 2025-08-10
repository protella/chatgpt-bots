"""
Base Client Abstract Class
Defines the interface that all chat clients (Slack, Discord, etc.) must implement
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from logger import LoggerMixin


@dataclass
class Message:
    """Universal message format"""
    text: str
    user_id: str
    channel_id: str
    thread_id: str
    attachments: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Response:
    """Universal response format"""
    type: str  # 'text', 'image', 'file'
    content: Any
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseClient(ABC, LoggerMixin):
    """Abstract base class for all chat clients"""
    
    def __init__(self, name: str):
        self.name = name
        self.log_info(f"{name} client initialized")
    
    @abstractmethod
    def start(self):
        """Start the client and begin listening for events"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the client gracefully"""
        pass
    
    @abstractmethod
    def send_message(self, channel_id: str, thread_id: str, text: str) -> bool:
        """Send a text message"""
        pass
    
    @abstractmethod
    def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str, caption: str = "") -> bool:
        """Send an image"""
        pass
    
    @abstractmethod
    def send_thinking_indicator(self, channel_id: str, thread_id: str) -> Optional[str]:
        """Send a thinking/processing indicator"""
        pass
    
    @abstractmethod
    def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message"""
        pass
    
    def update_message(self, channel_id: str, message_id: str, text: str) -> bool:
        """Update a message (optional - not all platforms support this)"""
        return False
    
    @abstractmethod
    def get_thread_history(self, channel_id: str, thread_id: str, limit: int = 50) -> List[Message]:
        """Get message history for a thread"""
        pass
    
    @abstractmethod
    def download_file(self, file_url: str, file_id: str) -> Optional[bytes]:
        """Download a file/image from the platform"""
        pass
    
    @abstractmethod
    def format_text(self, text: str) -> str:
        """Format text for the specific platform (markdown conversion)"""
        pass
    
    def handle_error(self, channel_id: str, thread_id: str, error: str):
        """Default error handler"""
        self.log_error(f"Error in {self.name}: {error}")
        
        # Format error message for better readability
        formatted_error = self.format_error_message(error)
        self.send_message(channel_id, thread_id, formatted_error)
    
    def format_error_message(self, error: str) -> str:
        """Format error messages for display (can be overridden by platform-specific clients)"""
        return f"Error: {error}"