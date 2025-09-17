from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Optional


@dataclass
class ImageData:
    """Container for image data"""

    base64_data: str
    format: str = "png"
    prompt: str = ""
    timestamp: float = 0
    slack_url: Optional[str] = None

    def to_bytes(self) -> BytesIO:
        """Convert base64 to BytesIO"""
        return BytesIO(base64.b64decode(self.base64_data))
