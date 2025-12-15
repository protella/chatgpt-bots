from __future__ import annotations

import aiohttp
import re
from typing import Optional

from slack_sdk.errors import SlackApiError

from config import config


def strip_citations(text: str) -> str:
    """
    Strip MCP citation markers from text while preserving web_search citations.

    Citation formats:
    - MCP: cite:emoji:mcp_<server>.<tool>result<N>:emoji:
      Example: cite:ship:mcp_aws_knowledge.aws_search_documentationresult8:walking:
    - MCP Tool: cite:emoji:turn<N>read_documentation:emoji:
      Example: cite:ship:turn0search1:ship:turn1read_documentation:walking:
    - Web Search: cite:emoji:turn<N>search<N>:emoji: (preserved - these are clickable links)
      Example: cite:ship:turn1search4:walking:

    MCP citations are meant for ChatGPT web UI and render as emojis + backend strings in Slack.
    Web search citations render as clickable links and should be preserved.

    Args:
        text: Text potentially containing citation markers

    Returns:
        Text with MCP citation markers removed, web search citations preserved
    """
    # Pattern to match citations that should be removed:
    # - Contains "mcp_" anywhere (MCP server results)
    # - Contains "read_", "get_", "list_", etc. (MCP tool prefixes)
    # - Contains "_documentation", "_library", etc. (MCP tool suffixes)
    # Note: Using [^:]+ for final emoji to match Unicode emojis (not just \w word chars)
    mcp_patterns = [
        # Citations containing "mcp_"
        r'\s*cite:[^:]+:mcp_[^:]+(?::[^:]+)*:[^:]+:\s*',
        # Citations containing MCP tool patterns (read_documentation, etc.)
        r'\s*cite:[^:]+:[^:]*(?:read_|get_|list_|fetch_|retrieve_)[^:]+(?::[^:]+)*:[^:]+:\s*',
        r'\s*cite:[^:]+:[^:]*(?:_documentation|_library|_docs)[^:]*(?::[^:]+)*:[^:]+:\s*',
        # Complex/nested citations that contain multiple turn references (likely MCP + web search mixed)
        r'\s*cite:[^:]+:turn\d+search\d+:[^:]+:turn\d+[^:]*(?::[^:]+)*:[^:]+:\s*',
    ]

    cleaned_text = text
    for pattern in mcp_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text)

    # Clean up any double spaces created by removing citations (preserve newlines)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

    return cleaned_text


class SlackUtilitiesMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = None  # Reusable aiohttp session

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _cleanup_session(self):
        """Clean up aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self.log_debug("SlackUtilities aiohttp session closed")
    async def get_username(self, user_id: str, client) -> str:
        """Get username from user ID, with caching"""
        # Check memory cache first
        if user_id in self.user_cache and 'username' in self.user_cache[user_id]:
            return self.user_cache[user_id]['username']
        
        # Check database for user
        user_data = await self.db.get_or_create_user_async(user_id)
        if user_data.get('username'):
            # Load full user info from DB to memory cache
            user_info = await self.db.get_user_info_async(user_id)
            if user_info:
                self.user_cache[user_id] = {
                    'username': user_data['username'],
                    'real_name': user_info.get('real_name'),
                    'email': user_info.get('email'),
                    'timezone': user_info.get('timezone', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                self.log_debug(f"Loaded user info from DB for {user_id}: email={user_info.get('email')}, real_name={user_info.get('real_name')}, timezone={user_info.get('timezone')}")
                return user_data['username']
        
        try:
            # Fetch user info from Slack API
            result = await client.users_info(user=user_id)
            if result["ok"]:
                user_info = result["user"]
                # Get both display name and real name
                display_name = user_info.get("profile", {}).get("display_name")
                real_name = user_info.get("profile", {}).get("real_name")
                email = user_info.get("profile", {}).get("email")
                # Prefer display name, fall back to real name, then just the ID
                username = display_name or real_name or user_info.get("name") or user_id
                
                # Debug log for email
                self.log_debug(f"Fetched user info for {user_id}: email={email}, real_name={real_name}")
                
                # Cache both username and timezone info in memory
                self.user_cache[user_id] = {
                    'username': username,
                    'real_name': real_name,
                    'email': email,
                    'timezone': user_info.get('tz', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                
                # Save to database with all user info
                await self.db.get_or_create_user_async(user_id, username)
                await self.db.save_user_info_async(
                    user_id,
                    username=username,
                    real_name=real_name,
                    email=email,
                    timezone=user_info.get('tz', 'UTC'),
                    tz_label=user_info.get('tz_label', 'UTC'),
                    tz_offset=user_info.get('tz_offset', 0)
                )
                
                self.log_debug(f"Cached timezone info for {username}: tz={user_info.get('tz')}, tz_label={user_info.get('tz_label')}")
                return username
        except Exception as e:
            self.log_debug(f"Could not fetch username for {user_id}: {e}")
        
        return user_id  # Fallback to user ID if fetch fails

    async def get_user_timezone(self, user_id: str, client) -> str:
        """Get user's timezone, fetching if necessary"""
        # Check memory cache first
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        # Check database
        tz_info = await self.db.get_user_timezone_async(user_id)
        if tz_info:
            # Load to memory cache
            if user_id not in self.user_cache:
                self.user_cache[user_id] = {}
            self.user_cache[user_id]['timezone'] = tz_info[0]
            self.user_cache[user_id]['tz_label'] = tz_info[1]
            self.user_cache[user_id]['tz_offset'] = tz_info[2] or 0
            return tz_info[0]
        
        # Fetch user info (which will also cache it)
        await self.get_username(user_id, client)
        
        # Return timezone from cache or default to UTC
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        return 'UTC'  # Default fallback

    def extract_file_id_from_url(self, file_url: str) -> Optional[str]:
        """Extract file ID from a Slack file URL
        
        Args:
            file_url: The Slack file URL
            
        Returns:
            File ID if found, None otherwise
        """
        import re
        
        # Try to extract file ID from the URL
        patterns = [
            r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
            r'/files/[^/]+/([^/]+)/',       # permalink format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, file_url)
            if match:
                file_id = match.group(1)
                self.log_debug(f"Extracted file ID from URL: {file_id}")
                return file_id
        
        return None

    async def download_file(self, file_url: str, file_id: Optional[str] = None) -> Optional[bytes]:
        """Download a file from Slack
        
        Args:
            file_url: The Slack file URL (can be url_private or permalink)
            file_id: Optional file ID (will be extracted from URL if not provided)
        """
        try:
            # If file_id not provided, try to extract from URL
            if not file_id:
                # URL format: https://files.slack.com/files-pri/[TEAM]-[FILE_ID]/filename
                # or https://[team].slack.com/files/[USER]/[FILE_ID]/filename
                import re

                # Try to extract file ID from the URL
                patterns = [
                    r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
                    r'/files/[^/]+/([^/]+)/',       # permalink format
                ]

                for pattern in patterns:
                    match = re.search(pattern, file_url)
                    if match:
                        file_id = match.group(1)
                        self.log_debug(f"Extracted file ID from URL: {file_id}")
                        break

                if not file_id:
                    # If we can't extract ID, try direct download with the URL
                    self.log_debug("Could not extract file ID, trying direct download")
                    headers = {"Authorization": f"Bearer {config.slack_bot_token}"}

                    session = self._get_session()
                    try:
                        async with session.get(file_url, headers=headers) as response:
                            if response.status == 200:
                                return await response.read()
                            else:
                                self.log_error(f"Failed to download file directly: HTTP {response.status}")
                                return None
                    except aiohttp.ClientError as e:
                        self.log_error(f"Network error downloading file directly: {e}")
                        return None
            
            # Get file info to get the private URL
            self.log_debug(f"Getting file info for file ID: {file_id}")
            file_info = await self.app.client.files_info(file=file_id)
            
            # Check if file exists and is accessible
            if not file_info.get("ok"):
                self.log_error(f"Failed to get file info: {file_info.get('error', 'Unknown error')}")
                return None
            
            # Get the URL for downloading
            file_data = file_info.get("file", {})
            url_private = file_data.get("url_private") or file_data.get("url_private_download")
            
            if not url_private:
                self.log_error("No private URL found in file info")
                self.log_debug(f"File info keys: {file_data.keys()}")
                return None
            
            self.log_debug(f"Downloading from private URL: {url_private[:50]}...")
            
            # Download file using aiohttp with auth header
            headers = {"Authorization": f"Bearer {config.slack_bot_token}"}

            session = self._get_session()
            try:
                async with session.get(url_private, headers=headers) as response:
                    if response.status == 200:
                        # Check if we got actual image data
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            self.log_error("Got HTML instead of image data from private URL")
                            text_preview = await response.text()
                            self.log_debug(f"Response preview: {text_preview[:200]}")
                            return None
                        return await response.read()
                    else:
                        self.log_error(f"Failed to download file: HTTP {response.status}")
                        return None
            except aiohttp.ClientError as e:
                self.log_error(f"Network error downloading file: {e}")
                return None
            
        except SlackApiError as e:
            self.log_error(f"Error getting file info: {e}")
            return None
        except Exception as e:
            self.log_error(f"Error downloading file: {e}")
            return None
