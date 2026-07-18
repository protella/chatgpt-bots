"""
Image URL Detection and Download Handler

This module handles detection of image URLs in text messages,
validates them, downloads the images, and prepares them for processing.
"""

import asyncio
import ipaddress
import re
import socket
import aiohttp
import base64
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict
from urllib.parse import urljoin, urlparse, unquote
import logging

# F50: these used to be defined here, and this module was the ONLY one that consulted them —
# the attachment path validated nothing at all and the gate kept a third, narrower list. One
# definition now, in image_validation. (Extensions are matched lower-cased at every call site
# below, so the old upper-case duplicates were never load-bearing.)
from image_validation import (
    ensure_api_compatible,
    API_IMAGE_EXTENSIONS as IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# SSRF protection for ARBITRARY user-posted image URLs.
#
# A URL a user drops in a message is attacker-controlled: left unguarded, "download this image"
# is a server-side request forgery primitive (hit the cloud metadata endpoint, a loopback admin
# port, an internal service) and an unbounded-body memory sink. ambient_fetch.py already solves
# exactly this for the link reader, but its hardened fetcher is a SEPARATE boundary that blocks
# Slack hosts outright and never sends auth — neither of which fits image download, where Slack
# CDN URLs are the primary, trusted, AUTHENTICATED case. So rather than reuse it (which would
# weaken its stance or break Slack downloads), the logic is MIRRORED here and applied ONLY to
# untrusted (non-Slack) hosts: Slack downloads keep the shared authenticated session; every
# other host is resolved-and-validated per hop, pinned to the validated IPs, and streamed under
# a hard byte cap regardless of Content-Length.
#
# The pure validation (IP classification, resolve) and the transport are behind test seams
# (set_resolver / set_guarded_opener), so the boundary is testable with no real sockets.

_SSRF_MAX_REDIRECTS = 5
# Slack hosts are trusted: an authenticated CDN we deliberately send the bot token to. Everything
# else is attacker-controlled and goes through the SSRF-guarded path with no auth.
_TRUSTED_HOST_SUFFIXES = ("slack.com", "slack-files.com")


class _SSRFBlocked(Exception):
    """A URL/host/IP the untrusted image fetcher refuses to touch."""


def _is_trusted_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return any(host == suf or host.endswith("." + suf) for suf in _TRUSTED_HOST_SUFFIXES)


def _ip_is_global(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """Globally routable public address only — loopback/private/link-local/reserved/multicast/
    unspecified, plus IPv4 smuggled inside IPv6 (mapped/6to4/teredo), are refused. Mirrors
    ambient_fetch._ip_is_global; the cloud metadata endpoint 169.254.169.254 is link-local and
    thus rejected, including when wrapped as ::ffff:169.254.169.254."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _ip_is_global(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return _ip_is_global(ip.sixtofour)
        teredo = ip.teredo
        if teredo is not None:
            return _ip_is_global(teredo[1])
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        return False
    return bool(getattr(ip, "is_global", True))


def _system_getaddrinfo(host: str, port: Optional[int]) -> List[Tuple]:
    return socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP)


_resolver = _system_getaddrinfo


def set_resolver(fn) -> None:
    """Test seam: replace the getaddrinfo-shaped resolver so no real DNS is touched in tests."""
    global _resolver
    _resolver = fn


def _resolve_and_validate(host: str, port: Optional[int]) -> List[str]:
    """Resolve `host` to its globally-routable IPs, or raise _SSRFBlocked. STRICT: if the name
    resolves to ANY non-global address, the whole name is rejected — a rebinding answer mixing a
    public and a private IP must not slip the private one through by ordering. A literal IP host
    is validated directly (no DNS)."""
    try:
        literal = ipaddress.ip_address(host)
        if not _ip_is_global(literal):
            raise _SSRFBlocked(f"non-global literal ip: {host}")
        return [str(literal)]
    except ValueError:
        pass  # a hostname; resolve it
    try:
        infos = _resolver(host, port)
    except _SSRFBlocked:
        raise
    except Exception as e:  # noqa: BLE001 — DNS failure is a hard stop
        raise _SSRFBlocked(f"dns resolution failed for {host}: {e}")
    ips: List[str] = []
    for info in infos or []:
        sockaddr = info[4] if len(info) >= 5 else None
        if not sockaddr:
            continue
        try:
            ip = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])  # strip zone id
        except ValueError:
            raise _SSRFBlocked(f"unparseable resolved address: {sockaddr[0]!r}")
        if not _ip_is_global(ip):
            raise _SSRFBlocked(f"{host} resolves to non-global {sockaddr[0]}")
        ips.append(str(ip))
    if not ips:
        raise _SSRFBlocked(f"no addresses for {host}")
    return list(dict.fromkeys(ips))  # de-dup, preserve order


class _PinnedResolver:
    """aiohttp resolver returning ONLY the pre-validated IPs, so aiohttp performs no second,
    unvalidated DNS lookup between our check and its connect (the DNS-rebinding window)."""

    def __init__(self, ips: List[str]):
        self._ips = ips

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        out = []
        for ip in self._ips:
            try:
                fam = socket.AF_INET6 if ipaddress.ip_address(ip).version == 6 else socket.AF_INET
            except ValueError:
                continue
            out.append({"hostname": host, "host": ip, "port": port,
                        "family": fam, "proto": socket.IPPROTO_TCP, "flags": 0})
        if not out:
            raise OSError(f"no validated addresses for {host}")
        return out

    async def close(self):
        return None


@dataclass
class _GuardedResponse:
    """Minimal response the guarded fetcher consumes; lets the transport be mocked in tests."""
    status: int
    headers: Dict[str, str]
    url: str
    iter_chunks: Any            # (chunk_size) -> async iterator of byte chunks
    release: Any                # awaitable cleanup


async def _default_guarded_opener(url: str, validated_ips: List[str], *, timeout: float,
                                  max_bytes: int) -> _GuardedResponse:
    """Real aiohttp transport for the untrusted path: pinned to `validated_ips`, redirects OFF
    (the fetcher follows them manually, re-validating each hop), no auth, no ambient env."""
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(
        resolver=_PinnedResolver(validated_ips), ttl_dns_cache=0,  # type: ignore[arg-type]
        force_close=True, enable_cleanup_closed=True)
    session = aiohttp.ClientSession(connector=connector, timeout=client_timeout, trust_env=False)
    try:
        resp = await session.get(url, allow_redirects=False)
    except BaseException:
        await session.close()
        raise

    async def _iter(chunk_size: int):
        async for chunk in resp.content.iter_chunked(chunk_size):
            yield chunk

    async def _release():
        try:
            resp.release()
        finally:
            await session.close()

    return _GuardedResponse(status=resp.status, headers=dict(resp.headers),
                            url=str(resp.url), iter_chunks=_iter, release=_release)


_guarded_opener = None  # Optional transport override for tests


def set_guarded_opener(fn) -> None:
    """Test seam: replace the untrusted-path transport (default uses aiohttp)."""
    global _guarded_opener
    _guarded_opener = fn


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
        # NO network for ANY host — only a cheap shape check. The real gate is download_image,
        # which sniffs the actual bytes and, crucially, is fully SSRF/redirect-hardened for both
        # host classes: a trusted (Slack) URL follows redirects MANUALLY (auth only ever sent to
        # Slack hosts; the first off-Slack hop is handed to the credential-free guarded flow), and
        # an untrusted URL runs the full guards. An authenticated HEAD here — the old trusted path,
        # with allow_redirects=True — would reintroduce exactly what the download path was hardened
        # against: an off-Slack redirect could leak the bot's auth header and hit an unvalidated
        # target during validation. So validate never touches the network; download decides.
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
            return False, None, "Not a valid http(s) URL"
        return True, None, None
    
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
            if _is_trusted_host(url):
                content, content_type = await self._download_trusted(url, auth_token)
            else:
                # An untrusted host never receives the auth token, and is fetched only under the
                # SSRF guards (validated IPs, pinned connection, manual redirect re-validation,
                # streaming cap).
                content, content_type = await self._download_guarded(url)
        except aiohttp.ClientError as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {url}: {str(e)}")
            return None

        if content is None:
            return None

        # Check if we got HTML instead of an image (common with auth failures)
        if content_type and 'text/html' in content_type:
            logger.error(f"Got HTML instead of image from {url} - likely authentication required")
            return None

        # The bytes decide. Not the content-type header, not the URL's extension, and not the
        # `mimetype` our caller passed — that came from a HEAD response, which is a claim, not
        # proof. The old code here trusted a passed mimetype outright and accepted `GIF` magic
        # unconditionally, so a URL to an ANIMATED gif sailed through and 400'd the turn.
        api_bytes, verdict = ensure_api_compatible(content)
        if api_bytes is None:
            logger.error(f"Rejecting image from {url}: {verdict}")
            logger.debug(f"First 20 bytes: {content[:20]}")
            return None
        content = api_bytes  # transcoded PNG when the source format needed it
        mimetype = verdict

        # The streaming cap above bounded the DOWNLOAD (and guards memory); it does not bound the
        # RESULT. A compressed source under the limit can decode and re-encode to a much larger
        # PNG, which would then be base64'd and sent unchecked. Enforce the ceiling again on the
        # bytes we actually send.
        if len(content) > self.max_image_size:
            logger.error(
                f"Rejecting image from {url}: transcoded to "
                f"{len(content) / 1024 / 1024:.1f}MB "
                f"(max {self.max_image_size / 1024 / 1024:.1f}MB)")
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

    async def _read_capped(self, response) -> Optional[bytes]:
        """Stream a response body into memory under the hard cap, REGARDLESS of Content-Length.
        Returns the bytes, or None if the body ran past the ceiling — an absent or lying
        Content-Length header must not turn the cap off (the old ``response.read()`` buffered the
        whole body first and only then checked its length)."""
        buf = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > self.max_image_size:
                return None
        return bytes(buf)

    async def _download_trusted(self, url: str,
                                auth_token: Optional[str]) -> Tuple[Optional[bytes], Optional[str]]:
        """Slack CDN download: the shared AUTHENTICATED session, body streamed under the hard cap.

        Redirects are followed MANUALLY and validated per hop. A Slack ``url_private`` legitimately
        30x's, but auto-following would make a Slack-hosted link that redirects OFF Slack an
        open-redirect SSRF path — and would carry the bot's auth token to wherever it pointed. So a
        hop that stays on a trusted Slack host keeps the auth header; the FIRST hop that leaves
        Slack is handed to the guarded untrusted flow (SSRF-validated, no credentials) instead of
        being followed blindly. Returns (bytes, content_type) or (None, None)."""
        headers = {}
        if auth_token:
            headers['Authorization'] = f"Bearer {auth_token}"
            logger.debug(f"Using auth token for {url}: Bearer {auth_token[:10]}...")
        session = self._get_session()
        current = url
        seen = set()
        for _ in range(_SSRF_MAX_REDIRECTS + 1):
            async with session.get(current, headers=headers, allow_redirects=False) as response:
                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get('Location') or response.headers.get('location')
                    if not location:
                        logger.error(f"Slack redirect with no Location for {current}")
                        return None, None
                    nxt = urljoin(current, location)
                    if nxt in seen:
                        logger.error(f"Redirect loop fetching {current}")
                        return None, None
                    seen.add(current)
                    if _is_trusted_host(nxt):
                        current = nxt
                        continue
                    # Left Slack. Do NOT send the auth token off-platform and do NOT follow
                    # blindly — hand the target to the guarded flow, which validates the host
                    # (SSRF) and sends no credentials. Covers the legitimate case of Slack
                    # redirecting to a public signed-CDN URL as well as an outright open redirect.
                    logger.info(
                        f"Slack URL redirected off-host to {nxt}; fetching under SSRF guards")
                    return await self._download_guarded(nxt)
                if response.status != 200:
                    logger.error(f"Failed to download image from {current}: Status {response.status}")
                    return None, None
                content_type = response.headers.get('content-type', '').lower()
                content = await self._read_capped(response)
                if content is None:
                    logger.error(
                        f"Image from {current} exceeded {self.max_image_size} bytes; aborted")
                    return None, None
                return content, content_type
        logger.warning(f"Too many redirects fetching {url}")
        return None, None

    async def _download_guarded(self, url: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch an UNTRUSTED image URL under full SSRF + size guards (mirrors ambient_fetch).

        Per hop: validate the URL shape, resolve every hostname and reject any non-global IP
        BEFORE connecting, pin aiohttp to exactly those IPs (no second, unvalidated resolution),
        follow redirects manually (each hop re-validated from scratch), and stream the body with
        a hard cap regardless of Content-Length. No auth is ever sent to a non-Slack host.
        Returns (bytes, content_type) or (None, None)."""
        opener = _guarded_opener or _default_guarded_opener
        loop = asyncio.get_running_loop()
        current = url
        seen = set()
        for _ in range(_SSRF_MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            host = parsed.hostname
            if (parsed.scheme.lower() not in ("http", "https") or not host
                    or parsed.username or parsed.password):
                logger.warning(f"Refusing malformed/unsafe image URL: {current}")
                return None, None
            try:
                # getaddrinfo blocks; run it OFF the event loop, bounded by the download timeout,
                # so a stalled resolver can't freeze the whole loop.
                validated = await asyncio.wait_for(
                    loop.run_in_executor(None, _resolve_and_validate, host, parsed.port),
                    timeout=self.timeout)
            except _SSRFBlocked as e:
                logger.warning(f"Blocked SSRF image fetch for {current}: {e}")
                return None, None
            except Exception as e:  # noqa: BLE001 — DNS timeout / unexpected resolver failure
                logger.warning(f"DNS validation failed for {current}: {e}")
                return None, None
            try:
                resp = await opener(current, validated, timeout=self.timeout,
                                    max_bytes=self.max_image_size)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Guarded fetch failed for {current}: {e}")
                return None, None
            try:
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location") or resp.headers.get("location")
                    if not location:
                        return None, None
                    nxt = urljoin(current, location)
                    if nxt in seen:
                        logger.warning(f"Redirect loop fetching {current}")
                        return None, None
                    seen.add(current)
                    current = nxt
                    continue
                if resp.status != 200:
                    logger.error(f"Failed to download image from {current}: status {resp.status}")
                    return None, None
                content_type = (resp.headers.get("Content-Type")
                                or resp.headers.get("content-type") or "").lower()
                clen = resp.headers.get("Content-Length") or resp.headers.get("content-length")
                if clen and str(clen).isdigit() and int(clen) > self.max_image_size:
                    logger.error(f"Image from {current} too large (content-length {clen})")
                    return None, None
                buf = bytearray()
                async for chunk in resp.iter_chunks(64 * 1024):
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if len(buf) > self.max_image_size:
                        logger.error(f"Image from {current} exceeded {self.max_image_size} bytes "
                                     "mid-stream; aborting the download")
                        return None, None
                return bytes(buf), content_type
            finally:
                try:
                    await resp.release()
                except Exception:  # noqa: BLE001
                    pass
        logger.warning(f"Too many redirects fetching {url}")
        return None, None

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
            # Check if this is a Slack file URL that needs auth. `_is_trusted_host` parses the
            # hostname (never a substring match) so an attacker URL like
            # https://evil.com/slack.com/files/ cannot smuggle the auth token off-platform, and
            # download_image itself re-checks trust so the token can never leak to an untrusted
            # host even if the flag here were wrong.
            is_slack_url = _is_trusted_host(url)
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