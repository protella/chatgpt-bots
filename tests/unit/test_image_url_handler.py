"""Unit tests for image_url_handler.py (Async Version)"""

import socket

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import base64
import image_url_handler
from image_url_handler import ImageURLHandler


@pytest.fixture(autouse=True)
def _reset_ssrf_seams():
    """The SSRF resolver/transport are module globals with test seams; restore them after each
    test so no real DNS or sockets leak between cases."""
    yield
    image_url_handler.set_resolver(image_url_handler._system_getaddrinfo)
    image_url_handler.set_guarded_opener(None)


def _resolver_map(mapping):
    """A getaddrinfo-shaped resolver: hostname -> IP string, or gaierror for anything else."""
    def _fn(host, port):
        ip = mapping.get(host)
        if ip is None:
            raise socket.gaierror(f"no address for {host}")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]
    return _fn


def _guarded_opener(responses):
    """A guarded-transport stub. `responses` is a list of dicts, one per hop:
    {status, headers, chunks}. Records the URLs it was asked to open on `.opened`."""
    seq = list(responses)
    opened = []

    async def _opener(url, ips, *, timeout, max_bytes):
        opened.append(url)
        spec = seq.pop(0)

        async def _iter(n):
            for c in spec.get("chunks", (b"",)):
                yield c

        async def _release():
            return None

        return image_url_handler._GuardedResponse(
            status=spec.get("status", 200), headers=spec.get("headers", {}),
            url=url, iter_chunks=_iter, release=_release)

    _opener.opened = opened
    return _opener


def _cm(resp):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


def _mk_resp(status=200, headers=None, chunks=(b"",)):
    async def _iter(n):
        for c in chunks:
            yield c

    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.content.iter_chunked = _iter
    return resp


def _seq_session(response_specs):
    """A shared-session mock whose GET returns a FRESH response per call, walking
    `response_specs` in order — the shape needed to exercise the trusted manual-redirect loop.
    Records each GET's url/headers/allow_redirects on `.get_calls`."""
    responses = [_mk_resp(**s) for s in response_specs]
    calls = []

    def _get(url, headers=None, allow_redirects=True):
        calls.append({"url": url, "headers": headers or {}, "allow_redirects": allow_redirects})
        return _cm(responses[len(calls) - 1])

    session = MagicMock()
    session.get = MagicMock(side_effect=_get)
    session.get_calls = calls
    return session


def _streaming_session(status=200, headers=None, chunks=(b"",)):
    """A shared-session mock (the TRUSTED/Slack path) whose GET streams via content.iter_chunked
    and whose HEAD returns the same status/headers."""
    async def _iter(n):
        for c in chunks:
            yield c

    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    resp.content.iter_chunked = _iter
    resp.text = AsyncMock(return_value="")

    session = MagicMock()
    session.get.return_value.__aenter__.return_value = resp
    session.get.return_value.__aexit__.return_value = None
    session.head.return_value.__aenter__.return_value = resp
    session.head.return_value.__aexit__.return_value = None
    return session


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
    async def test_validate_image_url_accepts_wellformed(self, handler):
        """validate is now a no-network shape check for EVERY host — a well-formed URL passes,
        and the guarded download decides for real. No mimetype is reported (nothing was fetched)."""
        assert await handler.validate_image_url("https://files.slack.com/image.jpg") == \
            (True, None, None)
        assert await handler.validate_image_url("https://example.com/image.jpg") == \
            (True, None, None)

    @pytest.mark.asyncio
    async def test_validate_image_url_rejects_malformed(self, handler):
        is_valid, mimetype, error = await handler.validate_image_url("not_a_url")
        assert is_valid is False and mimetype is None and error is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url", [
        "https://files.slack.com/image.jpg",   # trusted — the residual gap: no AUTHED HEAD
        "https://example.com/image.jpg",       # untrusted
    ])
    async def test_validate_does_no_network_for_any_host(self, handler, url):
        """The residual T1-4 gap: validate used to run an AUTHENTICATED, auto-redirecting HEAD for
        trusted URLs, which could leak the auth header to an off-Slack redirect during validation.
        It must now touch the network for NO host — the (fully guarded) download is the only gate."""
        def _boom():
            raise AssertionError("validate_image_url must not touch the network")

        with patch.object(handler, '_get_session', side_effect=_boom):
            result = await handler.validate_image_url(url, auth_token="xoxb-secret")
        assert result == (True, None, None)

    @pytest.mark.asyncio
    async def test_download_image_success_trusted(self, handler):
        """A genuinely-decodable PNG downloaded over the trusted (Slack) streaming path."""
        from io import BytesIO

        from PIL import Image
        _b = BytesIO()
        Image.new("RGB", (2, 2), "red").save(_b, format="PNG")
        image_data = _b.getvalue()

        mock_session = _streaming_session(
            status=200, headers={'content-type': 'image/png'}, chunks=(image_data,))

        with patch.object(handler, '_get_session', return_value=mock_session):
            result = await handler.download_image(
                "https://files.slack.com/image.png", mimetype='image/png', auth_token="xoxb-x")

            assert result is not None
            assert result['url'] == "https://files.slack.com/image.png"
            assert result['mimetype'] == 'image/png'
            assert result['size'] == len(image_data)
            assert result['data'] == image_data
            assert result['base64_data'] == base64.b64encode(image_data).decode('utf-8')

    @pytest.mark.asyncio
    async def test_download_rejects_when_transcode_exceeds_max_size(self, handler, monkeypatch):
        """Finding 3: the streaming cap bounds the DOWNLOAD, not the RESULT. A small source that
        re-encodes to a PNG over the limit must be rejected on the POST-transcode bytes, never
        base64'd and sent unchecked."""
        handler.max_image_size = 100
        small_source = b"\x00" * 50                       # under the cap: clears the stream cap
        big_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 500     # what the transcode "produced": over it
        monkeypatch.setattr(image_url_handler, "ensure_api_compatible",
                            lambda raw: (big_png, "image/png"))

        mock_session = _streaming_session(
            status=200, headers={'content-type': 'image/bmp'}, chunks=(small_source,))
        with patch.object(handler, '_get_session', return_value=mock_session):
            result = await handler.download_image(
                "https://files.slack.com/x.bmp", mimetype='image/bmp', auth_token="xoxb-x")

        assert result is None

    @pytest.mark.asyncio
    async def test_download_accepts_transcode_within_max_size(self, handler, monkeypatch):
        """The companion: a transcoded result UNDER the ceiling passes, carrying the post-transcode
        bytes and their new mimetype."""
        handler.max_image_size = 1000
        small_source = b"\x00" * 50
        small_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        monkeypatch.setattr(image_url_handler, "ensure_api_compatible",
                            lambda raw: (small_png, "image/png"))

        mock_session = _streaming_session(
            status=200, headers={'content-type': 'image/bmp'}, chunks=(small_source,))
        with patch.object(handler, '_get_session', return_value=mock_session):
            result = await handler.download_image(
                "https://files.slack.com/x.bmp", mimetype='image/bmp', auth_token="xoxb-x")

        assert result is not None
        assert result['mimetype'] == 'image/png'
        assert result['data'] == small_png
        assert result['size'] == len(small_png)

    # ----------------------------------------------------------------- F4: SSRF + streaming cap

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_ip", [
        "127.0.0.1",            # loopback
        "10.0.0.5",             # RFC1918 private
        "169.254.169.254",      # cloud metadata (link-local)
        "192.168.1.1",          # private
        "::1",                  # IPv6 loopback
    ])
    async def test_download_untrusted_blocks_non_global_ip(self, handler, bad_ip):
        image_url_handler.set_resolver(_resolver_map({"attacker.test": bad_ip}))
        opener = _guarded_opener([{"status": 200, "chunks": (b"\x89PNG\r\n\x1a\n",)}])
        image_url_handler.set_guarded_opener(opener)

        result = await handler.download_image("http://attacker.test/x.png")

        assert result is None
        assert opener.opened == []          # blocked BEFORE any connection was opened

    @pytest.mark.asyncio
    async def test_download_untrusted_blocks_ipv4_mapped_metadata(self, handler):
        # ::ffff:169.254.169.254 reports is_global against the v6 space; it must be unwrapped.
        image_url_handler.set_resolver(_resolver_map({"sneaky.test": "::ffff:169.254.169.254"}))
        opener = _guarded_opener([{"status": 200, "chunks": (b"x",)}])
        image_url_handler.set_guarded_opener(opener)

        assert await handler.download_image("http://sneaky.test/x.png") is None
        assert opener.opened == []

    @pytest.mark.asyncio
    async def test_download_untrusted_blocks_redirect_to_private(self, handler):
        # The first hop is a public host that 302s to an internal one. The redirect target must be
        # re-validated and refused — the classic SSRF-via-redirect bypass.
        image_url_handler.set_resolver(_resolver_map({
            "cdn.public.test": "93.184.216.34", "internal.evil": "127.0.0.1"}))
        opener = _guarded_opener([
            {"status": 302, "headers": {"Location": "http://internal.evil/secret"}},
        ])
        image_url_handler.set_guarded_opener(opener)

        result = await handler.download_image("http://cdn.public.test/pic.png")

        assert result is None
        assert opener.opened == ["http://cdn.public.test/pic.png"]   # never connected to internal

    @pytest.mark.asyncio
    async def test_download_untrusted_streaming_cap_without_content_length(self, handler):
        # No Content-Length header: the ONLY defense is the streaming cap. A body past the ceiling
        # must abort mid-stream rather than buffer unbounded.
        handler.max_image_size = 100
        image_url_handler.set_resolver(_resolver_map({"cdn.public.test": "93.184.216.34"}))
        opener = _guarded_opener([{"status": 200, "chunks": (b"\x00" * 80, b"\x00" * 80)}])
        image_url_handler.set_guarded_opener(opener)

        assert await handler.download_image("http://cdn.public.test/big.png") is None

    @pytest.mark.asyncio
    async def test_download_untrusted_content_length_over_cap(self, handler):
        handler.max_image_size = 100
        image_url_handler.set_resolver(_resolver_map({"cdn.public.test": "93.184.216.34"}))
        opener = _guarded_opener([
            {"status": 200, "headers": {"Content-Length": "999999"}, "chunks": (b"\x00",)}])
        image_url_handler.set_guarded_opener(opener)

        assert await handler.download_image("http://cdn.public.test/big.png") is None

    @pytest.mark.asyncio
    async def test_download_untrusted_success_public_ip(self, handler):
        from io import BytesIO

        from PIL import Image
        _b = BytesIO()
        Image.new("RGB", (2, 2), "blue").save(_b, format="PNG")
        png = _b.getvalue()

        image_url_handler.set_resolver(_resolver_map({"cdn.public.test": "93.184.216.34"}))
        image_url_handler.set_guarded_opener(_guarded_opener([
            {"status": 200, "headers": {"Content-Type": "image/png"}, "chunks": (png,)}]))

        result = await handler.download_image("http://cdn.public.test/pic.png")

        assert result is not None
        assert result['mimetype'] == 'image/png'
        assert result['data'] == png

    # ----------------------------------------------------------- T1-4: trusted-path redirects

    @pytest.mark.asyncio
    async def test_trusted_slack_redirect_off_host_is_ssrf_validated_not_followed_with_auth(self, handler):
        # A Slack link that 302s OFF Slack to a host resolving to a private IP. It must not be
        # followed with the auth token; it drops to the guarded flow, which blocks the private
        # target before ever connecting. This is the open-redirect SSRF the manual loop closes.
        image_url_handler.set_resolver(_resolver_map({"internal.evil": "127.0.0.1"}))
        guard_opener = _guarded_opener([{"status": 200, "chunks": (b"x",)}])
        image_url_handler.set_guarded_opener(guard_opener)

        session = _seq_session([
            {"status": 302, "headers": {"Location": "https://internal.evil/secret.png"}},
        ])
        with patch.object(handler, '_get_session', return_value=session):
            result = await handler.download_image(
                "https://files.slack.com/pic.png", auth_token="xoxb-secret")

        assert result is None
        assert guard_opener.opened == []                 # blocked before connecting (private IP)
        # The auth token was sent ONLY to the Slack host, never to the redirect target, and no
        # second authed GET was issued to the evil host.
        assert len(session.get_calls) == 1
        assert session.get_calls[0]["headers"].get("Authorization") == "Bearer xoxb-secret"
        assert session.get_calls[0]["allow_redirects"] is False   # never auto-followed

    @pytest.mark.asyncio
    async def test_trusted_slack_redirect_to_a_slack_host_is_followed_with_auth(self, handler):
        from io import BytesIO

        from PIL import Image
        _b = BytesIO()
        Image.new("RGB", (2, 2), "green").save(_b, format="PNG")
        png = _b.getvalue()

        session = _seq_session([
            {"status": 302, "headers": {"Location": "https://files-edge.slack.com/real.png"}},
            {"status": 200, "headers": {"content-type": "image/png"}, "chunks": (png,)},
        ])
        with patch.object(handler, '_get_session', return_value=session):
            result = await handler.download_image(
                "https://files.slack.com/pic.png", auth_token="xoxb-secret")

        assert result is not None and result["mimetype"] == "image/png"
        # Both hops stayed on Slack, so both carried the auth token.
        assert len(session.get_calls) == 2
        assert session.get_calls[1]["url"] == "https://files-edge.slack.com/real.png"
        assert all(c["headers"].get("Authorization") == "Bearer xoxb-secret"
                   for c in session.get_calls)

    @pytest.mark.asyncio
    async def test_trusted_slack_redirect_off_host_to_public_uses_guarded_flow(self, handler):
        # The legitimate case: Slack 302s to a public signed-CDN URL. It succeeds — but through
        # the guarded flow (SSRF-validated, NO auth), not the authed session.
        from io import BytesIO

        from PIL import Image
        _b = BytesIO()
        Image.new("RGB", (2, 2), "red").save(_b, format="PNG")
        png = _b.getvalue()

        image_url_handler.set_resolver(_resolver_map({"cdn.public.test": "93.184.216.34"}))
        image_url_handler.set_guarded_opener(_guarded_opener([
            {"status": 200, "headers": {"Content-Type": "image/png"}, "chunks": (png,)}]))

        session = _seq_session([
            {"status": 302, "headers": {"Location": "https://cdn.public.test/real.png"}},
        ])
        with patch.object(handler, '_get_session', return_value=session):
            result = await handler.download_image(
                "https://files.slack.com/pic.png", auth_token="xoxb-secret")

        assert result is not None and result["data"] == png
        assert len(session.get_calls) == 1               # authed session stopped at the Slack hop

    def test_is_trusted_host_classification(self):
        assert image_url_handler._is_trusted_host("https://files.slack.com/x.png")
        assert image_url_handler._is_trusted_host("https://myteam.slack.com/x")
        assert image_url_handler._is_trusted_host("https://x.slack-files.com/y")
        # A substring or path smuggle must NOT be trusted (no auth token leaks off-platform).
        assert not image_url_handler._is_trusted_host("https://evil.com/slack.com/x")
        assert not image_url_handler._is_trusted_host("https://notslack.com/x")
        assert not image_url_handler._is_trusted_host("https://example.com/x.png")

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