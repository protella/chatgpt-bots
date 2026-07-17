"""F51 — SSRF boundary tests for ambient_fetch. These define the security contract; they mock
at the resolver and transport layers so NO real network or DNS is ever touched."""
import asyncio
import socket

import pytest

import ambient_fetch
from ambient_fetch import (
    ERR_BAD_URL,
    ERR_BLOCKED_SSRF,
    ERR_HTTP_STATUS,
    ERR_REDIRECT_LIMIT,
    ERR_TIMEOUT,
    ERR_TOO_LARGE,
    ERR_UNSUPPORTED_TYPE,
    SSRFError,
    _RawResponse,
    fetch_url,
    resolve_and_validate,
    validate_url,
)

pytestmark = pytest.mark.unit

FETCH_KW = dict(max_bytes=1_000_000, connect_timeout=1, read_timeout=1,
                total_timeout=2, max_redirects=5, max_chars=8000)


# --------------------------------------------------------------- URL structure validation

@pytest.mark.parametrize("url", [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "gopher://example.com",
    "javascript:alert(1)",
    "",
    "http://",
    "https://" + "a" * 5000 + ".com",
])
def test_validate_url_rejects_bad_shapes(url):
    with pytest.raises(SSRFError):
        validate_url(url)


def test_validate_url_rejects_userinfo():
    with pytest.raises(SSRFError) as ei:
        validate_url("http://user:pass@example.com/")
    assert ei.value.error_code == ERR_BLOCKED_SSRF


@pytest.mark.parametrize("host", [
    "files.slack.com", "myteam.slack.com", "cdn.slack-edge.com",
    "x.slack-files.com", "slack.com",
])
def test_validate_url_blocks_slack_hosts(host):
    with pytest.raises(SSRFError) as ei:
        validate_url(f"https://{host}/whatever")
    assert ei.value.error_code == ERR_BLOCKED_SSRF


def test_validate_url_accepts_plain_https():
    assert validate_url("https://example.com/a?b=c") == "https://example.com/a?b=c"


@pytest.mark.parametrize("host", [
    "slack。com",       # U+3002 ideographic full stop
    "files.slack．com",  # U+FF0E fullwidth full stop
    "slack｡com",       # U+FF61 halfwidth ideographic full stop
])
def test_validate_url_blocks_idn_dot_homoglyph_slack(host):
    # The HTTP stack IDNA-normalizes these dot variants back to a real ".", reaching slack.com —
    # so the host blocklist must normalize the same way or it is trivially bypassed.
    with pytest.raises(SSRFError) as ei:
        validate_url(f"https://{host}/whatever")
    assert ei.value.error_code == ERR_BLOCKED_SSRF


# --------------------------------------------------------------- DNS resolution validation

def _resolver_returning(*ips):
    def _r(host, port):
        infos = []
        for ip in ips:
            fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
            infos.append((fam, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0)))
        return infos
    return _r


@pytest.fixture(autouse=True)
def _restore_seams():
    yield
    ambient_fetch.set_resolver(
        lambda host, port: socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP))
    ambient_fetch.set_opener(None)


@pytest.mark.parametrize("ip", [
    "127.0.0.1",        # loopback
    "10.0.0.5",         # private
    "192.168.1.1",      # private
    "172.16.9.9",       # private
    "169.254.169.254",  # link-local (cloud metadata!)
    "0.0.0.0",          # unspecified
    "224.0.0.1",        # multicast
    "::1",              # ipv6 loopback
    "fe80::1",          # ipv6 link-local
    "fc00::1",          # ipv6 unique-local
    "::ffff:169.254.169.254",  # IPv4-mapped IPv6 metadata
])
def test_resolve_rejects_non_global(ip):
    ambient_fetch.set_resolver(_resolver_returning(ip))
    with pytest.raises(SSRFError) as ei:
        resolve_and_validate("evil.example")
    assert ei.value.error_code == ERR_BLOCKED_SSRF


def test_resolve_accepts_global():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    assert resolve_and_validate("example.com") == ["93.184.216.34"]


def test_resolve_rejects_mixed_rebinding_response():
    # A DNS-rebinding response mixing a public and a private IP must be rejected wholesale —
    # never let the private one through by connection ordering.
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34", "127.0.0.1"))
    with pytest.raises(SSRFError) as ei:
        resolve_and_validate("rebind.example")
    assert ei.value.error_code == ERR_BLOCKED_SSRF


def test_resolve_literal_private_ip_rejected():
    with pytest.raises(SSRFError):
        resolve_and_validate("10.1.2.3")


def test_resolve_literal_public_ip_ok():
    assert resolve_and_validate("93.184.216.34") == ["93.184.216.34"]


# --------------------------------------------------------------- transport / fetch behaviour

def _opener_for(responses):
    """responses: list of dicts describing successive hops:
    {status, headers, chunks} — one per opener call."""
    calls = {"i": 0}

    async def _opener(url, validated_ips, **kw):
        idx = calls["i"]
        calls["i"] += 1
        spec = responses[min(idx, len(responses) - 1)]

        async def _iter(chunk_size):
            for c in spec.get("chunks", []):
                yield c

        async def _release():
            return None

        return _RawResponse(status=spec["status"], headers=spec.get("headers", {}),
                            url=url, iter_chunks=_iter, release=_release)

    return _opener, calls


@pytest.mark.asyncio
async def test_fetch_follows_redirect_and_revalidates_each_hop():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    opener, calls = _opener_for([
        {"status": 302, "headers": {"Location": "https://example.com/final"}},
        {"status": 200, "headers": {"Content-Type": "text/html"},
         "chunks": [b"<title>Hi</title><p>Body text here</p>"]},
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/start", **FETCH_KW)
    assert res.kind == "text"
    assert res.title == "Hi"
    assert "Body text here" in res.text
    assert calls["i"] == 2


@pytest.mark.asyncio
async def test_fetch_redirect_to_private_host_blocked():
    # First hop resolves global; the redirect target resolves to a private IP → blocked.
    def _r(host, port):
        ip = "127.0.0.1" if host == "internal.evil" else "93.184.216.34"
        fam = socket.AF_INET
        return [(fam, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0))]
    ambient_fetch.set_resolver(_r)
    opener, _ = _opener_for([
        {"status": 302, "headers": {"Location": "http://internal.evil/admin"}},
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://public.example/go", **FETCH_KW)
    assert res.kind == "error"
    assert res.error_code == ERR_BLOCKED_SSRF


@pytest.mark.asyncio
async def test_fetch_redirect_limit():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    # Always redirect to a NEW url so the loop-detector doesn't short-circuit first.
    calls = {"i": 0}

    async def _opener(url, validated_ips, **kw):
        calls["i"] += 1
        n = calls["i"]

        async def _iter(chunk_size):
            if False:
                yield b""

        async def _release():
            return None

        return _RawResponse(status=302, headers={"Location": f"https://example.com/h{n}"},
                            url=url, iter_chunks=_iter, release=_release)

    ambient_fetch.set_opener(_opener)
    res = await fetch_url("https://example.com/start", **{**FETCH_KW, "max_redirects": 3})
    assert res.error_code == ERR_REDIRECT_LIMIT


@pytest.mark.asyncio
async def test_fetch_oversized_stream_capped():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    opener, _ = _opener_for([
        {"status": 200, "headers": {"Content-Type": "text/plain"},
         "chunks": [b"x" * 600, b"y" * 600]},  # 1200 bytes over a 1000 cap
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/big", **{**FETCH_KW, "max_bytes": 1000})
    assert res.error_code == ERR_TOO_LARGE


@pytest.mark.asyncio
async def test_fetch_content_length_precheck():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    opener, _ = _opener_for([
        {"status": 200, "headers": {"Content-Type": "text/plain", "Content-Length": "9999"}},
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/big", **{**FETCH_KW, "max_bytes": 1000})
    assert res.error_code == ERR_TOO_LARGE


@pytest.mark.asyncio
async def test_fetch_dns_stall_times_out_without_freezing_loop():
    # A hostname whose resolution stalls must NOT block the event loop: resolution runs off-loop
    # under dns_timeout. While it stalls, the loop stays responsive (a concurrent task advances),
    # and the fetch returns ERR_TIMEOUT rather than hanging for the OS resolver's full duration.
    import time as _time

    def _slow_resolver(host, port):
        _time.sleep(2.0)  # blocking, like the real getaddrinfo
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 0))]

    ambient_fetch.set_resolver(_slow_resolver)
    ambient_fetch.set_opener(lambda *a, **k: (_ for _ in ()).throw(AssertionError("opener reached")))

    ticks = 0

    async def _heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    hb = asyncio.ensure_future(_heartbeat())
    res = await fetch_url("https://slow.example/x", **{**FETCH_KW, "dns_timeout": 0.2})
    await hb
    assert res.error_code == ERR_TIMEOUT
    assert ticks > 0  # the loop kept running while DNS was stalled off-loop


@pytest.mark.asyncio
async def test_fetch_direct_image_routes_to_vision():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    # A genuinely-decodable PNG — the downstream vision worker now PARSES image bytes
    # (validate_image_bytes), so a signature-plus-junk stub would be a dishonest fixture that
    # never survives the real path.
    from io import BytesIO

    from PIL import Image
    _b = BytesIO()
    Image.new("RGB", (2, 2), "red").save(_b, format="PNG")
    png = _b.getvalue()
    opener, _ = _opener_for([
        {"status": 200, "headers": {"Content-Type": "image/png"}, "chunks": [png]},
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/pic", **FETCH_KW)
    assert res.kind == "image"
    assert res.content_type == "image/png"
    assert res.raw_bytes == png


@pytest.mark.asyncio
async def test_fetch_unsupported_type():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    opener, _ = _opener_for([
        {"status": 200, "headers": {"Content-Type": "application/octet-stream"},
         "chunks": [b"\x00\x01\x02binary"]},
    ])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/bin", **FETCH_KW)
    assert res.error_code == ERR_UNSUPPORTED_TYPE


@pytest.mark.asyncio
async def test_fetch_http_error_status():
    ambient_fetch.set_resolver(_resolver_returning("93.184.216.34"))
    opener, _ = _opener_for([{"status": 404, "headers": {}}])
    ambient_fetch.set_opener(opener)
    res = await fetch_url("https://example.com/missing", **FETCH_KW)
    assert res.error_code == ERR_HTTP_STATUS


@pytest.mark.asyncio
async def test_fetch_bad_url_never_calls_opener():
    called = {"n": 0}

    async def _opener(*a, **k):
        called["n"] += 1
        raise AssertionError("opener must not run for a bad url")

    ambient_fetch.set_opener(_opener)
    res = await fetch_url("ftp://example.com/x", **FETCH_KW)
    assert res.error_code == ERR_BAD_URL
    assert called["n"] == 0


def test_html_to_text_strips_scripts_and_bounds():
    html = "<title>T</title><script>evil()</script><style>x{}</style><p>Hello &amp; bye</p>"
    text, title = ambient_fetch.html_to_text(html, max_chars=1000)
    assert title == "T"
    assert "evil()" not in text
    assert "Hello & bye" in text
