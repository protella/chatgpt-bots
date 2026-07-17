"""F51 — SSRF-hardened link fetcher.

One fetcher, two entry points: the ambient background worker (a link posted in a channel gets
opened and summarized) and the model-callable `fetch_url` tool (a directly-asked "read this
link" actually opens it). The existing image/Slack downloaders follow redirects automatically
and buffer whole bodies before capping — they are NOT a security boundary and are never reused
here.

The security decisions (URL shape, per-hop DNS validation, non-global IP rejection, byte caps)
are pure functions with no network IO so they can be tested at the resolver/transport layer with
no real sockets. The only IO is `_default_opener`, injected via `set_opener()` in tests.

Threat model handled:
- SSRF to internal services: every hostname is resolved and EVERY resolved IP validated as
  globally routable BEFORE connecting; a name resolving to any loopback/private/link-local/
  reserved/multicast/unspecified/IPv4-mapped address is rejected wholesale.
- DNS rebinding: we resolve once, validate, then connect THROUGH exactly those validated IPs
  (a pinned resolver) — aiohttp never performs a second, unvalidated resolution.
- Redirect-based bypass: redirects are followed MANUALLY (aiohttp auto-redirect off), each hop's
  URL re-validated from scratch, capped at a small maximum.
- Credential leak: Slack hosts are blocked outright, and no auth headers/cookies are ever sent.
- Resource exhaustion: Content-Length is checked AND the decoded stream is capped at
  max_bytes+1; separate connect/read/total timeouts bound latency.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

from logger import setup_logger

logger = setup_logger(name="slack_bot.AmbientFetch")

# error_code taxonomy — every failure persists one of these, never a silent drop.
ERR_BLOCKED_SSRF = "blocked_ssrf"
ERR_REDIRECT_LIMIT = "redirect_limit"
ERR_TIMEOUT = "timeout"
ERR_TOO_LARGE = "too_large"
ERR_UNSUPPORTED_TYPE = "unsupported_type"
ERR_HTTP_STATUS = "http_status"
ERR_DECODE_FAILED = "decode_failed"
ERR_EXTRACT_FAILED = "extract_failed"
ERR_BAD_URL = "bad_url"

# MIME allowlist for link fetches. Images do NOT belong here — a URL whose bytes sniff as an
# image is routed to the vision worker (result.kind == "image"), never rejected.
_TEXTUAL_MIMES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
    "application/pdf",
)

# Hosts whose content we must never fetch as an external link (credential/loopback surface).
_BLOCKED_HOST_SUFFIXES = (
    "slack.com",
    "slack-edge.com",
    "slack-files.com",
    "slackb.com",
    "slack.dev",
    "amazonaws.com.slack",  # defensive
)

_MAX_URL_LEN = 2048

# Magic-byte sniff for the "direct image URL" case (route to vision, don't reject).
_IMAGE_MAGIC: Tuple[Tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # RIFF....WEBP — refined below
)


class SSRFError(Exception):
    """Raised by the URL/DNS validation layer; carries an error_code from the taxonomy."""

    def __init__(self, error_code: str, message: str = ""):
        super().__init__(message or error_code)
        self.error_code = error_code


@dataclass
class FetchResult:
    """Outcome of a fetch. `kind`: 'text' (extracted textual content), 'image' (bytes sniffed as
    an image — route to the vision worker), or 'error' (see error_code)."""
    kind: str
    final_url: Optional[str] = None
    content_type: Optional[str] = None
    title: Optional[str] = None
    text: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    error_code: Optional[str] = None
    error_detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.kind in ("text", "image")


# --------------------------------------------------------------------------- URL validation

def validate_url(url: str) -> str:
    """Return a canonical http(s) URL or raise SSRFError(ERR_BAD_URL / ERR_BLOCKED_SSRF).

    Rejects: non-http(s) schemes, embedded credentials (userinfo), missing host, over-long URLs,
    and blocked (Slack) hosts. Does NOT resolve DNS — that is resolve_and_validate()."""
    if not url or not isinstance(url, str):
        raise SSRFError(ERR_BAD_URL, "empty url")
    url = url.strip()
    if len(url) > _MAX_URL_LEN:
        raise SSRFError(ERR_BAD_URL, "url too long")
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise SSRFError(ERR_BAD_URL, f"unparseable url: {e}")
    if parts.scheme.lower() not in ("http", "https"):
        raise SSRFError(ERR_BAD_URL, f"scheme not allowed: {parts.scheme!r}")
    if parts.username or parts.password or "@" in (parts.netloc or ""):
        raise SSRFError(ERR_BLOCKED_SSRF, "userinfo in url")
    host = parts.hostname
    if not host:
        raise SSRFError(ERR_BAD_URL, "no host")
    if _is_blocked_host(host):
        raise SSRFError(ERR_BLOCKED_SSRF, f"blocked host: {host}")
    return url


def _normalize_host(host: str) -> str:
    """Fold a hostname to the ASCII form the HTTP stack will actually connect to, so a suffix
    check can't be bypassed by a non-ASCII label. aiohttp IDNA-encodes the host before dialing,
    which maps the Unicode dot variants (U+3002 `。`, U+FF0E `．`, U+FF61 `｡`) to ".", so
    `https://slack。com/` reaches slack.com — normalize the same way before the blocklist runs."""
    import unicodedata
    h = unicodedata.normalize("NFKC", host or "")
    for dot in ("。", "．", "｡"):
        h = h.replace(dot, ".")
    h = h.lower().rstrip(".")
    # Best-effort IDNA round-trip for any remaining homoglyph labels; keep the folded form on
    # failure (empty labels / invalid punycode are handled by the caller's own validation).
    try:
        h = h.encode("idna").decode("ascii").lower()
    except Exception:  # noqa: BLE001
        pass
    return h


def _is_blocked_host(host: str) -> bool:
    h = _normalize_host(host)
    return any(h == suf or h.endswith("." + suf) for suf in _BLOCKED_HOST_SUFFIXES)


def _ip_is_global(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """A globally routable public address — everything else (loopback, private, link-local,
    reserved, multicast, unspecified, IPv4-mapped/6to4/teredo IPv6) is refused."""
    # ipaddress.is_global is the base signal, but it does NOT catch an IPv4 address smuggled
    # inside IPv6 (::ffff:169.254.169.254 reports is_global against the v6 space). Unwrap first.
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return _ip_is_global(ip.ipv4_mapped)
        if ip.sixtofour is not None:
            return _ip_is_global(ip.sixtofour)
        teredo = ip.teredo
        if teredo is not None:
            # teredo -> (server, client); the client side is the tunneled address.
            return _ip_is_global(teredo[1])
    if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
            or ip.is_reserved or ip.is_unspecified):
        return False
    return bool(getattr(ip, "is_global", True))


# Injectable resolver so tests validate the boundary with no real DNS. Signature mirrors
# socket.getaddrinfo's return (family, type, proto, canonname, sockaddr).
def _system_getaddrinfo(host: str, port: Optional[int]) -> List[Tuple]:
    return socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP)


_getaddrinfo: Callable[[str, Optional[int]], List[Tuple]] = _system_getaddrinfo


def set_resolver(fn: Callable[[str, Optional[int]], List[Tuple]]) -> None:
    """Test seam: replace the getaddrinfo-shaped resolver."""
    global _getaddrinfo
    _getaddrinfo = fn


def resolve_and_validate(host: str, port: Optional[int] = None) -> List[str]:
    """Resolve `host` and return its validated globally-routable IPs, or raise SSRFError.

    STRICT: if the name resolves to ANY non-global address, the whole name is rejected — a
    rebinding response mixing one public and one private IP must not sneak the private one
    through by ordering. A literal IP host is validated directly (no DNS)."""
    # Literal IP? Validate directly.
    try:
        literal = ipaddress.ip_address(host)
        if not _ip_is_global(literal):
            raise SSRFError(ERR_BLOCKED_SSRF, f"non-global literal ip: {host}")
        return [str(literal)]
    except ValueError:
        pass  # a hostname, resolve it
    try:
        infos = _getaddrinfo(host, port)
    except SSRFError:
        raise
    except Exception as e:  # noqa: BLE001 — DNS failure is a hard stop
        raise SSRFError(ERR_BLOCKED_SSRF, f"dns resolution failed for {host}: {e}")
    ips: List[str] = []
    for info in infos or []:
        sockaddr = info[4] if len(info) >= 5 else None
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str.split("%", 1)[0])  # strip zone id
        except ValueError:
            raise SSRFError(ERR_BLOCKED_SSRF, f"unparseable resolved address: {ip_str!r}")
        if not _ip_is_global(ip):
            raise SSRFError(ERR_BLOCKED_SSRF, f"{host} resolves to non-global {ip_str}")
        ips.append(str(ip))
    if not ips:
        raise SSRFError(ERR_BLOCKED_SSRF, f"no addresses for {host}")
    # De-dup, preserve order.
    return list(dict.fromkeys(ips))


# --------------------------------------------------------------------------- HTTP IO (injectable)

@dataclass
class _RawResponse:
    """Minimal response shape the fetcher consumes — lets the opener be mocked in tests."""
    status: int
    headers: dict
    url: str
    # async generator yielding decoded/raw byte chunks
    iter_chunks: Callable[[int], Any]
    release: Callable[[], Awaitable[None]]


# Opener: (url, validated_ips, timeouts) -> awaitable _RawResponse. Redirects are NOT followed
# by the opener (allow_redirects is off); this fetcher follows them manually.
Opener = Callable[..., Awaitable["_RawResponse"]]

_opener: Optional[Opener] = None


def set_opener(fn: Optional[Opener]) -> None:
    """Test seam: replace the transport opener (default uses aiohttp)."""
    global _opener
    _opener = fn


class _PinnedResolver:
    """aiohttp AbstractResolver returning ONLY the pre-validated IPs — no second resolution,
    closing the DNS-rebinding window between our validation and aiohttp's connect."""

    def __init__(self, ips: List[str]):
        self._ips = ips

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        out = []
        for ip in self._ips:
            try:
                fam = socket.AF_INET6 if ipaddress.ip_address(ip).version == 6 else socket.AF_INET
            except ValueError:
                continue
            out.append({
                "hostname": host, "host": ip, "port": port,
                "family": fam, "proto": socket.IPPROTO_TCP, "flags": 0,
            })
        if not out:
            raise OSError(f"no validated addresses for {host}")
        return out

    async def close(self):
        return None


async def _default_opener(url: str, validated_ips: List[str], *, connect_timeout: float,
                          read_timeout: float, total_timeout: float,
                          max_bytes: int) -> _RawResponse:
    import aiohttp

    timeout = aiohttp.ClientTimeout(
        total=total_timeout, connect=connect_timeout, sock_read=read_timeout)
    connector = aiohttp.TCPConnector(
        resolver=_PinnedResolver(validated_ips), ttl_dns_cache=0,  # type: ignore[arg-type]
        force_close=True, enable_cleanup_closed=True)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False)
    try:
        resp = await session.get(url, allow_redirects=False, headers={
            "User-Agent": "chatgpt-slackbot-linkfetch/1.0",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json,application/pdf;q=0.9,*/*;q=0.5",
        })
    except BaseException:
        # session.get can raise before a response exists (connect refused, timeout, cancel);
        # the session owns the connector, so it must be closed or the connector leaks. Reraise
        # after cleanup so the caller still sees the real failure.
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

    return _RawResponse(status=resp.status, headers=dict(resp.headers),
                        url=str(resp.url), iter_chunks=_iter, release=_release)


# --------------------------------------------------------------------------- extraction

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'",
    "&apos;": "'", "&nbsp;": " ",
}


def _unescape(s: str) -> str:
    for k, v in _HTML_ENTITIES.items():
        s = s.replace(k, v)
    return s


def html_to_text(html: str, *, max_chars: int) -> Tuple[str, Optional[str]]:
    """Strip HTML to bounded plain text and pull the <title>. Dependency-free and defensive —
    fetched HTML is untrusted, so we never execute or trust it, just flatten it."""
    title = None
    m = _TITLE_RE.search(html)
    if m:
        title = _unescape(_TAG_RE.sub("", m.group(1))).strip()[:200] or None
    body = _SCRIPT_STYLE_RE.sub(" ", html)
    body = _TAG_RE.sub(" ", body)
    body = _unescape(body)
    body = _WS_RE.sub(" ", body)
    body = _MULTI_NL_RE.sub("\n\n", body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = _MULTI_NL_RE.sub("\n\n", body).strip()
    return body[:max_chars], title


def _sniff_image(prefix: bytes) -> Optional[str]:
    for magic, mime in _IMAGE_MAGIC:
        if prefix.startswith(magic):
            if magic == b"RIFF":
                if prefix[8:12] == b"WEBP":
                    return "image/webp"
                continue
            return mime
    return None


def _content_type_family(content_type: Optional[str]) -> str:
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    return ct


# --------------------------------------------------------------------------- fetch

async def fetch_url(
    url: str, *, max_bytes: int, connect_timeout: float, read_timeout: float,
    total_timeout: float, max_redirects: int, max_chars: int,
    dns_timeout: Optional[float] = None,
) -> FetchResult:
    """Fetch a URL under all SSRF/size/timeout guards. Never raises for expected failures —
    returns a FetchResult with kind='error' and an error_code from the taxonomy.

    DNS resolution runs OFF the event loop (executor) under its own `dns_timeout`: the system
    resolver is synchronous and blocking, so a hostname whose resolution stalls would otherwise
    freeze every coroutine on the loop for the OS resolver's duration — outside all the aiohttp
    timeouts, which only start after connect. Defaults to `connect_timeout` when unset."""
    import asyncio

    dns_timeout = float(connect_timeout if dns_timeout is None else dns_timeout)
    loop = asyncio.get_running_loop()

    async def _resolve(host: str, port: Optional[int]) -> List[str]:
        # resolve_and_validate is CPU-cheap but does a blocking getaddrinfo; run it in a worker
        # thread bounded by dns_timeout. It raises SSRFError (propagated) for a non-global name.
        return await asyncio.wait_for(
            loop.run_in_executor(None, resolve_and_validate, host, port),
            timeout=dns_timeout)

    try:
        current = validate_url(url)
    except SSRFError as e:
        return FetchResult(kind="error", error_code=e.error_code, error_detail=str(e), final_url=url)

    opener = _opener or _default_opener
    seen = set()
    try:
        for hop in range(max_redirects + 1):
            parts = urlsplit(current)
            host = parts.hostname
            if not host:
                return FetchResult(kind="error", error_code=ERR_BAD_URL, final_url=current)
            # Per-hop DNS validation — the primary boundary. Off-loop + timeout-bounded.
            # Raises SSRFError (non-global) or asyncio.TimeoutError (stalled resolver).
            try:
                validated = await _resolve(host, parts.port)
            except asyncio.TimeoutError:
                return FetchResult(kind="error", error_code=ERR_TIMEOUT,
                                   error_detail="dns timeout", final_url=current)
            resp = await opener(
                current, validated, connect_timeout=connect_timeout,
                read_timeout=read_timeout, total_timeout=total_timeout, max_bytes=max_bytes)
            try:
                # Redirects: follow manually, re-validating each hop.
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location") or resp.headers.get("location")
                    if not location:
                        return FetchResult(kind="error", error_code=ERR_HTTP_STATUS,
                                           error_detail=f"{resp.status} no location", final_url=current)
                    nxt = urljoin(current, location)
                    try:
                        nxt = validate_url(nxt)
                    except SSRFError as e:
                        return FetchResult(kind="error", error_code=e.error_code,
                                           error_detail=str(e), final_url=nxt)
                    if nxt in seen:
                        return FetchResult(kind="error", error_code=ERR_REDIRECT_LIMIT,
                                           error_detail="redirect loop", final_url=nxt)
                    seen.add(current)
                    current = nxt
                    continue
                if resp.status >= 400 or resp.status < 200:
                    return FetchResult(kind="error", error_code=ERR_HTTP_STATUS,
                                       error_detail=f"http {resp.status}", final_url=current,
                                       content_type=resp.headers.get("Content-Type"))
                # Content-Length pre-check (cheap rejection before streaming).
                clen = resp.headers.get("Content-Length") or resp.headers.get("content-length")
                if clen and clen.isdigit() and int(clen) > max_bytes:
                    return FetchResult(kind="error", error_code=ERR_TOO_LARGE,
                                       error_detail=f"content-length {clen}", final_url=current)
                content_type = resp.headers.get("Content-Type") or resp.headers.get("content-type")
                # Stream with a hard cap at max_bytes+1.
                buf = bytearray()
                too_large = False
                try:
                    async for chunk in resp.iter_chunks(64 * 1024):
                        if not chunk:
                            continue
                        buf.extend(chunk)
                        if len(buf) > max_bytes:
                            too_large = True
                            break
                except asyncio.TimeoutError:
                    return FetchResult(kind="error", error_code=ERR_TIMEOUT, final_url=current)
                if too_large:
                    return FetchResult(kind="error", error_code=ERR_TOO_LARGE, final_url=current)
                raw = bytes(buf)
                # Direct image? Route to vision instead of rejecting.
                img_mime = _sniff_image(raw[:16])
                if img_mime:
                    return FetchResult(kind="image", final_url=current, content_type=img_mime,
                                       raw_bytes=raw)
                fam = _content_type_family(content_type)
                if fam and fam not in _TEXTUAL_MIMES and not fam.startswith("text/"):
                    return FetchResult(kind="error", error_code=ERR_UNSUPPORTED_TYPE,
                                       error_detail=fam, final_url=current, content_type=content_type)
                try:
                    text, title = _extract(raw, fam, max_chars=max_chars)
                except Exception as e:  # noqa: BLE001
                    return FetchResult(kind="error", error_code=ERR_EXTRACT_FAILED,
                                       error_detail=str(e), final_url=current)
                if text is None:
                    return FetchResult(kind="error", error_code=ERR_DECODE_FAILED, final_url=current)
                return FetchResult(kind="text", final_url=current, content_type=content_type,
                                   title=title, text=text)
            finally:
                try:
                    await resp.release()
                except Exception:  # noqa: BLE001
                    pass
        return FetchResult(kind="error", error_code=ERR_REDIRECT_LIMIT, final_url=current)
    except SSRFError as e:
        return FetchResult(kind="error", error_code=e.error_code, error_detail=str(e), final_url=current)
    except asyncio.TimeoutError:
        return FetchResult(kind="error", error_code=ERR_TIMEOUT, final_url=current)
    except Exception as e:  # noqa: BLE001 — unexpected transport failure
        logger.debug(f"fetch_url unexpected error for {current}: {e}")
        return FetchResult(kind="error", error_code=ERR_DECODE_FAILED, error_detail=str(e),
                           final_url=current)


def _extract(raw: bytes, fam: str, *, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    if fam == "application/pdf":
        return _extract_pdf(raw, max_chars=max_chars)
    try:
        decoded = raw.decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None, None
    if fam in ("text/html", "application/xhtml+xml"):
        return html_to_text(decoded, max_chars=max_chars)
    # text/plain, json, xml, markdown — flatten whitespace, no tag stripping.
    cleaned = _MULTI_NL_RE.sub("\n\n", decoded).strip()
    return cleaned[:max_chars], None


def _extract_pdf(raw: bytes, *, max_chars: int) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort in-memory PDF text (no disk, no OCR). Degrades to an honest marker."""
    try:
        from io import BytesIO

        from pypdf import PdfReader  # type: ignore
    except Exception:  # noqa: BLE001 — dependency optional
        return "[PDF content not extractable in this environment]", None
    try:
        reader = PdfReader(BytesIO(raw))
        chunks: List[str] = []
        total = 0
        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                chunks.append(t)
                total += len(t)
            if total >= max_chars:
                break
        text = _MULTI_NL_RE.sub("\n\n", "\n\n".join(chunks)).strip()
        return (text[:max_chars] or "[PDF had no extractable text]"), None
    except Exception as e:  # noqa: BLE001
        return f"[PDF not extractable: {e}]", None
