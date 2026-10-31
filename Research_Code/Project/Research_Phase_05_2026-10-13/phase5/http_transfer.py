"""Retry interrupted downloads without sending credentials to another origin."""
import base64
import http.client
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from .common import digest


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old, new = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if req.has_header('Authorization') and (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise ValueError('Authenticated redirect to another origin refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def receive(url, partial, authenticated):
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'User-Agent': 'AutoAVSR-Research/1.0'})
    if authenticated:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'thor.robots.ox.ac.uk':
            raise ValueError('LRS2 authentication is restricted to the official HTTPS host')
        user, password = os.environ.get('LRS2_USER'), os.environ.get('LRS2_PASSWORD')
        if not user or not password:
            raise ValueError('Set LRS2_USER and LRS2_PASSWORD in the process environment')
        request.add_header('Authorization', 'Basic ' + base64.b64encode(f'{user}:{password}'.encode()).decode())
    if offset:
        request.add_header('Range', f'bytes={offset}-')
    with urllib.request.build_opener(Redirect).open(request, timeout=120) as response:
        if 'text/html' in response.headers.get('Content-Type', '').lower():
            raise ValueError('Received HTML instead of asset bytes')
        resumed = response.status == 206
        total = None
        if resumed:
            match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
            if not match or int(match[1]) != offset or int(match[2]) + 1 != int(match[3]):
                raise ValueError('Unexpected Content-Range')
            total = int(match[3])
        elif response.status != 200:
            raise ValueError(f'Unexpected HTTP status {response.status}')
        expected = response.headers.get('Content-Length')
        count = 0
        with partial.open('ab' if resumed else 'wb') as handle:
            while block := response.read(1 << 20):
                handle.write(block)
                count += len(block)
        if expected is not None and count != int(expected):
            raise OSError('Interrupted HTTP body; partial data retained')
        if total is not None and partial.stat().st_size != total:
            raise OSError('Range response did not complete the asset')


def download(url, destination, expected=None, algorithm='sha256', authenticated=False, retries=3):
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if expected and digest(target, algorithm) != expected:
            raise ValueError('Existing file hash mismatch')
        return digest(target)
    partial = target.with_suffix(target.suffix + '.part')
    complete = bool(expected and partial.exists() and digest(partial, algorithm) == expected)
    for attempt in range(retries + 1):
        if complete:
            break
        try:
            receive(url, partial, authenticated)
            break
        except (OSError, urllib.error.URLError, http.client.IncompleteRead) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))
    if expected and digest(partial, algorithm) != expected:
        bad = partial.with_suffix('.bad-' + digest(partial)[:12])
        partial.replace(bad)
        raise ValueError('Checksum mismatch; corrupt bytes quarantined; rerun to download afresh')
    partial.replace(target)
    return digest(target)
