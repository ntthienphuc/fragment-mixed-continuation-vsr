import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import unittest
from pathlib import Path
from phase5.http_transfer import download

BODY = bytes(range(256)) * 30

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        offset = int(self.headers.get('Range', 'bytes=0-')[6:-1])
        if self.path == '/ignore':
            offset = 0
        body = BODY[offset:]
        self.send_response(206 if offset else 200)
        self.send_header('Content-Type', 'text/html' if self.path == '/html' else 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        if offset:
            self.send_header('Content-Range', f'bytes {offset}-{len(BODY)-1}/{len(BODY)}')
        self.end_headers()
        if self.path == '/break' and not offset:
            body = body[:2000]
        self.wfile.write(body)
        self.close_connection = True

class Transfers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_resume_and_range_ignored(self):
        for endpoint in ['/normal', '/ignore']:
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / 'asset.bin'
                target.with_suffix('.bin.part').write_bytes(BODY[:256])
                download(self.base + endpoint, target, hashlib.sha256(BODY).hexdigest())
                self.assertEqual(target.read_bytes(), BODY)

    def test_connection_break_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'asset.bin'
            download(self.base + '/break', target, hashlib.sha256(BODY).hexdigest())
            self.assertEqual(target.read_bytes(), BODY)

    def test_checksum_recovery_and_html_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'asset.bin'
            with self.assertRaises(ValueError):
                download(self.base + '/normal', target, '0' * 64)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix('.bin.part').exists())
            download(self.base + '/normal', target, hashlib.sha256(BODY).hexdigest())
            self.assertEqual(target.read_bytes(), BODY)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                download(self.base + '/html', Path(folder) / 'asset')

    def test_credentials_refused_on_wrong_origin(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'official HTTPS'):
                download(self.base + '/normal', Path(folder) / 'asset', authenticated=True)
