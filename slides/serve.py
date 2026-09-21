#!/usr/bin/env python3
"""Preview server for the decks: python3 serve.py [port]  (default 8765, http://127.0.0.1:8765/).

Like `python3 -m http.server`, but every response says Cache-Control: no-store, so edited slides, stylesheets
and widget modules show up on a plain reload instead of being served stale from the browser cache.
"""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet: one line per request is noise during a lecture preview
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    root = os.path.dirname(os.path.abspath(__file__))
    print(f'serving {root} at http://127.0.0.1:{port}/ (no caching)')
    ThreadingHTTPServer(('127.0.0.1', port), partial(Handler, directory=root)).serve_forever()
