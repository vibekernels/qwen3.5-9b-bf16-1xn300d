#!/bin/bash
set -e

PORT="${PORT:-8888}"

echo "Qwen 3.5-9B on Tenstorrent N300"
echo "Health check on port $PORT"
echo "Chat UI not yet implemented — replace this entrypoint when ready."

# Minimal health-check server (placeholder for the chat UI).
# When the chat UI is built, replace this entire script with the real server.
exec python3 -c "
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'status': 'ok',
            'model': 'Qwen3.5-9B',
            'device': 'Tenstorrent N300',
            'port': ${PORT},
            'note': 'Chat UI coming soon. Inference binary available at /app/build/qwen-chat.'
        }).encode())
    def log_message(self, format, *args):
        pass

print('Listening on port ${PORT}')
HTTPServer(('0.0.0.0', ${PORT}), Handler).serve_forever()
"
