"""
Render.com 入口 — 静态文件 + Firebase API 代理
"""
import os
import urllib.request
import urllib.parse
import urllib.error
from flask import Flask, request, Response, send_from_directory

app = Flask(__name__, static_folder="docs", static_url_path="")

BACKENDS = {
    "identitytoolkit": "https://identitytoolkit.googleapis.com",
    "securetoken": "https://securetoken.googleapis.com",
    "firestore": "https://firestore.googleapis.com",
}


@app.route("/")
def index():
    return send_from_directory("docs", "index.html")


@app.route("/api/fb-proxy", methods=["GET", "POST", "OPTIONS"])
def fb_proxy():
    # CORS
    if request.method == "OPTIONS":
        resp = Response("", status=204)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp

    p = request.args.get("p", "")
    slash_idx = p.find("/")
    if slash_idx < 0:
        return {"error": "Missing service path"}, 400

    service = p[:slash_idx]
    rest = p[slash_idx + 1 :]
    backend = BACKENDS.get(service)
    if not backend:
        return {"error": "Unknown service"}, 404

    # Rebuild query string excluding 'p'
    qs_parts = [
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
        for k, v in request.args.items()
        if k != "p"
    ]
    qs = "&".join(qs_parts)
    target_url = f"{backend}/{rest}{'?' + qs if qs else ''}"

    # Forward body
    body = request.get_data() or None
    if body is not None and len(body) == 0:
        body = None

    # Forward headers
    headers = {}
    if request.content_type:
        headers["Content-Type"] = request.content_type
    auth = request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth

    try:
        req = urllib.request.Request(
            target_url,
            data=body,
            headers=headers,
            method=request.method,
        )
        with urllib.request.urlopen(req, timeout=15) as upstream:
            data = upstream.read()
            ct = upstream.headers.get("Content-Type", "application/json")
            resp = Response(data, status=upstream.status)
            resp.headers["Content-Type"] = ct
    except urllib.error.HTTPError as e:
        data = e.read()
        ct = e.headers.get("Content-Type", "application/json")
        resp = Response(data, status=e.code)
        resp.headers["Content-Type"] = ct
    except Exception as e:
        resp = Response(
            f'{{"error":"Proxy upstream error","message":"{str(e)}"}}',
            status=502,
            content_type="application/json",
        )

    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
