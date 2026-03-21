"""
Quick smoke test: verify a DashScope API key works from a server-side context.

Mimics exactly what the GitHub Actions pipeline does:
  - Uses the same base URL (https://coding.dashscope.aliyuncs.com/v1)
  - Uses the same model (glm-5)
  - Sends a minimal chat completion request

Usage:
    python tests/test_dashscope_api_key.py <API_KEY>
    # or via env var:
    BLT_API_KEY=sk-... python tests/test_dashscope_api_key.py
"""

import sys
import os
import json
import urllib.request
import urllib.error

BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
MODEL = "glm-5"


def verify_api_key(api_key):
    url = f"{BASE_URL}/chat/completions"
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 4,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    print(f"Base URL : {BASE_URL}")
    print(f"Model    : {MODEL}")
    print(f"Key      : {api_key[:6]}...{api_key[-4:]}")
    print(f"Endpoint : {url}")
    print("-" * 50)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))

            print(f"HTTP {status} — SUCCESS")
            # Extract the reply
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
                print(f"Model reply: {content}")
            # Show usage if available
            usage = body.get("usage", {})
            if usage:
                print(f"Token usage: {usage}")
            print("-" * 50)
            print("✅ Pipeline will work — API key is valid and DashScope is reachable.")
            return True

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} — FAILED")
        try:
            err_json = json.loads(error_body)
            err_code = err_json.get("error", {}).get("code", "")
            err_msg = err_json.get("error", {}).get("message", "")
            print(f"Error code   : {err_code}")
            print(f"Error message: {err_msg}")
        except json.JSONDecodeError:
            print(f"Raw response : {error_body[:300]}")
        print("-" * 50)
        print("❌ Pipeline will NOT work — check the error above.")
        return False

    except urllib.error.URLError as e:
        print(f"Network error: {e.reason}")
        print("-" * 50)
        print("❌ Cannot reach DashScope — check network/DNS.")
        return False


if __name__ == "__main__":
    key = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("BLT_API_KEY", "")
    if not key:
        print("Usage: python tests/test_dashscope_api_key.py <API_KEY>")
        print("   or: BLT_API_KEY=sk-... python tests/test_dashscope_api_key.py")
        sys.exit(1)
    ok = verify_api_key(key)
    sys.exit(0 if ok else 1)
