"""
Smoke-test: verify every DashScope-dependent function in the pipeline works.

Covers all Python scripts that use BLT_API_KEY + DashScope base URL:
  1. llm.py        — BltClient chat (used by all scripts)
  2. 0.enrich      — query rewriting   (BLT_REWRITE_MODEL)
  3. 3.rank        — reranking         (BLT_API_KEY)
  4. 4.llm_refine  — paper filtering   (BLT_FILTER_MODEL)
  5. 6.generate    — paper summarising  (Summarized_LLM_MODEL)

Usage:
    python tests/test_dashscope_pipeline.py <API_KEY>
    # or:
    BLT_API_KEY=sk-... python tests/test_dashscope_pipeline.py

Also audits frontend (browser-side) calls to flag CORS-blocked features.
"""

import json
import os
import sys
import urllib.request
import urllib.error

DASHSCOPE_BASE = "https://coding.dashscope.aliyuncs.com/v1"
MODEL = "glm-5"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _post_json(url, payload, api_key, timeout=30):
    """Send a POST request and return (status, body_dict)."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return resp.status, body
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(body_text)
        except json.JSONDecodeError:
            body = {"raw": body_text[:300]}
        return e.code, body


def _check(label, passed, detail=""):
    icon = "✅" if passed else "❌"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  —  {detail}"
    print(msg)
    return passed


# ──────────────────────────────────────────────────────────────
# 1. Chat Completions (core — used by all pipeline steps)
# ──────────────────────────────────────────────────────────────

def test_chat_completions(api_key):
    print("\n[1/4] Chat Completions (BltClient.chat)")
    url = f"{DASHSCOPE_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 4,
        "temperature": 0.1,
    }
    status, body = _post_json(url, payload, api_key)
    content = ""
    if "choices" in body and body["choices"]:
        content = body["choices"][0].get("message", {}).get("content", "")
    return _check(
        "chat/completions",
        status == 200 and bool(content),
        f"HTTP {status}, reply={content[:50]!r}" if content else f"HTTP {status}, body={body}",
    )


# ──────────────────────────────────────────────────────────────
# 2. Chat with response_format=json_object (enrich + refine)
# ──────────────────────────────────────────────────────────────

def test_json_mode(api_key):
    print("\n[2/4] JSON mode (response_format, used by enrich & refine)")
    url = f"{DASHSCOPE_BASE}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": 'Return JSON: {"status":"ok"}'},
        ],
        "max_tokens": 20,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    status, body = _post_json(url, payload, api_key)
    content = ""
    if "choices" in body and body["choices"]:
        content = body["choices"][0].get("message", {}).get("content", "")
    json_ok = False
    if content:
        try:
            json.loads(content)
            json_ok = True
        except json.JSONDecodeError:
            pass
    return _check(
        "json_object response_format",
        status == 200 and json_ok,
        f"HTTP {status}, parseable JSON={json_ok}, content={content[:80]!r}",
    )


# ──────────────────────────────────────────────────────────────
# 3. Rerank endpoint (used by 3.rank_papers.py)
# ──────────────────────────────────────────────────────────────

def test_rerank(api_key):
    print("\n[3/4] Rerank (/v1/rerank, used by rank_papers)")
    url = f"{DASHSCOPE_BASE}/rerank"
    payload = {
        "model": MODEL,
        "query": "machine learning",
        "documents": [
            "Deep learning for NLP",
            "Cooking recipes for beginners",
        ],
        "top_n": 2,
    }
    status, body = _post_json(url, payload, api_key)
    # Rerank may not be supported by DashScope — that's important to know
    has_results = isinstance(body.get("results"), list) and len(body.get("results", [])) > 0
    if status == 200 and has_results:
        return _check("rerank", True, f"HTTP {status}, results={len(body['results'])}")
    elif status == 404:
        return _check(
            "rerank",
            False,
            "HTTP 404 — DashScope does NOT support /v1/rerank. "
            "The pipeline step 3.rank_papers.py will FAIL if it tries to rerank.",
        )
    else:
        err = body.get("error", {})
        detail = err.get("message", "") or str(body)[:200]
        return _check("rerank", False, f"HTTP {status} — {detail}")


# ──────────────────────────────────────────────────────────────
# 4. Frontend (browser) audit — CORS check
# ──────────────────────────────────────────────────────────────

def test_cors_headers(api_key):
    print("\n[4/4] CORS headers (browser-side feasibility)")
    url = f"{DASHSCOPE_BASE}/chat/completions"
    # Simulate a preflight-like request by checking response headers
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Origin": "https://example.github.io",
        },
        method="POST",
    )
    cors_ok = False
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            acao = resp.headers.get("Access-Control-Allow-Origin", "")
            cors_ok = bool(acao)
            _check(
                "CORS (Access-Control-Allow-Origin)",
                cors_ok,
                f"Header value: {acao!r}" if acao else "Header MISSING — browser calls will be blocked",
            )
    except urllib.error.HTTPError as e:
        acao = e.headers.get("Access-Control-Allow-Origin", "")
        cors_ok = bool(acao)
        _check(
            "CORS (Access-Control-Allow-Origin)",
            cors_ok,
            f"HTTP {e.code}, header: {acao!r}" if acao else f"HTTP {e.code}, header MISSING",
        )
    return cors_ok


# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────

def main():
    api_key = (sys.argv[1] if len(sys.argv) > 1 else None) or os.environ.get("BLT_API_KEY", "")
    if not api_key:
        print("Usage: python tests/test_dashscope_pipeline.py <API_KEY>")
        sys.exit(1)

    print("=" * 60)
    print("DashScope Pipeline Compatibility Test")
    print(f"  Base URL : {DASHSCOPE_BASE}")
    print(f"  Model    : {MODEL}")
    print(f"  Key      : {api_key[:6]}...{api_key[-4:]}")
    print("=" * 60)

    r1 = test_chat_completions(api_key)
    r2 = test_json_mode(api_key)
    r3 = test_rerank(api_key)
    cors = test_cors_headers(api_key)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Server-side (GitHub Actions pipeline) ---")
    rows = [
        ("src/llm.py (BltClient.chat)",         "chat/completions", r1),
        ("src/0.enrich_config_queries.py",       "chat (json_mode)", r2),
        ("src/4.llm_refine_papers.py",           "chat (json_mode)", r2),
        ("src/6.generate_docs.py (summarise)",   "chat/completions", r1),
        ("src/3.rank_papers.py (rerank)",        "/v1/rerank",       r3),
    ]
    for script, endpoint, ok in rows:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {script:<42s} [{endpoint}]")

    print("\n--- Browser-side (frontend, GitHub Pages) ---")
    browser_features = [
        ("生成候选 (smart-query LLM call)",           "chat/completions", cors, "CORS blocked"),
        ("私人研讨 Chat (chat.discussion.js)",         "chat/completions", cors, "CORS blocked"),
        ("验证 DashScope Key (secret.session.js)",     "—",                True,  "format-only check, no API call"),
        ("验证 GitHub Token",                          "api.github.com",   True,  "GitHub API supports CORS"),
        ("保存 config.yaml / Secrets",                 "api.github.com",   True,  "GitHub API supports CORS"),
        ("触发 GitHub Actions workflow",               "api.github.com",   True,  "GitHub API supports CORS"),
        ("加载 config.yaml (read-only)",               "same-origin",      True,  "static file, no CORS issue"),
    ]
    for feature, endpoint, ok, note in browser_features:
        icon = "✅" if ok else "❌"
        suffix = f"  ({note})" if note else ""
        print(f"  {icon} {feature:<42s} [{endpoint}]{suffix}")

    print()
    all_pipeline_ok = r1 and r2
    if all_pipeline_ok and r3:
        print("🟢 All pipeline functions work with DashScope.")
    elif all_pipeline_ok and not r3:
        print("🟡 Core pipeline works. Rerank is NOT supported — see note below.")
    else:
        print("🔴 Core pipeline functions FAILED. Check errors above.")

    if not cors:
        print("🟡 Browser LLM features (生成候选, 私人研讨 Chat) are CORS-blocked.")
        print("   These features require a CORS-friendly API or a proxy.")

    if not r3:
        print("\n⚠️  RERANK NOTE:")
        print("   DashScope does not support /v1/rerank.")
        print("   src/3.rank_papers.py will fail if it attempts reranking.")
        print("   Check if your pipeline config skips reranking or uses a fallback.")

    print()
    sys.exit(0 if all_pipeline_ok else 1)


if __name__ == "__main__":
    main()
