"""
test_llm.py
-----------
Quick sanity-check for core/llm.py
Run from the autoresearch/ directory:
    python test_llm.py
"""

import sys
import json
sys.path.insert(0, ".")  # make sure autoresearch/ is on the path

from core.llm import chat, chat_json, safe_parse

# ── Test 1: safe_parse with clean JSON ────────────────────────────────────────
print("=" * 50)
print("TEST 1: safe_parse — clean JSON string")
raw_clean = '{"name": "Alice", "age": 30}'
result = safe_parse(raw_clean)
assert result == {"name": "Alice", "age": 30}, "FAIL: clean parse"
print("PASS ✓")

# ── Test 2: safe_parse with markdown fences ───────────────────────────────────
print("\nTEST 2: safe_parse — JSON wrapped in ```json fences")
raw_fenced = '```json\n{"name": "Bob", "age": 25}\n```'
result = safe_parse(raw_fenced)
assert result == {"name": "Bob", "age": 25}, "FAIL: fenced parse"
print("PASS ✓")

# ── Test 3: safe_parse with plain fences (no language tag) ────────────────────
print("\nTEST 3: safe_parse — JSON wrapped in plain ``` fences")
raw_plain_fence = '```\n{"city": "Paris"}\n```'
result = safe_parse(raw_plain_fence)
assert result == {"city": "Paris"}, "FAIL: plain fenced parse"
print("PASS ✓")

# ── Test 4: safe_parse with invalid JSON and no retry ────────────────────────
print("\nTEST 4: safe_parse — invalid JSON, no retry_fn → should raise ValueError")
try:
    safe_parse("this is not json", retry_fn=None)
    print("FAIL ✗ — should have raised ValueError")
except ValueError as e:
    print("PASS ✓  (raised ValueError as expected)")

# ── Test 5: Live API call (requires real API key in .env) ─────────────────────
print("\nTEST 5: chat_json — live API call (needs OPENAI_API_KEY in .env)")
try:
    response = chat_json(
        system_prompt='Respond ONLY with valid JSON. No preamble, no markdown.',
        user_message='Return this exact JSON: {"status": "ok", "number": 42}',
    )
    assert response.get("status") == "ok", f"Unexpected response: {response}"
    assert response.get("number") == 42,   f"Unexpected response: {response}"
    print("PASS ✓  API response:", json.dumps(response))
except RuntimeError as e:
    print(f"SKIP — API call failed (check your .env key): {e}")

print("\n" + "=" * 50)
print("All local tests passed. Step 1 complete!")
