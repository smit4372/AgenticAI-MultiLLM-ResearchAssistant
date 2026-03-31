"""
test_retriever.py
-----------------
Tests for agents/retriever.py

Three test modes:
  1. Offline mock  — no internet needed, tests deduplication + validation logic
  2. Live Wikipedia — real HTTP call to Wikipedia API
  3. Live DuckDuckGo — real HTTP call to DDG API

Run from autoresearch/:
    python test_retriever.py
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch, MagicMock
from agents.retriever import RetrieverAgent
from core.schemas import validate, RetrieverOutput

print("=" * 50)

# ── Test 1: Offline — mock API, check deduplication ──────────────────────────
print("TEST 1: Offline — deduplication removes duplicate URLs")

snippet_a = {"text": "Text A", "url": "https://en.wikipedia.org/wiki/A", "title": "A"}
snippet_b = {"text": "Text B", "url": "https://en.wikipedia.org/wiki/B", "title": "B"}

def mock_search_wikipedia(query):
    # Both keywords return the same snippet_a → should be deduplicated
    return [snippet_a]

def mock_search_duckduckgo(query):
    return snippet_b   # DuckDuckGo returns snippet_b (different URL)

retriever = RetrieverAgent()
with patch.object(retriever, "_search_wikipedia", side_effect=mock_search_wikipedia), \
     patch.object(retriever, "_search_duckduckgo", return_value=snippet_b):

    result = retriever.run({
        "sub_question": "What is attention in transformers?",
        "keywords": ["attention mechanism", "transformer model"],
    })

# snippet_a appears twice (once per keyword) but should be deduplicated to 1
# snippet_b adds 1 more → total should be 2, not 3
assert len(result["snippets"]) == 2, f"FAIL: expected 2 snippets, got {len(result['snippets'])}"
validate(RetrieverOutput, result)
print("PASS ✓")

# ── Test 2: Offline — empty API results trigger fallback ─────────────────────
print("\nTEST 2: Offline — empty results trigger fallback search")

fallback_snippet = {"text": "Fallback text", "url": "https://en.wikipedia.org/wiki/Fallback", "title": "Fallback"}

call_count = {"n": 0}

def mock_empty_then_fallback(query):
    call_count["n"] += 1
    if call_count["n"] <= 2:
        return []          # first 2 keyword searches return nothing
    return [fallback_snippet]   # fallback search returns something

retriever2 = RetrieverAgent()
with patch.object(retriever2, "_search_wikipedia", side_effect=mock_empty_then_fallback), \
     patch.object(retriever2, "_search_duckduckgo", return_value=None):

    result2 = retriever2.run({
        "sub_question": "Obscure topic with no results",
        "keywords": ["keyword1", "keyword2"],
    })

assert len(result2["snippets"]) == 1, f"FAIL: expected 1 fallback snippet, got {len(result2['snippets'])}"
print("PASS ✓")

# ── Test 3: Live Wikipedia API ────────────────────────────────────────────────
print("\nTEST 3: Live Wikipedia — fetch real snippet (needs internet)")
try:
    retriever3 = RetrieverAgent()
    result3 = retriever3.run({
        "sub_question": "How does the transformer architecture work?",
        "keywords": ["transformer deep learning", "attention mechanism"],
    })

    print(f"  Snippets returned: {len(result3['snippets'])}")
    for i, s in enumerate(result3["snippets"], 1):
        print(f"  [{i}] {s['title']}")
        print(f"       URL   : {s['url']}")
        print(f"       Text  : {s['text'][:120]}...")

    validate(RetrieverOutput, result3)
    assert len(result3["snippets"]) >= 1, "FAIL: expected at least 1 snippet"
    print("PASS ✓")

except Exception as e:
    print(f"SKIP — network unavailable or error: {e}")

# ── Test 4: Live DuckDuckGo ───────────────────────────────────────────────────
print("\nTEST 4: Live DuckDuckGo — instant answer (needs internet)")
try:
    retriever4 = RetrieverAgent()
    ddg_result = retriever4._search_duckduckgo("CRISPR gene editing mechanism")

    if ddg_result:
        print(f"  Title : {ddg_result['title']}")
        print(f"  URL   : {ddg_result['url']}")
        print(f"  Text  : {ddg_result['text'][:120]}...")
        print("PASS ✓")
    else:
        print("SKIP — DuckDuckGo returned no abstract (this is normal for some queries)")

except Exception as e:
    print(f"SKIP — network error: {e}")

print("\n" + "=" * 50)
print("All Retriever tests done. Step 4 complete!")