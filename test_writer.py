"""
test_writer.py
--------------
Tests for agents/writer.py

Run from autoresearch/:
    python test_writer.py
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch
from agents.writer import WriterAgent

print("=" * 50)

# ── Reusable test data ─────────────────────────────────────────────────────────
SAMPLE_INPUT = {
    "question": "What are the primary mechanisms by which LLMs hallucinate?",
    "scope": "Covers transformer-based LLMs only, not rule-based systems.",
    "sub_questions": [
        "What causes LLM hallucinations?",
        "What mitigation strategies exist?",
        "How does RLHF reduce hallucination?",
    ],
    "all_facts": [
        {
            "claim": "LLMs hallucinate because they predict tokens statistically, not factually.",
            "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
            "confidence": "high",
        },
        {
            "claim": "RAG reduces hallucination by grounding responses in retrieved documents.",
            "source_url": "https://en.wikipedia.org/wiki/RAG",
            "confidence": "high",
        },
        {
            "claim": "RLHF trains models on human preference data to prefer accurate answers.",
            "source_url": "https://en.wikipedia.org/wiki/RLHF",
            "confidence": "high",
        },
        {
            "claim": "Some researchers suggest hallucination may be impossible to fully eliminate.",
            "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
            "confidence": "low",
        },
    ],
}

MOCK_REPORT = """# What are the primary mechanisms by which LLMs hallucinate?

## Executive Summary
Large language models hallucinate because they generate text through statistical
token prediction rather than factual lookup. Mitigation strategies such as
Retrieval-Augmented Generation (RAG) and RLHF have shown promise in reducing
these errors [Source](https://en.wikipedia.org/wiki/Hallucination_(AI)).

## Scope
This report covers transformer-based large language models only and does not
address rule-based or symbolic AI systems.

## Findings

### What causes LLM hallucinations?
LLMs generate text by predicting the most statistically likely next token,
which can produce plausible-sounding but factually incorrect output
[Source](https://en.wikipedia.org/wiki/Hallucination_(AI)).

### What mitigation strategies exist?
Retrieval-Augmented Generation (RAG) reduces hallucination by grounding model
responses in documents retrieved at inference time
[Source](https://en.wikipedia.org/wiki/RAG).

### How does RLHF reduce hallucination?
RLHF trains models using human preference data, steering them toward
responses that human raters consider accurate
[Source](https://en.wikipedia.org/wiki/RLHF).

## Limitations
- One fact carries low confidence: the claim that hallucination may be
  impossible to fully eliminate (low confidence).
- Sources limited to Wikipedia and DuckDuckGo.

## References
1. https://en.wikipedia.org/wiki/Hallucination_(AI)
2. https://en.wikipedia.org/wiki/RAG
3. https://en.wikipedia.org/wiki/RLHF
"""

# ── Test 1: Offline — valid mock report is returned as-is ────────────────────
print("TEST 1: Offline — valid mock report passes length check")

with patch("agents.writer.chat", return_value=MOCK_REPORT):
    writer = WriterAgent()
    result = writer.run(SAMPLE_INPUT)

assert isinstance(result, str),      "FAIL: output should be a string"
assert len(result) > 100,            "FAIL: report is too short"
assert "## Executive Summary" in result, "FAIL: missing Executive Summary"
assert "## Findings"          in result, "FAIL: missing Findings section"
assert "## Limitations"       in result, "FAIL: missing Limitations section"
assert "## References"        in result, "FAIL: missing References section"
print("PASS ✓")

# ── Test 2: Offline — short output raises ValueError ─────────────────────────
print("\nTEST 2: Offline — suspiciously short output raises ValueError")

with patch("agents.writer.chat", return_value="Too short."):
    try:
        writer2 = WriterAgent()
        writer2.run(SAMPLE_INPUT)
        print("FAIL ✗ — should have raised ValueError")
    except ValueError as e:
        print(f"PASS ✓  (raised ValueError: {e})")

# ── Test 3: Offline — low confidence facts present in report ─────────────────
print("\nTEST 3: Offline — low confidence facts flagged in mock report")

assert "low confidence" in MOCK_REPORT.lower(), \
    "FAIL: low confidence facts should be flagged in report"
print("PASS ✓")

# ── Test 4: Live API ──────────────────────────────────────────────────────────
print("\nTEST 4: Live API — real report generation (needs OPENAI_API_KEY)")
try:
    writer4 = WriterAgent()
    result4 = writer4.run(SAMPLE_INPUT)

    print(f"  Report length : {len(result4)} chars")
    # Print first 500 chars as preview
    print(f"  Preview:\n{result4[:500]}...")

    assert "## Executive Summary" in result4, "FAIL: missing Executive Summary"
    assert "## Findings"          in result4, "FAIL: missing Findings"
    print("\nPASS ✓")

except RuntimeError as e:
    print(f"SKIP — API unavailable: {e}")

print("\n" + "=" * 50)
print("All Writer tests done. Step 7 complete!")