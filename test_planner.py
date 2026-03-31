"""
test_planner.py
---------------
Tests for agents/planner.py

Two test modes:
  1. Offline mock test — no API key needed, tests validation logic
  2. Live API test    — needs OPENAI_API_KEY in .env

Run from autoresearch/:
    python test_planner.py
"""

import sys
import json
sys.path.insert(0, ".")

from unittest.mock import patch
from agents.planner import PlannerAgent
from core.schemas import validate, PlannerOutput

print("=" * 50)

# ── Test 1: Offline — valid mock response passes validation ───────────────────
print("TEST 1: Offline — mock LLM response with valid schema")

mock_response = {
    "sub_questions": [
        "What are the main causes of LLM hallucinations?",
        "How do attention mechanisms contribute to hallucination?",
        "What mitigation strategies have been proposed for hallucination?",
    ],
    "keywords": [
        ["LLM hallucination causes", "language model errors"],
        ["attention mechanism", "transformer hallucination"],
        ["hallucination mitigation", "RLHF", "retrieval augmented generation"],
    ],
    "scope": "Covers transformer-based large language models only, not smaller rule-based systems.",
}

# Patch chat_json so no real API call is made
with patch("agents.planner.chat_json", return_value=mock_response):
    planner = PlannerAgent()
    result  = planner.run({"question": "Why do LLMs hallucinate?"})

assert result["scope"] == mock_response["scope"], "FAIL: scope mismatch"
assert len(result["sub_questions"]) == 3,         "FAIL: wrong sub_question count"
assert len(result["keywords"]) == 3,              "FAIL: wrong keywords count"
print("PASS ✓")

# ── Test 2: Offline — mismatched list lengths are caught ─────────────────────
print("\nTEST 2: Offline — mismatched sub_questions/keywords raises ValueError")

bad_response = {
    "sub_questions": ["Q1", "Q2"],
    "keywords":      [["kw1"]],         # Only 1 keyword list for 2 sub-questions
    "scope":         "Some scope.",
}

with patch("agents.planner.chat_json", return_value=bad_response):
    try:
        planner = PlannerAgent()
        planner.run({"question": "Test?"})
        print("FAIL ✗ — should have raised ValueError")
    except ValueError as e:
        print(f"PASS ✓  (raised ValueError: {e})")

# ── Test 3: Offline — missing schema key is caught ───────────────────────────
print("\nTEST 3: Offline — missing 'scope' key raises ValueError")

missing_key_response = {
    "sub_questions": ["Q1"],
    "keywords":      [["kw1"]],
    # 'scope' is missing
}

with patch("agents.planner.chat_json", return_value=missing_key_response):
    try:
        planner = PlannerAgent()
        planner.run({"question": "Test?"})
        print("FAIL ✗ — should have raised ValueError")
    except ValueError as e:
        print(f"PASS ✓  (raised ValueError: {e})")

# ── Test 4: Live API test ─────────────────────────────────────────────────────
print("\nTEST 4: Live API — real question (needs OPENAI_API_KEY)")
try:
    planner = PlannerAgent()
    result  = planner.run({
        "question": "What are the primary mechanisms by which large language models hallucinate?"
    })

    print(f"  Scope        : {result['scope']}")
    print(f"  Sub-questions: {len(result['sub_questions'])}")
    for i, (q, kws) in enumerate(zip(result["sub_questions"], result["keywords"]), 1):
        print(f"    {i}. {q}")
        print(f"       Keywords: {kws}")

    validate(PlannerOutput, result)
    print("PASS ✓  Schema validated.")

except RuntimeError as e:
    print(f"SKIP — API unavailable: {e}")

print("\n" + "=" * 50)
print("All Planner tests done. Step 3 complete!")