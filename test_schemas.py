"""
test_schemas.py
---------------
Sanity checks for core/schemas.py — no API key needed.
Run from autoresearch/:
    python test_schemas.py
"""

import sys
sys.path.insert(0, ".")

from core.schemas import (
    Snippet, Fact,
    PlannerOutput, RetrieverOutput, SummarizerOutput, CriticOutput,
    validate,
)

print("=" * 50)

# ── Test 1: Valid PlannerOutput passes validation ─────────────────────────────
print("TEST 1: validate() — valid PlannerOutput")
valid_plan: PlannerOutput = {
    "sub_questions": ["What causes LLM hallucinations?"],
    "keywords":      [["hallucination", "LLM", "causes"]],
    "scope":         "Covers transformer-based LLMs only.",
}
validate(PlannerOutput, valid_plan)
print("PASS ✓")

# ── Test 2: Missing key raises ValueError ─────────────────────────────────────
print("\nTEST 2: validate() — missing 'scope' key raises ValueError")
bad_plan = {"sub_questions": ["Q1"], "keywords": [["kw1"]]}   # missing 'scope'
try:
    validate(PlannerOutput, bad_plan)
    print("FAIL ✗ — should have raised ValueError")
except ValueError as e:
    print(f"PASS ✓  (raised ValueError: {e})")

# ── Test 3: Valid CriticOutput with requery=False ─────────────────────────────
print("\nTEST 3: validate() — valid CriticOutput (no requery)")
critic_out: CriticOutput = {
    "gaps":               ["No data on mitigation strategies"],
    "contradictions":     [],
    "requery_needed":     False,
    "requery_questions":  [],
}
validate(CriticOutput, critic_out)
print("PASS ✓")

# ── Test 4: Valid RetrieverOutput structure ───────────────────────────────────
print("\nTEST 4: validate() — valid RetrieverOutput")
snippet: Snippet = {
    "text":  "LLMs hallucinate because they predict tokens probabilistically.",
    "url":   "https://en.wikipedia.org/wiki/Hallucination_(AI)",
    "title": "Hallucination (artificial intelligence)",
}
retriever_out: RetrieverOutput = {
    "sub_question": "What causes LLM hallucinations?",
    "snippets":     [snippet],
}
validate(RetrieverOutput, retriever_out)
print("PASS ✓")

# ── Test 5: Valid SummarizerOutput ────────────────────────────────────────────
print("\nTEST 5: validate() — valid SummarizerOutput")
fact: Fact = {
    "claim":      "LLMs hallucinate due to statistical token prediction.",
    "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
    "confidence": "high",
}
summarizer_out: SummarizerOutput = {
    "sub_question": "What causes LLM hallucinations?",
    "facts":        [fact],
}
validate(SummarizerOutput, summarizer_out)
print("PASS ✓")

print("\n" + "=" * 50)
print("All schema tests passed. Step 2 complete!")
