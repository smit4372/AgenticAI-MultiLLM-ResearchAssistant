"""
test_orchestrator.py
--------------------
Tests for core/orchestrator.py

All tests are offline — every agent is mocked so no API key or
internet connection is needed.

Run from autoresearch/:
    python test_orchestrator.py
"""

import sys
import json
sys.path.insert(0, ".")

from pathlib import Path
from unittest.mock import patch, MagicMock
from core.orchestrator import Orchestrator

print("=" * 50)

# ── Shared mock return values ──────────────────────────────────────────────────

MOCK_PLAN = {
    "sub_questions": [
        "What causes LLM hallucinations?",
        "What mitigation strategies exist?",
    ],
    "keywords": [
        ["LLM hallucination", "language model errors"],
        ["hallucination mitigation", "RAG RLHF"],
    ],
    "scope": "Covers transformer-based LLMs only.",
}

MOCK_RETRIEVER_OUT = {
    "sub_question": "What causes LLM hallucinations?",
    "snippets": [
        {"text": "LLMs hallucinate due to statistical prediction.",
         "url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
         "title": "Hallucination (AI)"},
    ],
}

MOCK_SUMMARIZER_OUT = {
    "sub_question": "What causes LLM hallucinations?",
    "facts": [
        {"claim": "LLMs hallucinate due to statistical token prediction.",
         "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
         "confidence": "high"},
        {"claim": "RAG reduces hallucination by grounding responses.",
         "source_url": "https://en.wikipedia.org/wiki/RAG",
         "confidence": "high"},
    ],
}

MOCK_CRITIC_NO_REQUERY = {
    "gaps":              [],
    "contradictions":    [],
    "requery_needed":    False,
    "requery_questions": [],
}

MOCK_CRITIC_REQUERY = {
    "gaps":              ["Missing detail on RLHF"],
    "contradictions":    [],
    "requery_needed":    True,
    "requery_questions": ["How does RLHF specifically reduce hallucination?"],
}

MOCK_REPORT = """# Research Report

## Executive Summary
Some summary here with enough text to pass the length check.
This is a test report generated for testing purposes only.

## Scope
Covers transformer-based LLMs only.

## Findings

### What causes LLM hallucinations?
LLMs hallucinate due to statistical token prediction.

## Limitations
- Sources limited to Wikipedia and DuckDuckGo.

## References
1. https://en.wikipedia.org/wiki/Hallucination_(AI)
"""

# ── Test 1: Happy path — no requery needed ────────────────────────────────────
print("TEST 1: Offline — happy path, no requery")

with patch("core.orchestrator.PlannerAgent") as MockPlanner, \
     patch("core.orchestrator.RetrieverAgent") as MockRetriever, \
     patch("core.orchestrator.SummarizerAgent") as MockSummarizer, \
     patch("core.orchestrator.CriticAgent") as MockCritic, \
     patch("core.orchestrator.WriterAgent") as MockWriter:

    MockPlanner.return_value.run.return_value    = MOCK_PLAN
    MockRetriever.return_value.run.return_value  = MOCK_RETRIEVER_OUT
    MockSummarizer.return_value.run.return_value = MOCK_SUMMARIZER_OUT
    MockCritic.return_value.run.return_value     = MOCK_CRITIC_NO_REQUERY
    MockWriter.return_value.run.return_value     = MOCK_REPORT

    orch = Orchestrator()
    report_path = orch.run("Why do LLMs hallucinate?")

# Check outputs were created
assert Path(report_path).exists(), f"FAIL: report not saved at {report_path}"
assert "run_trace" in str(list(Path("outputs").glob("run_trace_*.json"))[0])

# Check trace has correct number of entries
# Planner(1) + Retriever×2(2) + Summarizer×2(2) + Critic(1) + Writer(1) = 7
assert len(orch.trace) == 7, f"FAIL: expected 7 trace entries, got {len(orch.trace)}"

# Critic should only be called once (no requery)
assert MockCritic.return_value.run.call_count == 1, "FAIL: Critic called too many times"

print(f"PASS ✓  (report at {report_path})")

# ── Test 2: Requery path — critic triggers one extra loop ─────────────────────
print("\nTEST 2: Offline — critic triggers one requery loop")

with patch("core.orchestrator.PlannerAgent") as MockPlanner, \
     patch("core.orchestrator.RetrieverAgent") as MockRetriever, \
     patch("core.orchestrator.SummarizerAgent") as MockSummarizer, \
     patch("core.orchestrator.CriticAgent") as MockCritic, \
     patch("core.orchestrator.WriterAgent") as MockWriter:

    MockPlanner.return_value.run.return_value    = MOCK_PLAN
    MockRetriever.return_value.run.return_value  = MOCK_RETRIEVER_OUT
    MockSummarizer.return_value.run.return_value = MOCK_SUMMARIZER_OUT
    MockWriter.return_value.run.return_value     = MOCK_REPORT

    # First critic call → requery; second call → satisfied
    MockCritic.return_value.run.side_effect = [
        MOCK_CRITIC_REQUERY,       # loop 1: needs requery
        MOCK_CRITIC_NO_REQUERY,    # loop 2: satisfied
    ]

    orch2 = Orchestrator()
    orch2.run("Why do LLMs hallucinate?")

# Critic should be called exactly twice
assert MockCritic.return_value.run.call_count == 2, \
    f"FAIL: expected 2 Critic calls, got {MockCritic.return_value.run.call_count}"
print("PASS ✓")

# ── Test 3: MAX_CRITIC_LOOPS enforced — critic never satisfied ────────────────
print("\nTEST 3: Offline — MAX_CRITIC_LOOPS=2 hard cap is enforced")

with patch("core.orchestrator.PlannerAgent") as MockPlanner, \
     patch("core.orchestrator.RetrieverAgent") as MockRetriever, \
     patch("core.orchestrator.SummarizerAgent") as MockSummarizer, \
     patch("core.orchestrator.CriticAgent") as MockCritic, \
     patch("core.orchestrator.WriterAgent") as MockWriter:

    MockPlanner.return_value.run.return_value    = MOCK_PLAN
    MockRetriever.return_value.run.return_value  = MOCK_RETRIEVER_OUT
    MockSummarizer.return_value.run.return_value = MOCK_SUMMARIZER_OUT
    MockWriter.return_value.run.return_value     = MOCK_REPORT

    # Critic ALWAYS wants a requery — but should be capped at MAX_CRITIC_LOOPS
    MockCritic.return_value.run.return_value = MOCK_CRITIC_REQUERY

    orch3 = Orchestrator()
    orch3.run("Why do LLMs hallucinate?")

# With MAX_CRITIC_LOOPS=2, the critic runs at most twice regardless
assert MockCritic.return_value.run.call_count <= 2, \
    f"FAIL: Critic ran {MockCritic.return_value.run.call_count} times, max is 2"
print(f"PASS ✓  (Critic ran {MockCritic.return_value.run.call_count} time(s))")

# ── Test 4: Trace structure is correct ────────────────────────────────────────
print("\nTEST 4: Offline — run trace has correct structure")

with patch("core.orchestrator.PlannerAgent") as MockPlanner, \
     patch("core.orchestrator.RetrieverAgent") as MockRetriever, \
     patch("core.orchestrator.SummarizerAgent") as MockSummarizer, \
     patch("core.orchestrator.CriticAgent") as MockCritic, \
     patch("core.orchestrator.WriterAgent") as MockWriter:

    MockPlanner.return_value.run.return_value    = MOCK_PLAN
    MockRetriever.return_value.run.return_value  = MOCK_RETRIEVER_OUT
    MockSummarizer.return_value.run.return_value = MOCK_SUMMARIZER_OUT
    MockCritic.return_value.run.return_value     = MOCK_CRITIC_NO_REQUERY
    MockWriter.return_value.run.return_value     = MOCK_REPORT

    orch4 = Orchestrator()
    orch4.run("Test question for trace?")

# Every trace entry must have the required keys
for entry in orch4.trace:
    assert "agent"     in entry, f"FAIL: trace entry missing 'agent': {entry}"
    assert "timestamp" in entry, f"FAIL: trace entry missing 'timestamp': {entry}"
    assert "input"     in entry, f"FAIL: trace entry missing 'input': {entry}"
    assert "output"    in entry, f"FAIL: trace entry missing 'output': {entry}"

# Agent names should appear in the expected order
agent_names = [e["agent"] for e in orch4.trace]
assert agent_names[0] == "Planner", "FAIL: first agent should be Planner"
assert agent_names[-1] == "Writer",  "FAIL: last agent should be Writer"
print(f"PASS ✓  (trace order: {agent_names})")

# ── Test 5: Slugify produces clean filenames ──────────────────────────────────
print("\nTEST 5: Offline — _slugify produces safe filenames")

orch5 = Orchestrator.__new__(Orchestrator)  # skip __init__
cases = [
    ("How does CRISPR work?",             "how_does_crispr_work"),
    ("What is AI?",                        "what_is_ai"),
    ("Effect of lithium mining — 2024!",   "effect_of_lithium_mining__2024"),
]
for question, expected_start in cases:
    slug = orch5._slugify(question)
    assert slug.replace("-","_").startswith(expected_start[:15]), \
        f"FAIL: slug '{slug}' doesn't start with '{expected_start[:15]}'"
    assert " "  not in slug, f"FAIL: slug contains space: '{slug}'"
    assert "?"  not in slug, f"FAIL: slug contains '?': '{slug}'"
print("PASS ✓")

print("\n" + "=" * 50)
print("All Orchestrator tests done. Step 8 complete!")