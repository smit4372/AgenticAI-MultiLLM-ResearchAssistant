"""
test_critic.py
--------------
Tests for agents/critic.py

Run from autoresearch/:
    python test_critic.py
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import patch
from agents.critic import CriticAgent
from core.schemas import validate, CriticOutput

print("=" * 50)

# ── Reusable test data ─────────────────────────────────────────────────────────
SUB_QUESTIONS = [
    "What causes LLM hallucinations?",
    "What mitigation strategies exist for hallucination?",
    "How does RLHF reduce hallucination?",
]

GOOD_FACTS = [
    {"claim": "LLMs hallucinate due to statistical token prediction.",
     "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
     "confidence": "high"},
    {"claim": "RAG grounds responses in retrieved documents to reduce hallucination.",
     "source_url": "https://en.wikipedia.org/wiki/RAG",
     "confidence": "high"},
    {"claim": "RLHF trains models using human feedback to prefer accurate answers.",
     "source_url": "https://en.wikipedia.org/wiki/RLHF",
     "confidence": "high"},
    {"claim": "Constitutional AI is another alignment technique.",
     "source_url": "https://en.wikipedia.org/wiki/Constitutional_AI",
     "confidence": "medium"},
    {"claim": "Hallucination rates vary widely across model sizes.",
     "source_url": "https://en.wikipedia.org/wiki/Hallucination_(AI)",
     "confidence": "medium"},
]

WEAK_FACTS = [
    {"claim": "No information could be retrieved for: What causes LLM hallucinations?",
     "source_url": "N/A",
     "confidence": "low"},
    {"claim": "Some vague information about mitigation.",
     "source_url": "N/A",
     "confidence": "low"},
]

# ── Test 1: Good facts → requery_needed = False ───────────────────────────────
print("TEST 1: Offline — strong facts → requery_needed should be False")

mock_no_requery = {
    "gaps":              [],
    "contradictions":    [],
    "requery_needed":    False,
    "requery_questions": [],
}

with patch("agents.critic.chat_json", return_value=mock_no_requery):
    critic  = CriticAgent()
    result  = critic.run({"sub_questions": SUB_QUESTIONS, "all_facts": GOOD_FACTS})

validate(CriticOutput, result)
assert result["requery_needed"]    == False, "FAIL: should not need requery"
assert result["requery_questions"] == [],    "FAIL: requery_questions should be empty"
print("PASS ✓")

# ── Test 2: Weak facts → requery_needed = True ────────────────────────────────
print("\nTEST 2: Offline — weak facts → requery_needed should be True")

mock_requery = {
    "gaps":              ["No good answer for what causes hallucinations."],
    "contradictions":    [],
    "requery_needed":    True,
    "requery_questions": ["What are the technical causes of LLM hallucination?"],
}

with patch("agents.critic.chat_json", return_value=mock_requery):
    critic2 = CriticAgent()
    result2 = critic2.run({"sub_questions": SUB_QUESTIONS, "all_facts": WEAK_FACTS})

validate(CriticOutput, result2)
assert result2["requery_needed"]           == True, "FAIL: should need requery"
assert len(result2["requery_questions"])   >= 1,    "FAIL: should have requery questions"
print("PASS ✓")

# ── Test 3: Zero facts → automatic requery without LLM call ──────────────────
print("\nTEST 3: Offline — zero facts → automatic requery (no LLM call)")

critic3 = CriticAgent()
result3 = critic3.run({"sub_questions": SUB_QUESTIONS, "all_facts": []})

# Should return immediately without calling the LLM
assert result3["requery_needed"]         == True,          "FAIL: should need requery"
assert len(result3["requery_questions"]) == len(SUB_QUESTIONS), "FAIL: should requery all questions"
assert len(result3["gaps"])              == len(SUB_QUESTIONS), "FAIL: all questions should be gaps"
print("PASS ✓")

# ── Test 4: Inconsistency fix — requery=False but questions present ───────────
print("\nTEST 4: Offline — requery=False with leftover questions → questions cleared")

inconsistent_response = {
    "gaps":              [],
    "contradictions":    [],
    "requery_needed":    False,
    "requery_questions": ["This should be cleared"],   # LLM made a mistake
}

with patch("agents.critic.chat_json", return_value=inconsistent_response):
    critic4 = CriticAgent()
    result4 = critic4.run({"sub_questions": SUB_QUESTIONS, "all_facts": GOOD_FACTS})

assert result4["requery_questions"] == [], "FAIL: questions should have been cleared"
print("PASS ✓")

# ── Test 5: Inconsistency fix — requery=True but no questions ────────────────
print("\nTEST 5: Offline — requery=True with no questions → falls back to sub_questions")

inconsistent_response2 = {
    "gaps":              ["Some gap"],
    "contradictions":    [],
    "requery_needed":    True,
    "requery_questions": [],    # LLM forgot to include questions
}

with patch("agents.critic.chat_json", return_value=inconsistent_response2):
    critic5 = CriticAgent()
    result5 = critic5.run({"sub_questions": SUB_QUESTIONS, "all_facts": WEAK_FACTS})

assert result5["requery_questions"] == SUB_QUESTIONS, "FAIL: should fall back to sub_questions"
print("PASS ✓")

# ── Test 6: Live API ───────────────────────────────────────────────────────────
print("\nTEST 6: Live API — real critique (needs OPENAI_API_KEY)")
try:
    critic6 = CriticAgent()
    result6 = critic6.run({
        "sub_questions": SUB_QUESTIONS,
        "all_facts": GOOD_FACTS,
    })

    print(f"  Gaps           : {len(result6['gaps'])}")
    print(f"  Contradictions : {len(result6['contradictions'])}")
    print(f"  Requery needed : {result6['requery_needed']}")
    if result6["requery_questions"]:
        print(f"  Requery Qs     : {result6['requery_questions']}")

    validate(CriticOutput, result6)
    print("PASS ✓")

except RuntimeError as e:
    print(f"SKIP — API unavailable: {e}")

print("\n" + "=" * 50)
print("All Critic tests done. Step 6 complete!")