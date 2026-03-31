"""
core/schemas.py
---------------
Defines the input/output message schemas for every agent in the pipeline.

We use TypedDict so that:
  1. Each agent's expected I/O is self-documenting
  2. The Orchestrator can validate messages before passing them on
  3. Your editor can provide autocomplete and type checking

Pipeline data flow:
  User question
    → PlannerOutput
    → RetrieverInput  → RetrieverOutput      (once per sub-question)
    → SummarizerInput → SummarizerOutput     (once per sub-question)
    → CriticInput     → CriticOutput
    → WriterInput     → str (Markdown report)
"""

from typing import TypedDict


#SHARED PRIMITIVE TYPES
#These small TypedDicts are reused as list items inside larger messages.

class Snippet(TypedDict):
    """A single raw text chunk returned by the Retriever."""
    text:  str   # The raw passage retrieved from Wikipedia / DuckDuckGo
    url:   str   # Source URL (used for citations later)
    title: str   # Page or result title


class Fact(TypedDict):
    """A single verified fact extracted by the Summarizer."""
    claim:      str    # The factual statement in plain English
    source_url: str    # Where this claim came from
    confidence: str    # "high" | "medium" | "low"  (Summarizer's assessment)


#AGENT-01 — PLANNER

class PlannerInput(TypedDict):
    """What the Planner receives — just the raw research question."""
    question: str


class PlannerOutput(TypedDict):
    """
    What the Planner produces.

    sub_questions : 3–6 focused questions that together cover the main topic.
    keywords      : Parallel list — keywords[i] belongs to sub_questions[i].
    scope         : One-sentence statement of what is and isn't covered.
    """
    sub_questions: list[str]
    keywords:      list[list[str]]   # keywords[i] = list of keywords for sub_questions[i]
    scope:         str


#AGENT-02 — RETRIEVER

class RetrieverInput(TypedDict):
    """What the Retriever receives for one sub-question."""
    sub_question: str
    keywords:     list[str]   # Keywords to search with


class RetrieverOutput(TypedDict):
    """What the Retriever returns — raw text passages with sources."""
    sub_question: str          # Echoed back so the Summarizer knows the context
    snippets:     list[Snippet]


#AGENT-03 — SUMMARIZER

class SummarizerInput(TypedDict):
    """What the Summarizer receives — a sub-question and its raw snippets."""
    sub_question: str
    snippets:     list[Snippet]


class SummarizerOutput(TypedDict):
    """
    What the Summarizer produces — 3–5 key facts for one sub-question.
    The Summarizer also flags low-confidence claims so the Critic knows
    which facts to scrutinise.
    """
    sub_question: str         # Echoed back for traceability
    facts:        list[Fact]


#AGENT-04 — CRITIC

class CriticInput(TypedDict):
    """
    What the Critic receives — everything collected so far.
    The Critic looks across ALL sub-questions at once to find
    contradictions and gaps.
    """
    sub_questions: list[str]
    all_facts:     list[Fact]


class CriticOutput(TypedDict):
    """
    What the Critic produces.

    gaps              : Sub-questions that are inadequately answered.
    contradictions    : Pairs or groups of conflicting facts (as plain strings).
    requery_needed    : True if another retrieval loop should run.
    requery_questions : New or refined questions to pass back to the Retriever.
                        Empty list if requery_needed is False.
    """
    gaps:               list[str]
    contradictions:     list[str]
    requery_needed:     bool
    requery_questions:  list[str]


#AGENT-05 — WRITER

class WriterInput(TypedDict):
    """
    What the Writer receives — the full set of verified facts plus context.
    The Writer must NOT add any information beyond what is present here.
    """
    question:      str          # Original user question (for the report title)
    scope:         str          # From the Planner — frames the report
    sub_questions: list[str]
    all_facts:     list[Fact]


#Writer output is a plain str (Markdown), not a TypedDict.
#The Orchestrator saves it directly to disk.


#SCHEMA VALIDATOR

#Maps each schema class to the keys that MUST be present in a valid message.
#Used by the Orchestrator before passing messages between agents.
REQUIRED_KEYS: dict[type, list[str]] = {
    PlannerOutput:   ["sub_questions", "keywords", "scope"],
    RetrieverOutput: ["sub_question", "snippets"],
    SummarizerOutput:["sub_question", "facts"],
    CriticOutput:    ["gaps", "contradictions", "requery_needed", "requery_questions"],
}


def validate(schema_class: type, data: dict) -> None:
    """
    Check that `data` contains all required keys for `schema_class`.

    Raises
    ------
    ValueError
        With a descriptive message listing which keys are missing.

    Example
    -------
    >>> validate(PlannerOutput, {"sub_questions": [...], "keywords": [...], "scope": "..."})
    # No error — all keys present

    >>> validate(PlannerOutput, {"sub_questions": [...]})
    # ValueError: PlannerOutput is missing required keys: ['keywords', 'scope']
    """
    required = REQUIRED_KEYS.get(schema_class, [])
    missing  = [key for key in required if key not in data]

    if missing:
        raise ValueError(
            f"{schema_class.__name__} is missing required keys: {missing}\n"
            f"Received keys: {list(data.keys())}"
        )

        