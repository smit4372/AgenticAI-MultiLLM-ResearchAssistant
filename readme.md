# AutoResearch

A multi-agent pipeline that autonomously decomposes a research question,
retrieves information from free web sources, critically evaluates findings,
and produces a structured Markdown research report.

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone / download the project
cd autoresearch/

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
# Single question
python main.py "How does CRISPR work?"

# With verbose logs (see every agent's steps)
python main.py "How does CRISPR work?" --verbose

# Run all 3 sample questions from the assignment
python main.py --run-samples
```

Output files are saved to the `outputs/` directory:
- `outputs/<question-slug>.md` — the research report
- `outputs/run_trace_<question-slug>.json` — full agent message log

---

## Architecture

```
User question (str)
      │
      ▼
┌─────────────┐
│  AGENT-01   │  Planner — decomposes question into 3-6 sub-questions
│  Planner    │  Output: { sub_questions[], keywords[], scope }
└──────┬──────┘
       │
       │  ┌─────────────────────────────────────────────┐
       │  │  Retrieval loop (max 2 iterations)          │
       ▼  │                                             │
┌─────────────┐                                         │
│  AGENT-02   │  Retriever — fetches Wikipedia + DDG    │
│  Retriever  │  Output: { snippets[{text,url,title}] } │
└──────┬──────┘                                         │
       │                                                │
       ▼                                                │
┌─────────────┐                                         │
│  AGENT-03   │  Summarizer — extracts 3-5 key facts    │
│  Summarizer │  Output: { facts[{claim,url,conf}] }    │
└──────┬──────┘                                         │
       │  (runs once per sub-question)                  │
       ▼                                                │
┌─────────────┐                                         │
│  AGENT-04   │  Critic — reviews ALL facts             │
│  Critic     │  Output: { gaps, contradictions,        │
└──────┬──────┘           requery_needed,               │
       │                  requery_questions }           │
       │  requery_needed=True ───────────────────────── ┘
       │  requery_needed=False (or max loops hit)
       ▼
┌─────────────┐
│  AGENT-05   │  Writer — writes final Markdown report
│  Writer     │  Output: Markdown string (saved to disk)
└─────────────┘
```

### Agent Communication Protocol

All inter-agent messages are serialisable JSON, defined as `TypedDict`
classes in `core/schemas.py`. The Orchestrator calls `validate()` on
every message before passing it to the next agent.

The Writer is the only exception — its output is a plain Markdown string.

---

## Project Structure

```
autoresearch/
├── agents/
│   ├── planner.py       # Agent-01: decomposes question
│   ├── retriever.py     # Agent-02: fetches Wikipedia + DuckDuckGo
│   ├── summarizer.py    # Agent-03: extracts facts from snippets
│   ├── critic.py        # Agent-04: reviews facts, decides re-loop
│   └── writer.py        # Agent-05: writes final report
├── core/
│   ├── llm.py           # OpenAI wrapper + safe_parse()
│   ├── schemas.py       # TypedDicts + validate()
│   └── orchestrator.py  # Main pipeline loop
├── outputs/             # Reports + run traces saved here
├── main.py              # CLI entry point
├── requirements.txt
├── .env
└── README.md
```

---

## Agent Design Decisions

### Why TypedDict instead of dataclasses or Pydantic?
TypedDicts are lightweight, require no imports at runtime, and produce
plain dicts that serialise directly to JSON — which is exactly what agents
pass to each other. Pydantic would add validation power but also
significant complexity and an extra dependency.

### Why is the Retriever the only agent with no LLM call?
Retrieval is deterministic — given a keyword, Wikipedia returns the same
article every time. Adding an LLM to "decide what to search for" would
introduce non-determinism and cost without improving accuracy. The Planner
already produces targeted keywords.

### Why does the Critic not enforce the 2-loop limit itself?
Separation of concerns. The Critic answers: *"is more research needed?"*
The Orchestrator answers: *"are we allowed to do more?"* Mixing those
decisions would make both harder to test independently.

### Why `temperature=0.1` for the Summarizer?
The Summarizer's job is faithful extraction — it must report what the
source says, not what it "thinks" the source means. Low temperature
minimises the chance of the LLM paraphrasing facts into inaccuracies.

### safe_parse() retry strategy
If the LLM returns malformed JSON, `safe_parse()` retries once with
`temperature=0.0` and an explicit reminder to return only JSON. This
handles the most common failure mode (markdown fences, preamble text)
without infinite loops.

---

## Known Limitations

1. **Wikipedia + DuckDuckGo only** — Sources are limited to free APIs.
   Paywalled or niche academic sources are not accessible.

2. **DuckDuckGo Instant Answers are sparse** — The DDG API only returns
   an abstract for well-known topics. Many queries return nothing from DDG
   and rely entirely on Wikipedia.

3. **Facts not attributed to sub-questions** — The Critic receives all
   facts as a flat list, without labels indicating which sub-question each
   fact came from. This can make gap detection less precise.

4. **No cross-run memory** — Each run is fully independent. The system
   cannot build on previous research runs.

5. **Token limits on snippets** — Wikipedia extracts are capped at 2000
   characters to stay within model context limits. Long articles are
   truncated and nuanced details may be lost.

6. **Max 2 retrieval loops** — Cost control limits depth. Complex topics
   may need more iterations than the hard cap allows.

---

## Estimated Cost Per Run

Using `gpt-4o-mini` for all agents (default):

| Agent | Avg calls | Tokens/call | Subtotal |
|---|---|---|---|
| Planner | 1 | ~500 | ~500 |
| Summarizer | 4–6 | ~1500 | ~7500 |
| Critic | 1–2 | ~1000 | ~1500 |
| Writer | 1 | ~3000 | ~3000 |
| **Total** | | | **~12500 tokens** |

At `gpt-4o-mini` pricing (~$0.15 / 1M input tokens, ~$0.60 / 1M output),
a typical run costs approximately **$0.01–$0.02**.

---


Each agent has its own test file. All offline tests run without an API key:

```bash
python test_llm.py
python test_schemas.py
python test_planner.py
python test_retriever.py
python test_summarizer.py
python test_critic.py
python test_writer.py
python test_orchestrator.py
```

To run all tests at once:
```bash
for f in test_*.py; do echo "--- $f ---" && python $f; done
```