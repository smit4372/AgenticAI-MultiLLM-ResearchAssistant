"""
agents/writer.py
----------------
AGENT-05: Writer

Responsibility:
    Synthesise all verified facts into a structured Markdown research report.
    This is the only agent whose output is a plain string (Markdown), not JSON.

    STRICT RULE: The Writer must not add any information beyond what is
    present in the facts. No hallucination, no general knowledge fill-ins.

Input  (WriterInput) : { question, scope, sub_questions[], all_facts[] }
Output               : str  (Markdown report, saved to disk by Orchestrator)
"""

import logging
from core.llm import chat
from core.schemas import WriterInput

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a precise research report writer. You will be given a research
question, a scope statement, a list of sub-questions, and a set of
verified facts with source URLs and confidence ratings.

Your job is to write a structured Markdown research report based ONLY
on the facts provided. Do not add any information, background knowledge,
or context that is not present in the facts list.

The report must follow this exact structure:

# [Research Question as Title]

## Executive Summary
2-3 sentence overview of the key findings. Cite sources inline as [Source](url).

## Scope
One paragraph stating what this report covers and what it excludes
(use the scope statement provided).

## Findings

### [Sub-question 1]
A paragraph synthesising the facts relevant to this sub-question.
Cite every claim inline as [Source](url).

### [Sub-question 2]
...and so on for each sub-question.

## Limitations
A bullet-point list covering:
- Any sub-questions with only low-confidence facts
- Any gaps identified in the research
- The fact that sources are limited to Wikipedia and DuckDuckGo

## References
A numbered list of all unique source URLs cited in the report.

Rules:
- Only use information from the provided facts. Do not invent or infer.
- Every factual claim must have an inline citation [Source](url).
- Low-confidence facts must be clearly flagged: write "(low confidence)" after the claim.
- Write in clear, formal, academic English.
- Do not include a markdown code fence around your output — just write the report directly.
""".strip()


class WriterAgent:
    """
    Writes the final Markdown research report from verified facts.

    Usage
    -----
    writer = WriterAgent()
    report = writer.run({
        "question":      "How do LLMs hallucinate?",
        "scope":         "Covers transformer-based LLMs only.",
        "sub_questions": ["Q1", "Q2", "Q3"],
        "all_facts":     [fact1, fact2, ...]
    })
    # report is a Markdown string ready to save to disk
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        logger.info("WriterAgent initialised (model=%s)", model)

    def run(self, input_data: WriterInput) -> str:
        """
        Generate the final Markdown research report.

        Parameters
        ----------
        input_data : WriterInput
            { question, scope, sub_questions[], all_facts[] }

        Returns
        -------
        str
            A complete Markdown research report.

        Raises
        ------
        RuntimeError
            If the LLM call fails.
        """
        question      = input_data["question"]
        scope         = input_data["scope"]
        sub_questions = input_data["sub_questions"]
        all_facts     = input_data["all_facts"]

        logger.info("Writer generating report for: '%s'", question)
        logger.info(
            "Writing from %d fact(s) across %d sub-question(s)",
            len(all_facts),
            len(sub_questions),
        )

        #Build the user message
        user_message = self._build_user_message(
            question, scope, sub_questions, all_facts
        )

        #Call the LLM
        # Note: we use chat() not chat_json() — the output is Markdown, not JSON.
        # We also use a higher token limit since reports can be long.
        logger.info("Calling LLM to write report...")
        report = chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self.model,
            temperature=0.3,    # Slightly more fluent prose than pure extraction
            max_tokens=3000,
        )

        #Basic sanity check
        # Make sure we got something resembling a report
        if len(report.strip()) < 100:
            raise ValueError(
                f"Writer returned suspiciously short output ({len(report)} chars). "
                f"Raw output: {report}"
            )

        logger.info("Writer produced report (%d chars)", len(report))
        return report

    #Private helpers

    def _build_user_message(
        self,
        question:      str,
        scope:         str,
        sub_questions: list[str],
        all_facts:     list,
    ) -> str:
        """
        Format all inputs into a structured prompt for the Writer.

        Facts are grouped under their relevant sub-question where possible,
        or listed under "General Facts" if they can't be matched.
        """
        # Format sub-questions
        sq_block = "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(sub_questions)
        )

        # Format all facts with confidence tags
        facts_lines = []
        for i, fact in enumerate(all_facts, 1):
            confidence_tag = f"[{fact['confidence'].upper()}]"
            facts_lines.append(
                f"  Fact {i:02d} {confidence_tag}\n"
                f"    Claim  : {fact['claim']}\n"
                f"    Source : {fact['source_url']}\n"
            )
        facts_block = "\n".join(facts_lines)

        return f"""
Research question:
\"\"\"{question}\"\"\"

Scope statement:
\"\"\"{scope}\"\"\"

Sub-questions to address ({len(sub_questions)} total):
{sq_block}

Verified facts to use ({len(all_facts)} total):
{facts_block}

Write the complete Markdown research report now.
Use ONLY the facts above. Cite every claim. Flag low-confidence facts.
""".strip()