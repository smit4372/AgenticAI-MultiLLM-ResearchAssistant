"""
agents/summarizer.py
--------------------
AGENT-03: Summarizer

Responsibility:
    Read raw text snippets retrieved for one sub-question and extract the
    3-5 most relevant facts. Each fact gets a confidence rating and a
    source URL so the Writer can cite it properly.

    The Summarizer also flags claims that look unreliable (e.g. vague,
    unverifiable, or contradicted within the same snippets).

Input  (SummarizerInput)  : { sub_question: str, snippets: list[Snippet] }
Output (SummarizerOutput) : { sub_question: str, facts: list[Fact] }
"""

import logging
import json

from core.llm import chat_json
from core.schemas import (
    SummarizerInput,
    SummarizerOutput,
    Snippet,
    Fact,
    validate,
)

logger=logging.getLogger(__name__)

# System prompt
SYSTEM_PROMPT="""
You are a careful research analyst. You will be given a research sub-question
and several raw text snippets retrieved from Wikipedia and the web.

Your job is to extract the 3 to 5 most relevant facts that directly answer
the sub-question. Be precise and faithful to the source — do not invent,
infer, or add information that is not explicitly stated in the snippets.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation,
no markdown code fences. Just raw JSON.

The JSON must follow this exact schema:
{
  "sub_question": "the sub-question you were given",
  "facts": [
    {
      "claim":      "A single factual statement in one or two sentences.",
      "source_url": "The URL of the snippet this claim came from.",
      "confidence": "high | medium | low"
    }
  ]
}

Confidence rating guide:
  "high"   — The claim is stated clearly and directly in the source text.
  "medium" — The claim is implied or partially supported by the source.
  "low"    — The claim is vague, speculative, or hard to verify from the text.

Rules:
  - Extract between 3 and 5 facts. No more, no less (unless fewer than 3
    facts are genuinely supported by the snippets — then return what you have).
  - Each fact must have a real source_url from the snippets provided.
  - Do not combine facts from multiple sources into one claim.
  - Do not add any keys beyond the two specified above (sub_question, facts).
  - If a snippet contains unreliable or contradictory information, give that
    claim a "low" confidence rating rather than omitting it.
""".strip()


class SummarizerAgent:
    """
    Extracts key facts from raw snippets for one sub-question.

    Usage
    -----
    summarizer = SummarizerAgent()
    output = summarizer.run({
        "sub_question": "How does attention work in transformers?",
        "snippets": [
            {"text": "...", "url": "https://...", "title": "..."},
        ]
    })
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        logger.info("SummarizerAgent initialised (model=%s)", model)

    def run(self, input_data: SummarizerInput) -> SummarizerOutput:
        """
        Summarise snippets for one sub-question into key facts.

        Parameters
        ----------
        input_data : SummarizerInput
            { sub_question: str, snippets: list[Snippet] }

        Returns
        -------
        SummarizerOutput
            { sub_question: str, facts: list[Fact] }

        Raises
        ------
        ValueError
            If the LLM response fails schema validation after one retry.
        """
        sub_question = input_data["sub_question"]
        snippets     = input_data["snippets"]

        logger.info("Summarizer working on: '%s'", sub_question)
        logger.info("Snippets to process: %d", len(snippets))

        #Handle empty snippets gracefully
        #If the Retriever found nothing, return an honest "no data" fact
        #rather than asking the LLM to summarise nothing.
        if not snippets:
            logger.warning("No snippets provided for: '%s'", sub_question)
            return SummarizerOutput(
                sub_question=sub_question,
                facts=[
                    Fact(
                        claim      = f"No information could be retrieved for: {sub_question}",
                        source_url = "N/A",
                        confidence = "low",
                    )
                ],
            )

        # Build the user message
        user_message = self._build_user_message(sub_question, snippets)

        # Call the LLM
        logger.info("Calling LLM to extract facts...")
        result = chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self.model,
            temperature=0.1,    # Very low — we want faithful extraction, not creativity
            max_tokens=1500,
        )

        # Validate schema
        validate(SummarizerOutput, result)

        # Validate each individual fact has required keys
        self._validate_facts(result["facts"])

        #Ensure sub_question is echoed correctly
        #LLMs sometimes paraphrase the sub_question. We enforce the original.
        result["sub_question"] = sub_question

        logger.info(
            "Summarizer extracted %d fact(s) for '%s'",
            len(result["facts"]),
            sub_question,
        )

        return result   # type: ignore[return-value]

    #Private helpers

    def _build_user_message(self, sub_question: str, snippets: list[Snippet]) -> str:
        """
        Format the sub-question and snippets into a clean prompt for the LLM.

        Each snippet is numbered and includes its title, URL, and text body
        so the LLM can cite sources accurately.
        """
        # Format each snippet as a numbered block
        formatted_snippets = []
        for i, snippet in enumerate(snippets, 1):
            block = (
                f"[Snippet {i}]\n"
                f"Title : {snippet['title']}\n"
                f"URL   : {snippet['url']}\n"
                f"Text  : {snippet['text']}\n"
            )
            formatted_snippets.append(block)

        snippets_text = "\n---\n".join(formatted_snippets)

        return f"""
Sub-question to answer:
\"\"\"{sub_question}\"\"\"

Retrieved snippets:
---
{snippets_text}
---

Extract 3-5 key facts from the snippets above that directly answer the sub-question.
Respond with ONLY the JSON object.
""".strip()

    def _validate_facts(self, facts: list) -> None:
        """
        Check that each fact in the list has the required keys and that
        confidence is one of the allowed values.

        Raises
        ------
        ValueError
            If any fact is malformed.
        """
        required_keys      = {"claim", "source_url", "confidence"}
        allowed_confidence = {"high", "medium", "low"}

        for i, fact in enumerate(facts):
            missing = required_keys - set(fact.keys())
            if missing:
                raise ValueError(
                    f"Fact {i} is missing required keys: {missing}\n"
                    f"Fact content: {fact}"
                )
            if fact["confidence"] not in allowed_confidence:
                logger.warning(
                    "Fact %d has unexpected confidence value '%s' — normalising to 'medium'",
                    i, fact["confidence"],
                )
                # Normalise unexpected values rather than crashing
                fact["confidence"] = "medium"

