"""
agents/critic.py
----------------
AGENT-04: Critic

Responsibility:
    Review ALL facts collected across ALL sub-questions in one pass.
    Identify:
      - Gaps: sub-questions that were poorly or not answered
      - Contradictions: facts that conflict with each other
      - Whether another retrieval loop is warranted (max 2 loops enforced
        by the Orchestrator, not here)

    The Critic is the only agent that sees the full picture — every other
    agent only processes one sub-question at a time.

Input  (CriticInput)  : { sub_questions: list[str], all_facts: list[Fact] }
Output (CriticOutput) : { gaps[], contradictions[], requery_needed: bool,
                          requery_questions[] }
"""

import logging

from core.llm import chat_json
from core.schemas import CriticInput, CriticOutput, Fact, validate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are a rigorous research critic. You will be given a list of research
sub-questions and all the facts that have been collected so far.

Your job is to evaluate the quality and completeness of the research by:
  1. Identifying GAPS — sub-questions that are poorly answered or not answered
     at all (e.g. only "low" confidence facts, or no facts at all).
  2. Identifying CONTRADICTIONS — facts that directly conflict with each other.
  3. Deciding whether another retrieval loop is needed to fill critical gaps.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation,
no markdown code fences. Just raw JSON.

The JSON must follow this exact schema:
{
  "gaps": [
    "Description of what information is missing or inadequate."
  ],
  "contradictions": [
    "Description of two facts that conflict with each other, naming both claims."
  ],
  "requery_needed": true or false,
  "requery_questions": [
    "A refined or new search question to fill a specific gap."
  ]
}

Rules:
  - gaps should describe WHAT is missing, not just which sub-question number.
  - contradictions should name the specific conflicting claims, not just say
    "there is a contradiction".
  - Set requery_needed to true ONLY if there are critical gaps that would
    prevent writing a useful research report.
  - requery_questions must be non-empty if requery_needed is true.
  - requery_questions must be empty ([]) if requery_needed is false.
  - Be decisive — if the facts are mostly adequate, set requery_needed to false.
  - Do not add any keys beyond the four specified above.
""".strip()


class CriticAgent:
    """
    Reviews all collected facts for gaps and contradictions, and decides
    whether another retrieval loop is needed.

    Usage
    -----
    critic = CriticAgent()
    output = critic.run({
        "sub_questions": ["Q1", "Q2", "Q3"],
        "all_facts": [fact1, fact2, ...]
    })
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        logger.info("CriticAgent initialised (model=%s)", model)

    def run(self, input_data: CriticInput) -> CriticOutput:
        """
        Review all facts and decide if more research is needed.

        Parameters
        ----------
        input_data : CriticInput
            { sub_questions: list[str], all_facts: list[Fact] }

        Returns
        -------
        CriticOutput
            { gaps[], contradictions[], requery_needed, requery_questions[] }

        Raises
        ------
        ValueError
            If the LLM response fails schema validation after one retry.
        """
        sub_questions = input_data["sub_questions"]
        all_facts     = input_data["all_facts"]

        logger.info(
            "Critic reviewing %d fact(s) across %d sub-question(s)",
            len(all_facts),
            len(sub_questions),
        )

        # Edge case: no facts at all
        # If retrieval completely failed, telling the Orchestrator to requery
        # everything rather than sending an empty report to the Writer.
        if not all_facts:
            logger.warning("Critic received zero facts — requesting full requery")
            return CriticOutput(
                gaps              = [f"No facts retrieved for: {q}" for q in sub_questions],
                contradictions    = [],
                requery_needed    = True,
                requery_questions = sub_questions,   # requery all original questions
            )

        # Build the user message
        user_message = self._build_user_message(sub_questions, all_facts)

        # Call the LLM
        logger.info("Calling LLM for critical review...")
        result = chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self.model,
            temperature=0.2,
            max_tokens=1000,
        )

        # Validate schema
        validate(CriticOutput, result)

        # Enforce requery_questions consistency
        # If the LLM says requery_needed=False but still put questions in the
        # list (or vic versa), will fix it here so the Orchestrator doesn't
        # get confused.
        if not result["requery_needed"]:
            if result["requery_questions"]:
                logger.warning(
                    "Critic set requery_needed=False but provided requery_questions. "
                    "Clearing requery_questions."
                )
                result["requery_questions"] = []
        else:
            if not result["requery_questions"]:
                logger.warning(
                    "Critic set requery_needed=True but provided no requery_questions. "
                    "Falling back to original sub-questions."
                )
                result["requery_questions"] = sub_questions

        logger.info(
            "Critic verdict — gaps: %d | contradictions: %d | requery: %s",
            len(result["gaps"]),
            len(result["contradictions"]),
            result["requery_needed"],
        )

        return result   # type: ignore[return-value]

    # Private helpers

    def _build_user_message(
        self,
        sub_questions: list[str],
        all_facts: list[Fact],
    ) -> str:
        """
        Format the sub-questions and all collected facts into a prompt.

        Facts are grouped by sub-question so the Critic can easily see
        which questions are well-covered and which are not.
        """
        # Group facts by sub_question for readability
        facts_by_question: dict[str, list[Fact]] = {}

        for fact in all_facts:
            # Find which sub-question this fact belongs to by checking
            # if we have a direct match, or use "Other" as a catch-all
            matched = False
            for q in sub_questions:
                if q not in facts_by_question:
                    facts_by_question[q] = []
            # Since facts don't store which sub-question they came from directly,
            # we add them to an "All Facts" group for the critic to assess
            facts_by_question.setdefault("__all__", []).append(fact)

        # Format sub-questions
        sq_block = "\n".join(
            f"  {i+1}. {q}" for i, q in enumerate(sub_questions)
        )

        # Format all facts with confidence indicator
        facts_block_lines = []
        for i, fact in enumerate(all_facts, 1):
            confidence_tag = f"[{fact['confidence'].upper()}]"
            facts_block_lines.append(
                f"  Fact {i:02d} {confidence_tag}\n"
                f"    Claim  : {fact['claim']}\n"
                f"    Source : {fact['source_url']}\n"
            )
        facts_block = "\n".join(facts_block_lines)

        # Summary stats — helps the LLM quickly assess coverage
        high_count   = sum(1 for f in all_facts if f["confidence"] == "high")
        medium_count = sum(1 for f in all_facts if f["confidence"] == "medium")
        low_count    = sum(1 for f in all_facts if f["confidence"] == "low")

        return f"""
Research sub-questions ({len(sub_questions)} total):
{sq_block}

Collected facts ({len(all_facts)} total | {high_count} high / {medium_count} medium / {low_count} low confidence):
{facts_block}

Review the facts above against the sub-questions.
Identify gaps, contradictions, and decide if more research is needed.
Respond with ONLY the JSON object.
""".strip()

