"""
agents/planner.py
-----------------
AGENT-01: Planner

Responsibility:
    Receive the raw research question from the user and decompose it into
    3-6 focused sub-questions, each with recommended search keywords, plus
    an overall scope statement.

Input  (PlannerInput)  : { question: str }
Output (PlannerOutput) : { sub_questions[], keywords[], scope }
"""

import logging
from core.llm import chat_json
from core.schemas import PlannerInput, PlannerOutput, validate

logger = logging.getLogger(__name__)

# System prompt
# This is the core instruction set for the Planner.
# Rules:
#   - Must respond ONLY in valid JSON (no prose, no markdown fences)
#   - Must follow the exact schema the Orchestrator expects
#   - sub_questions and keywords must be parallel lists (same length)
SYSTEM_PROMPT = """
You are a research planning assistant. Your job is to decompose a broad research
question into focused sub-questions that can be answered independently.

You MUST respond with ONLY a valid JSON object — no preamble, no explanation,
no markdown code fences. Just raw JSON.

The JSON must follow this exact schema:
{
  "sub_questions": [
    "Focused sub-question 1?",
    "Focused sub-question 2?",
    ...
  ],
  "keywords": [
    ["keyword1", "keyword2", "keyword3"],
    ["keyword4", "keyword5"],
    ...
  ],
  "scope": "One sentence describing what this research covers and what it excludes."
}

Rules:
- Produce between 3 and 6 sub-questions.
- sub_questions and keywords must be the same length (keywords[i] belongs to sub_questions[i]).
- Each keyword list should have 2-4 short search terms optimised for Wikipedia or web search.
- The scope statement should be one sentence, specific and honest about limitations.
- Do not number the sub-questions inside the string — just write the question.
- Do not add any keys beyond the three specified above.
""".strip()


class PlannerAgent:
    """
    Decomposes a research question into a structured research plan.

    Usage
    -----
    planner = PlannerAgent()
    output  = planner.run({"question": "How does CRISPR work?"})
    # output is a validated PlannerOutput dict
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        logger.info("PlannerAgent initialised (model=%s)", model)

    def run(self, input_data: PlannerInput) -> PlannerOutput:
        """
        Run the Planner on a research question.

        Parameters
        ----------
        input_data : PlannerInput
            Dict with a single key: "question" (str).

        Returns
        -------
        PlannerOutput
            Validated dict with sub_questions, keywords, and scope.

        Raises
        ------
        ValueError
            If the LLM returns a response that doesn't match the schema
            even after one retry.
        """
        question = input_data["question"]
        logger.info("Planner received question: %s", question)

        # Build the user message
        # We give the model the question in a clear, structured way so it
        # understands exactly what to decompose.
        user_message = f"""
Research question to decompose:
\"\"\"{question}\"\"\"

Remember: respond with ONLY the JSON object. No other text.
""".strip()

        # Call the LLM (with automatic JSON parsing + 1 retry)
        logger.info("Calling LLM for research plan...")
        result = chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self.model,
            temperature=0.3,   # Slightly higher than default — planning benefits
            max_tokens=1000,   # from a little more creative breadth
        )

        # Validate the schema before returning
        # This will raise ValueError if any required key is missing,
        # or preventing corrupt data from silently propagating downstream.
        validate(PlannerOutput, result)

        # Sanity check: parallel list lengths
        if len(result["sub_questions"]) != len(result["keywords"]):
            raise ValueError(
                f"Planner returned mismatched list lengths: "
                f"{len(result['sub_questions'])} sub_questions vs "
                f"{len(result['keywords'])} keyword lists."
            )

        logger.info(
            "Planner produced %d sub-questions. Scope: %s",
            len(result["sub_questions"]),
            result["scope"],
        )

        return result#type:ignore[return-value]