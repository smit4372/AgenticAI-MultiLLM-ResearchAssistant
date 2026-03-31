"""
core/orchestrator.py
--------------------
The Orchestrator ties all five agents together into a single pipeline.

Flow:
  1. Planner        — decompose the question
  2. Retriever      — fetch snippets per sub-question       ┐
  3. Summarizer     — extract facts per sub-question        │ loop
  4. Critic         — review all facts, decide to re-loop   ┘ (max 2x)
  5. Writer         — produce the final Markdown report

The Orchestrator also:
  - Validates every inter-agent message via schemas.validate()
  - Logs a full run trace (every input/output) to outputs/run_trace_*.json
  - Saves the final report to outputs/<slug>.md
  - Enforces the MAX_CRITIC_LOOPS = 2 hard limit
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

from agents.planner    import PlannerAgent
from agents.retriever  import RetrieverAgent
from agents.summarizer import SummarizerAgent
from agents.critic     import CriticAgent
from agents.writer     import WriterAgent
from core.schemas      import (
    PlannerOutput,
    RetrieverOutput,
    SummarizerOutput,
    CriticOutput,
    validate,
)

logger = logging.getLogger(__name__)

#Constants
MAX_CRITIC_LOOPS = 2          # Hard cap — never run more than 2 retrieval loops
OUTPUTS_DIR      = Path("outputs")


class Orchestrator:
    """
    Runs the full AutoResearch pipeline for a given research question.

    Usage
    -----
    orch   = Orchestrator()
    report = orch.run("How does CRISPR work?")
    # Report saved to outputs/<slug>.md
    # Run trace saved to outputs/run_trace_<slug>.json
    """

    def __init__(self, model: str = "gpt-4o-mini", writer_model: str = "gpt-4o-mini"):
        """
        Parameters
        ----------
        model        : Model for Planner, Summarizer, Critic (default gpt-4o-mini)
        writer_model : Model for Writer — can be upgraded to gpt-4o for better prose
        """
        self.planner    = PlannerAgent(model=model)
        self.retriever  = RetrieverAgent()               # no LLM — no model param
        self.summarizer = SummarizerAgent(model=model)
        self.critic     = CriticAgent(model=model)
        self.writer     = WriterAgent(model=writer_model)

        # Run trace accumulates every agent's input and output for submission
        self.trace: list[dict] = []

        OUTPUTS_DIR.mkdir(exist_ok=True)
        logger.info("Orchestrator ready (model=%s, writer_model=%s)", model, writer_model)

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def run(self, question: str) -> str:
        """
        Run the full pipeline for a research question.

        Parameters
        ----------
        question : str
            The raw research question from the user.

        Returns
        -------
        str
            Path to the saved Markdown report file.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Pipeline started for: '%s'", question)
        logger.info("=" * 60)

        # Reset trace for this run
        self.trace = []
        slug = self._slugify(question)

        #STEP 1: Planner
        plan = self._run_planner(question)

        # Accumulate all facts across loops
        all_facts = []

        #STEPS 2-4: Retrieve → Summarise → Critique (up to MAX loops)
        #First loop uses the Planner's sub-questions.
        #Subsequent loops use the Critic's requery_questions.
        current_questions = list(zip(plan["sub_questions"], plan["keywords"]))

        for loop_num in range(1, MAX_CRITIC_LOOPS + 1):
            logger.info("-" * 40)
            logger.info("Starting retrieval loop %d / %d", loop_num, MAX_CRITIC_LOOPS)

            #STEP 2: Retrieve snippets for each sub-question
            #STEP 3: Summarise snippets into facts
            loop_facts = self._run_retrieve_and_summarise(current_questions)
            all_facts.extend(loop_facts)

            logger.info(
                "Loop %d collected %d new facts (total: %d)",
                loop_num, len(loop_facts), len(all_facts),
            )

            #STEP 4: Critic reviews ALL facts collected so far
            critic_output = self._run_critic(plan["sub_questions"], all_facts)

            #Decision: loop again or move on?
            if not critic_output["requery_needed"]:
                logger.info("Critic satisfied — proceeding to Writer")
                break

            if loop_num == MAX_CRITIC_LOOPS:
                logger.info(
                    "Reached MAX_CRITIC_LOOPS (%d) — proceeding to Writer regardless",
                    MAX_CRITIC_LOOPS,
                )
                break

            # Prepare next loop with the Critic's requery questions
            # We use empty keyword lists — the Retriever will search by
            # the question text directly
            requery_qs = critic_output["requery_questions"]
            current_questions = [(q, [q]) for q in requery_qs]
            logger.info(
                "Critic requested requery — %d new question(s) for loop %d",
                len(requery_qs), loop_num + 1,
            )

        #STEP 5: Writer
        report_md = self._run_writer(question, plan, all_facts)

        #Save outputs
        report_path = self._save_report(slug, report_md)
        trace_path  = self._save_trace(slug, question)

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1fs", elapsed)
        logger.info("Report : %s", report_path)
        logger.info("Trace  : %s", trace_path)
        logger.info("=" * 60)

        return str(report_path)

    #PRIVATE STEP RUNNERS
    #Each method runs one agent, logs the trace, and validates the output.

    def _run_planner(self, question: str) -> PlannerOutput:
        """Run the Planner and record the trace."""
        logger.info("[PLANNER] Decomposing question...")

        agent_input = {"question": question}
        output      = self.planner.run(agent_input)

        validate(PlannerOutput, output)
        self._record_trace("Planner", agent_input, output)

        logger.info(
            "[PLANNER] Done — %d sub-questions | scope: %s",
            len(output["sub_questions"]),
            output["scope"],
        )
        return output

    def _run_retrieve_and_summarise(
        self,
        questions_and_keywords: list[tuple[str, list[str]]],
    ) -> list[dict]:
        """
        Run Retriever + Summarizer for each (sub-question, keywords) pair.
        Returns a flat list of all facts collected in this loop.
        """
        loop_facts = []

        for sub_question, keywords in questions_and_keywords:
            # ── Retriever ──────────────────────────────────────────────────────
            logger.info("[RETRIEVER] Fetching: '%s'", sub_question)
            retriever_input = {"sub_question": sub_question, "keywords": keywords}
            retriever_out   = self.retriever.run(retriever_input)

            validate(RetrieverOutput, retriever_out)
            self._record_trace("Retriever", retriever_input, retriever_out)

            logger.info(
                "[RETRIEVER] Got %d snippet(s) for '%s'",
                len(retriever_out["snippets"]), sub_question,
            )

            #Summarizer
            logger.info("[SUMMARIZER] Extracting facts for: '%s'", sub_question)
            summarizer_input = {
                "sub_question": sub_question,
                "snippets":     retriever_out["snippets"],
            }
            summarizer_out = self.summarizer.run(summarizer_input)

            validate(SummarizerOutput, summarizer_out)
            self._record_trace("Summarizer", summarizer_input, summarizer_out)

            logger.info(
                "[SUMMARIZER] Extracted %d fact(s) for '%s'",
                len(summarizer_out["facts"]), sub_question,
            )

            loop_facts.extend(summarizer_out["facts"])

        return loop_facts

    def _run_critic(self, sub_questions: list[str], all_facts: list) -> CriticOutput:
        """Run the Critic and record the trace."""
        logger.info("[CRITIC] Reviewing %d facts...", len(all_facts))

        critic_input = {"sub_questions": sub_questions, "all_facts": all_facts}
        critic_out   = self.critic.run(critic_input)

        validate(CriticOutput, critic_out)
        self._record_trace("Critic", critic_input, critic_out)

        logger.info(
            "[CRITIC] gaps=%d | contradictions=%d | requery=%s",
            len(critic_out["gaps"]),
            len(critic_out["contradictions"]),
            critic_out["requery_needed"],
        )
        return critic_out

    def _run_writer(
        self,
        question:  str,
        plan:      PlannerOutput,
        all_facts: list,
    ) -> str:
        """Run the Writer and record the trace."""
        logger.info("[WRITER] Writing final report (%d facts)...", len(all_facts))

        writer_input = {
            "question":      question,
            "scope":         plan["scope"],
            "sub_questions": plan["sub_questions"],
            "all_facts":     all_facts,
        }
        report = self.writer.run(writer_input)
        self._record_trace("Writer", writer_input, {"report_length": len(report)})

        logger.info("[WRITER] Report written (%d chars)", len(report))
        return report

    #TRACE + FILE HELPERS

    def _record_trace(self, agent_name: str, input_data: dict, output_data: dict) -> None:
        """Append one agent's input/output to the run trace."""
        self.trace.append({
            "agent":     agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "input":     input_data,
            "output":    output_data,
        })

    def _save_report(self, slug: str, report_md: str) -> Path:
        """Save the Markdown report to outputs/<slug>.md"""
        path = OUTPUTS_DIR / f"{slug}.md"
        path.write_text(report_md, encoding="utf-8")
        return path

    def _save_trace(self, slug: str, question: str) -> Path:
        """Save the full run trace to outputs/run_trace_<slug>.json"""
        path = OUTPUTS_DIR / f"run_trace_{slug}.json"
        trace_data = {
            "question":   question,
            "run_date":   datetime.utcnow().isoformat(),
            "num_agents": len(self.trace),
            "steps":      self.trace,
        }
        path.write_text(
            json.dumps(trace_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _slugify(self, text: str) -> str:
        """
        Convert a question string into a safe filename slug.
        e.g. "How does CRISPR work?" → "how_does_crispr_work"
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
        text = re.sub(r"\s+", "_", text.strip())  # spaces → underscores
        return text[:60]                           # cap length

