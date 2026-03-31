"""
main.py
-------
Command-line entry point for the AutoResearch pipeline.

Usage examples:
    # Basic usage
    python main.py "How does CRISPR work?"

    # Show verbose logs while running
    python main.py "How does CRISPR work?" --verbose

    # Run all three sample questions back to back
    python main.py --run-samples
"""

import argparse
import logging
import sys
from core.orchestrator import Orchestrator

#Sample questions from the assignment spec
SAMPLE_QUESTIONS = [
    "What are the primary mechanisms by which large language models hallucinate, and what mitigation strategies have been proposed?",
    "What are the environmental and social impacts of lithium mining for electric vehicle batteries?",
    "Explain the architecture and key innovations of the Transformer model.",
]


def setup_logging(verbose: bool) -> None:
    """
    Configure logging for the whole application.

    verbose=True  → DEBUG level, shows every agent's internal steps
    verbose=False → INFO level, shows high-level pipeline progress only
    """
    level  = logging.DEBUG if verbose else logging.INFO
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=format, stream=sys.stdout)

    # Silence noisy third-party loggers regardless of verbose mode
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def run_question(question: str, writer_model: str) -> None:
    """Run the pipeline for a single question and print the result path."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")

    try:
        orch        = Orchestrator(writer_model=writer_model)
        report_path = orch.run(question)

        print(f"\n Done!")
        print(f"   Report : {report_path}")
        print(f"   Trace  : {report_path.replace('.md', '').replace('outputs/', 'outputs/run_trace_')}.json")

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        logging.exception("Pipeline error")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoResearch — multi-agent research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "How does CRISPR work?"
  python main.py "What is quantum error correction?" --verbose
  python main.py --run-samples
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",             
        help="Research question to investigate.",
    )
    parser.add_argument(
        "--writer-model",
        default="gpt-4o-mini",
        help="Model for the Writer agent (default: gpt-4o-mini). Use gpt-4o for richer prose.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed logs from every agent.",
    )
    parser.add_argument(
        "--run-samples",
        action="store_true",
        help="Run all 3 sample questions from the assignment spec.",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Validate arguments
    if not args.question and not args.run_samples:
        parser.print_help()
        print("\nError: provide a question or use --run-samples")
        sys.exit(1)

    # Run
    if args.run_samples:
        print(f"Running {len(SAMPLE_QUESTIONS)} sample questions...\n")
        for question in SAMPLE_QUESTIONS:
            run_question(question, args.writer_model)
    else:
        run_question(args.question, args.writer_model)


if __name__ == "__main__":
    main()

    