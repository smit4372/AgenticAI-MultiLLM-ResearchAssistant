"""
core/llm.py
-----------
Central wrapper around the OpenAI Chat Completions API.

Responsibilities:
  - Load the API key from the environment (never hard-coded)
  - Provide a single `chat()` function all agents use
  - Provide `safe_parse()` to robustly parse JSON responses with one retry
"""

import os
import json
import logging
from typing import Any, Callable, Optional

from openai import OpenAI
from dotenv import load_dotenv

#Logging setup
# Each module gets its own logger so you can trace exactly which component
# produced a log line.
logger = logging.getLogger(__name__)

#Load environment variables from .env file
#python-dotenv reads the .env file in the project root and injects variables
#into os.environ.  Safe to call multiple times — it won't overwrite values
#that are already set in the real environment.
load_dotenv()

#Lazily initialise the OpenAI client
#We don't create the client at import time so that unit tests that only
#exercise safe_parse() can run without a real API key present.
_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """Return a cached OpenAI client, creating it on first use."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. "
                "Create a .env file with OPENAI_API_KEY=your-key-here."
            )
        _client = OpenAI(api_key=api_key)
    return _client

#Default model
DEFAULT_MODEL = "gpt-4o-mini"


#Core chat function

def chat(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> str:
    """
    Send a single-turn chat request to the OpenAI API and return the raw
    text response.

    Parameters
    ----------
    system_prompt : str
        Instructions that shape the model's behaviour (e.g. "respond only
        in valid JSON").
    user_message : str
        The actual content/data the model should act on.
    model : str
        Which OpenAI model to use.  Defaults to gpt-4o-mini.
    temperature : float
        0.0 = deterministic, 1.0 = creative.  We keep it low (0.2) so
        JSON output is consistent and predictable.
    max_tokens : int
        Hard cap on the response length.

    Returns
    -------
    str
        The raw text content of the model's reply.

    Raises
    ------
    RuntimeError
        If the API call fails for any reason (network, auth, rate limit).
    """
    try:
        logger.debug("Calling OpenAI | model=%s | system=%s...", model, system_prompt[:60])

        response = _get_client().chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        # Extract the text from the first choice
        content = response.choices[0].message.content.strip()
        logger.debug("OpenAI response received (%d chars)", len(content))
        return content

    except Exception as e:
        logger.error("OpenAI API call failed: %s", e)
        raise RuntimeError(f"LLM call failed: {e}") from e


#JSON safe-parser

def safe_parse(
    raw: str,
    retry_fn: Optional[Callable[[], str]] = None,
) -> dict[str, Any]:
    """
    Attempt to parse a JSON string returned by an LLM.

    LLMs occasionally wrap their JSON in markdown code fences like:
        ```json
        { ... }
        ```
    or add a brief preamble like "Here is the JSON:".
    This function strips those artefacts before parsing.

    If the first parse fails AND a retry_fn is provided, it calls retry_fn()
    once to get a fresh response, then tries to parse that.

    Parameters
    ----------
    raw : str
        The raw text returned by the LLM.
    retry_fn : Callable[[], str] | None
        A zero-argument callable that re-runs the LLM call and returns a new
        raw string.  Pass None if you don't want a retry.

    Returns
    -------
    dict
        Parsed JSON as a Python dict.

    Raises
    ------
    ValueError
        If JSON parsing fails even after the retry.
    """

    def _clean_and_parse(text: str) -> dict[str, Any]:
        """Strip markdown fences and leading/trailing whitespace, then parse."""
        # Remove ```json ... ``` or ``` ... ``` wrappers
        if "```" in text:
            # Keep only the content between the first ``` and the last ```
            text = text.split("```")[1]          # text after first ```
            if text.lower().startswith("json"):  # remove optional language tag
                text = text[4:]
        return json.loads(text.strip())

    #First attempt
    try:
        return _clean_and_parse(raw)
    except (json.JSONDecodeError, IndexError) as first_error:
        logger.warning("JSON parse failed on first attempt: %s", first_error)

        # ── Retry (if a retry function was supplied) ───────────────────────────
        if retry_fn is None:
            raise ValueError(
                f"JSON parsing failed and no retry_fn was provided.\n"
                f"Raw response was:\n{raw}"
            ) from first_error

        logger.info("Retrying LLM call to obtain valid JSON...")
        raw_retry = retry_fn()

        try:
            result = _clean_and_parse(raw_retry)
            logger.info("Retry succeeded.")
            return result
        except (json.JSONDecodeError, IndexError) as second_error:
            raise ValueError(
                f"JSON parsing failed after retry.\n"
                f"Original error : {first_error}\n"
                f"Retry error    : {second_error}\n"
                f"Last raw response:\n{raw_retry}"
            ) from second_error


#Convenience wrapper used by agents

def chat_json(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2000,
) -> dict[str, Any]:
    """
    Like `chat()` but automatically parses the response as JSON, with one
    retry on parse failure.

    All agents that expect JSON output should use this instead of `chat()`.

    Returns
    -------
    dict
        Parsed JSON response.
    """
    # Define the retry function as a closure so it reuses the same parameters
    def retry_fn() -> str:
        logger.info("Retrying chat call for JSON repair...")
        return chat(
            system_prompt=system_prompt + "\n\nIMPORTANT: Your previous response was not valid JSON. Respond with ONLY valid JSON, no other text.",
            user_message=user_message,
            model=model,
            temperature=0.0,   # Use 0 temp on retry for maximum determinism
            max_tokens=max_tokens,
        )

    raw = chat(system_prompt, user_message, model, temperature, max_tokens)
    return safe_parse(raw, retry_fn=retry_fn)
