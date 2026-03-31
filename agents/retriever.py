"""
agents/retriever.py
-------------------
AGENT-02: Retriever

Responsibility:
    Given one sub-question and its keywords, search the web using the free
    Wikipedia REST API and DuckDuckGo Instant Answer API, and return raw
    text snippets with their source URLs.

    No LLM call is made here — this agent is purely about data fetching.
    The Summarizer (Agent-03) handles the interpretation.

Input  (RetrieverInput)  : { sub_question: str, keywords: list[str] }
Output (RetrieverOutput) : { sub_question: str, snippets: list[Snippet] }

Free APIs used (no keys required):
    Wikipedia REST API   : https://en.wikipedia.org/api/rest_v1/page/summary/{title}
    Wikipedia Search API : https://en.wikipedia.org/w/api.php?action=query&list=search
    DuckDuckGo Instant   : https://api.duckduckgo.com/?q={query}&format=json
"""

import logging
import time
import requests
from urllib.parse import quote

from core.schemas import RetrieverInput, RetrieverOutput, Snippet, validate

logger=logging.getLogger(__name__)

# Constants
WIKIPEDIA_SEARCH_URL="https://en.wikipedia.org/w/api.php"
WIKIPEDIA_SUMMARY_URL="https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
DUCKDUCKGO_URL="https://api.duckduckgo.com/"

# How many Wikipedia results to fetch per keyword query
WIKI_RESULTS_PER_QUERY=2

# Polite delay between API calls (avoids rate limiting)
REQUEST_DELAY_SECONDS=0.5

# Request timeout in seconds
TIMEOUT=10

# User-Agent header — Wikipedia requires a descriptive User-Agent
HEADERS={
    "User-Agent": "AutoResearch/1.0 (educational project; contact: student@university.edu)"
}


class RetrieverAgent:
    """
    Fetches raw text snippets from Wikipedia and DuckDuckGo for a
    single sub-question.

    Usage
    -----
    retriever=RetrieverAgent()
    output=retriever.run({
        "sub_question": "How do transformers process sequences?",
        "keywords": ["transformer architecture", "attention mechanism"]
    })
    """

    def __init__(self):
        logger.info("RetrieverAgent initialised")

    def run(self, input_data: RetrieverInput) -> RetrieverOutput:
        """
        Fetch snippets for one sub-question using its keywords.

        Strategy:
          1. For each keyword → search Wikipedia → fetch summary of top result
          2. Query DuckDuckGo with the full sub-question → grab abstract
          3. Deduplicate by URL
          4. Return all snippets found (minimum 1, typically 3-6)

        Parameters
        ----------
        input_data : RetrieverInput
            { sub_question: str, keywords: list[str] }

        Returns
        -------
        RetrieverOutput
            { sub_question: str, snippets: list[Snippet] }
        """
        sub_question=input_data["sub_question"]
        keywords    =input_data["keywords"]

        logger.info("Retriever working on: '%s'", sub_question)
        logger.info("Keywords: %s", keywords)

        all_snippets: list[Snippet]=[]
        seen_urls:    set[str]     =set()   # for deduplication

        # ── Step 1: Wikipedia search for each keyword ──────────────────────────
        for keyword in keywords:
            wiki_snippets=self._search_wikipedia(keyword)
            for snippet in wiki_snippets:
                if snippet["url"] not in seen_urls:
                    all_snippets.append(snippet)
                    seen_urls.add(snippet["url"])
            time.sleep(REQUEST_DELAY_SECONDS)   # be polite to the API

        # ── Step 2: DuckDuckGo instant answer for the full sub-question ────────
        ddg_snippet=self._search_duckduckgo(sub_question)
        if ddg_snippet and ddg_snippet["url"] not in seen_urls:
            all_snippets.append(ddg_snippet)
            seen_urls.add(ddg_snippet["url"])

        # ── Step 3: Fallback — if we got nothing, try the sub-question itself ──
        if not all_snippets:
            logger.warning("No snippets found via keywords, trying sub-question directly")
            fallback=self._search_wikipedia(sub_question)
            all_snippets.extend(fallback)

        logger.info(
            "Retriever collected %d snippet(s) for '%s'",
            len(all_snippets),
            sub_question,
        )

        result: RetrieverOutput={
            "sub_question": sub_question,
            "snippets":     all_snippets,
        }

        validate(RetrieverOutput, result)
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _search_wikipedia(self, query: str) -> list[Snippet]:
        """
        Search Wikipedia for a query and return summaries of the top results.

        Two-step process:
          1. Use the Wikipedia Search API to find relevant page titles
          2. Fetch the summary for each page title via the REST summary endpoint

        Parameters
        ----------
        query : str
            Search term (usually a keyword or short phrase).

        Returns
        -------
        list[Snippet]
            0 to WIKI_RESULTS_PER_QUERY snippets, each with text/url/title.
        """
        snippets: list[Snippet]=[]

        # ── Step 1: Search for page titles ────────────────────────────────────
        try:
            search_params={
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "srlimit":  WIKI_RESULTS_PER_QUERY,
                "format":   "json",
            }
            resp=requests.get(
                WIKIPEDIA_SEARCH_URL,
                params=search_params,
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            search_results=resp.json().get("query", {}).get("search", [])

        except requests.RequestException as e:
            logger.warning("Wikipedia search failed for '%s': %s", query, e)
            return []

        # ── Step 2: Fetch the summary for each result ─────────────────────────
        for result in search_results:
            page_title=result.get("title", "")
            if not page_title:
                continue

            snippet=self._fetch_wikipedia_summary(page_title)
            if snippet:
                snippets.append(snippet)

            time.sleep(REQUEST_DELAY_SECONDS)  # be polite between requests

        return snippets

    def _fetch_wikipedia_summary(self, title: str) -> Snippet | None:
        """
        Fetch the plain-text summary of a specific Wikipedia page.

        Parameters
        ----------
        title : str
            Exact Wikipedia page title (e.g. "Transformer (deep learning)").

        Returns
        -------
        Snippet | None
            A Snippet dict, or None if the request failed or returned no text.
        """
        try:
            encoded_title=quote(title, safe="")
            url=WIKIPEDIA_SUMMARY_URL.format(title=encoded_title)

            resp=requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            data=resp.json()

            # Wikipedia returns 'extract' for the plain-text summary
            text=data.get("extract", "").strip()
            page_url=data.get("content_urls", {}).get("desktop", {}).get("page", url)

            if not text:
                logger.debug("Empty extract for Wikipedia page: %s", title)
                return None

            return Snippet(
                text =text[:2000],   # cap at 2000 chars to stay within token limits
                url  =page_url,
                title=data.get("title", title),
            )

        except requests.RequestException as e:
            logger.warning("Wikipedia summary fetch failed for '%s': %s", title, e)
            return None

    def _search_duckduckgo(self, query: str) -> Snippet | None:
        """
        Query the DuckDuckGo Instant Answer API for a search query.

        DuckDuckGo returns a free-form 'AbstractText' field for well-known
        topics. This is shorter than Wikipedia but can cover topics that
        Wikipedia doesn't summarise well.

        Parameters
        ----------
        query : str
            The search query (typically the full sub-question).

        Returns
        -------
        Snippet | None
            A Snippet if DuckDuckGo returned a non-empty abstract, else None.
        """
        try:
            params={
                "q":              query,
                "format":         "json",
                "no_html":        "1",
                "skip_disambig":  "1",
            }
            resp=requests.get(
                DUCKDUCKGO_URL,
                params=params,
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data=resp.json()

            abstract=data.get("AbstractText", "").strip()
            source  =data.get("AbstractURL",  "").strip()
            title   =data.get("Heading",       query).strip()

            if not abstract:
                logger.debug("DuckDuckGo returned no abstract for: '%s'", query)
                return None

            return Snippet(
                text =abstract[:2000],
                url  =source or DUCKDUCKGO_URL,
                title=title,
            )

        except requests.RequestException as e:
            logger.warning("DuckDuckGo search failed for '%s': %s", query, e)
            return None

