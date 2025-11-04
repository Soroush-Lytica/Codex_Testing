"""Utility to discover company headquarters using a search API and an LLM.

This script reads manufacturer names from a CSV file, sends a search query for each
manufacturer to a configurable search API, and then summarizes the search results
with a language model to infer the headquarters location.

Environment variables:
    SEARCH_API_ENDPOINT:  Base URL for the search API (defaults to Bing Web Search).
    SEARCH_API_KEY:       API key used to authenticate the search API request.
    OPENAI_API_KEY:       API key used for the OpenAI client.
    OPENAI_MODEL:         (Optional) model name to use for the OpenAI call. Defaults
                          to "gpt-4o-mini".

Usage:
    python search_headquarters.py manufacturers.csv --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import sys
import time
from collections.abc import Iterable
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


DEFAULT_SEARCH_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_SECONDS = 1.5  # polite delay between requests


@dataclasses.dataclass
class SearchResult:
    """Simplified representation of a web search result."""

    title: str
    snippet: str
    url: str

    def to_prompt_fragment(self) -> str:
        return f"- {self.title}\n  {self.snippet}\n  Source: {self.url}"


class HeadquartersFinder:
    """Orchestrates the search + language model workflow for manufacturers."""

    def __init__(
        self,
        search_endpoint: str,
        search_api_key: str,
        openai_api_key: str,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        *,
        search_result_count: int = 5,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.search_endpoint = search_endpoint
        self.search_api_key = search_api_key
        self.session = session or requests.Session()
        self.session.headers.update({"Ocp-Apim-Subscription-Key": search_api_key})

        self.llm_client = OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model
        self.search_result_count = max(1, search_result_count)

    # ------------------------------------------------------------
    # Search API integration
    # ------------------------------------------------------------
    def search(self, query: str, *, count: int = 5) -> List[SearchResult]:
        """Send the query to the search API and return normalized results."""

        logging.debug("Querying search API: %s", query)
        params = {"q": query, "count": count, "mkt": "en-US"}
        response = self.session.get(
            self.search_endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        return self._extract_results(payload)

    def _extract_results(self, payload: Dict[str, Any]) -> List[SearchResult]:
        """Extract search results from the API payload.

        The implementation focuses on the shape returned by Bing Web Search but tries
        to remain robust if the structure varies slightly. Results without a title or
        snippet are ignored.
        """

        web_pages = payload.get("webPages") or payload.get("value") or {}
        entries: Iterable[Dict[str, Any]]

        if isinstance(web_pages, dict):
            entries = web_pages.get("value", [])
        elif isinstance(web_pages, list):
            entries = web_pages
        else:
            entries = []

        results: List[SearchResult] = []
        for item in entries:
            title = item.get("name") or item.get("title")
            snippet = item.get("snippet") or item.get("description")
            url = item.get("url") or item.get("link")

            if not (title and snippet and url):
                continue

            results.append(SearchResult(title=title, snippet=snippet, url=url))

        logging.debug("Extracted %d normalized results", len(results))
        return results

    # ------------------------------------------------------------
    # Language model integration
    # ------------------------------------------------------------
    def infer_headquarters(self, manufacturer: str, results: List[SearchResult]) -> str:
        """Use an LLM to infer the headquarters from search results."""

        if not results:
            logging.warning("No search results available for %s", manufacturer)
            return ""

        prompt = self._build_prompt(manufacturer, results)
        logging.debug("Submitting prompt to LLM for %s", manufacturer)

        response = self.llm_client.responses.create(
            model=self.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You determine the headquarters location for electronics "
                        "manufacturers. Return a short statement with the city and "
                        "country (and state if in the US)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content_chunks = response.output or []
        for chunk in content_chunks:
            for message in chunk.get("content", []):
                if message.get("type") == "output_text":
                    return message.get("text", "").strip()

        # Fallback: try accessing the first text response directly if the
        # structured parsing above fails due to SDK variations.
        try:
            return response.output_text.strip()  # type: ignore[attr-defined]
        except AttributeError:
            logging.error("Unexpected response format from LLM for %s", manufacturer)
            return ""

    def _build_prompt(self, manufacturer: str, results: List[SearchResult]) -> str:
        lines = [
            f"Determine the headquarters of the manufacturer '{manufacturer}'.",
            "Use the following search results:",
        ]
        lines.extend(result.to_prompt_fragment() for result in results)
        lines.append(
            "Respond in the format: '<manufacturer>: <city>, <state/region>, <country>'."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------
    # High-level workflow
    # ------------------------------------------------------------
    def process_manufacturers(self, manufacturers: Iterable[str]) -> List[Dict[str, str]]:
        """Process each manufacturer and return structured results."""

        findings: List[Dict[str, str]] = []
        for name in manufacturers:
            query = f"Where is {name} headquarters located?"
            try:
                search_results = self.search(query, count=self.search_result_count)
                headquarters = self.infer_headquarters(name, search_results)
            except requests.HTTPError as exc:
                logging.error("Search API error for %s: %s", name, exc)
                headquarters = ""
            except requests.RequestException as exc:
                logging.error("Network error for %s: %s", name, exc)
                headquarters = ""

            findings.append({
                "manufacturer": name,
                "headquarters": headquarters,
            })

            logging.info("Processed %s -> %s", name, headquarters or "<unknown>")
            time.sleep(RATE_LIMIT_SECONDS)

        return findings


def read_manufacturers(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile)
        # Skip header row if present
        header = next(reader, [])
        manufacturers: List[str] = []
        if header:
            # If the header looks like a manufacturer name, include it.
            if header[0].strip() and header[0].lower() != "mfrs":
                manufacturers.append(header[0].strip())
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if name:
                manufacturers.append(name)
        return manufacturers


def write_results(path: str, results: Iterable[Dict[str, str]]) -> None:
    fieldnames = ["manufacturer", "headquarters"]
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manufacturers_csv",
        help="Path to the CSV file containing manufacturer names.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the headquarters results as CSV.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of search results to retrieve per manufacturer.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    search_endpoint = os.getenv("SEARCH_API_ENDPOINT", DEFAULT_SEARCH_ENDPOINT)
    search_api_key = os.getenv("SEARCH_API_KEY")
    if not search_api_key:
        parser.error("SEARCH_API_KEY environment variable must be set.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        parser.error("OPENAI_API_KEY environment variable must be set.")

    openai_model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    manufacturers = read_manufacturers(args.manufacturers_csv)
    if not manufacturers:
        logging.warning("No manufacturers found in %s", args.manufacturers_csv)
        return 0

    finder = HeadquartersFinder(
        search_endpoint=search_endpoint,
        search_api_key=search_api_key,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        search_result_count=args.count,
    )

    results = finder.process_manufacturers(manufacturers)

    if args.output:
        write_results(args.output, results)
        logging.info("Wrote results to %s", args.output)
    else:
        for row in results:
            print(f"{row['manufacturer']}: {row['headquarters']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
