"""Tooling to fetch company news, scrape article content, and summarise via GPT."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - dependency might not be installed during linting
    OpenAI = None  # type: ignore[misc]


SERP_API_URL = "https://serpapi.com/search"
DEFAULT_MAX_ARTICLES = 3
DEFAULT_TIMEOUT = 30
SCRAPER_DEFAULT_TIMEOUT = 60


class ConfigurationError(RuntimeError):
    """Raised when the runtime configuration is incomplete."""


@dataclass
class ArticleSummary:
    company: str
    title: str
    link: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "title": self.title,
            "link": self.link,
            "summary": self.summary,
        }


def read_companies(csv_path: str) -> List[str]:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        if not header:
            raise ValueError("CSV file must contain a header row with a column for company names")
        first_column = header[0]
        companies = [row[first_column].strip() for row in reader if row.get(first_column)]
        return [company for company in companies if company]


def fetch_news_articles(company: str, api_key: str, max_articles: int, *,
                         timeout: int = DEFAULT_TIMEOUT) -> List[dict]:
    params = {
        "engine": "google_news",
        "q": company,
        "api_key": api_key,
    }
    response = requests.get(SERP_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("news_results", [])
    return articles[:max_articles]


def scrape_article(url: str, scraper_url: str, *, scraper_api_key: Optional[str] = None,
                   timeout: int = SCRAPER_DEFAULT_TIMEOUT) -> str:
    headers = {}
    if scraper_api_key:
        headers["Authorization"] = f"Bearer {scraper_api_key}"

    payload = {"url": url}
    response = requests.post(scraper_url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        if "data" in data and isinstance(data["data"], dict):
            content = data["data"].get("content")
            if isinstance(content, str):
                return content
    raise ValueError("Scraping API response does not include a 'content' field")


def build_openai_client(api_key: str):
    if OpenAI is None:
        raise ImportError("The openai package is required to generate summaries")
    return OpenAI(api_key=api_key)


def summarise_with_llm(client, content: str, *, model: str, company: str, url: str,
                       max_retries: int = 3, retry_backoff: float = 2.0) -> str:
    prompt = (
        "Summarise the following news article about {company} in no more than 70 words. "
        "Focus on the facts and include the article's key insight.\n\n"
        "Article URL: {url}\n\n"
        "Article Content:\n{content}"
    ).format(company=company, url=url, content=content)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": "You are a concise financial news analyst."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=220,
            )
        except Exception:  # pragma: no cover - network errors
            if attempt == max_retries:
                raise
            time.sleep(retry_backoff ** attempt)
            continue

        # Responses API returns output in a structured format
        outputs = getattr(response, "output", None)
        if outputs:
            for item in outputs:
                if getattr(item, "type", None) == "output_text":
                    return item.text.strip()
        # Fallback for dict-like objects
        if isinstance(response, dict):
            candidates = response.get("output", []) or response.get("data", [])
            for item in candidates:
                text = item.get("content") or item.get("text")
                if isinstance(text, str):
                    return text.strip()
        raise RuntimeError("Unexpected response format from OpenAI API")

    raise RuntimeError("Failed to retrieve summary from OpenAI API")


def process_companies(companies: Iterable[str], *, serp_api_key: str, scraper_url: str,
                      openai_key: str, openai_model: str, max_articles: int) -> List[ArticleSummary]:
    client = build_openai_client(openai_key)
    summaries: List[ArticleSummary] = []

    scraper_api_key = os.getenv("SCRAPER_API_KEY")

    for company in companies:
        articles = fetch_news_articles(company, serp_api_key, max_articles)
        for article in articles:
            link = article.get("link") or article.get("url")
            title = article.get("title") or "Untitled"
            if not link:
                continue
            content = scrape_article(link, scraper_url, scraper_api_key=scraper_api_key)
            summary_text = summarise_with_llm(client, content, model=openai_model,
                                              company=company, url=link)
            summaries.append(ArticleSummary(company=company, title=title, link=link,
                                            summary=summary_text))
    return summaries


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise news articles for companies.")
    parser.add_argument("csv", help="Path to CSV containing a column with company names")
    parser.add_argument("output", help="Where to write the JSON summaries")
    parser.add_argument("--max-articles", type=int, default=DEFAULT_MAX_ARTICLES,
                        help="Number of top news articles to summarise per company")
    parser.add_argument("--scraper-url", required=False,
                        default=os.getenv("SCRAPER_API_URL"),
                        help="Endpoint of the scraping API accepting POST requests")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        help="OpenAI model identifier to use for summaries")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    serp_api_key = os.getenv("SERP_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not serp_api_key:
        raise ConfigurationError("SERP_API_KEY environment variable is required")
    if not openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY environment variable is required")
    if not args.scraper_url:
        raise ConfigurationError("Scraper API URL must be provided via --scraper-url or SCRAPER_API_URL")

    companies = read_companies(args.csv)
    summaries = process_companies(
        companies,
        serp_api_key=serp_api_key,
        scraper_url=args.scraper_url,
        openai_key=openai_api_key,
        openai_model=args.openai_model,
        max_articles=args.max_articles,
    )

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump([summary.to_dict() for summary in summaries], fh, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:  # pragma: no cover - CLI feedback
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
