"""Pipeline for fetching news articles, summarizing them, and analyzing sentiment."""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import requests
from openai import OpenAI

SERP_API_KEY = "REPLACE_WITH_YOUR_SERP_API_KEY"
SERP_API_URL = "https://serpapi.com/search.json"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class ArticleResult:
    title: str
    link: str
    source: str | None
    summary: str
    sentiment: str


def load_companies(csv_path: str) -> List[str]:
    logging.info("Loading company list from %s", csv_path)
    with open(csv_path, "r", encoding="utf-8-sig") as csv_file:
        reader = csv.reader(csv_file)
        companies = []
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            if not name or name.lower() == "mfrs":
                continue
            companies.append(name)
    logging.info("Loaded %d companies", len(companies))
    return companies


def fetch_company_news(company: str) -> List[Dict[str, Any]]:
    logging.info("Fetching news for company: %s", company)
    params = {
        "engine": "google_news",
        "q": company,
        "api_key": SERP_API_KEY,
        "gl": "us",
    }
    response = requests.get(SERP_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    results = payload.get("news_results", []) or []
    logging.info("Retrieved %d news results for %s", len(results), company)
    return results


def summarize_article(client: OpenAI, article: Dict[str, Any]) -> str:
    logging.info("Summarizing article: %s", article.get("title", "<no title>"))
    prompt = (
        "You are an assistant that summarizes news articles. "
        "Summarize the article concisely in 3-4 sentences using the available title, "
        "snippet, and link details."
    )
    title = article.get("title", "")
    snippet = article.get("snippet", "")
    link = article.get("link", "")

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n"
                "Provide the summary."
            ),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
    )
    summary = response.choices[0].message.content.strip()
    logging.info("Summary generated for article: %s", article.get("title", "<no title>"))
    return summary


def analyze_sentiment(client: OpenAI, summary: str) -> str:
    logging.info("Analyzing sentiment for summary: %s", summary[:60].replace("\n", " ") + ("..." if len(summary) > 60 else ""))
    messages = [
        {
            "role": "system",
            "content": (
                "Classify the sentiment of the provided news summary as Positive, Negative, or Neutral."
            ),
        },
        {"role": "user", "content": summary},
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    sentiment = response.choices[0].message.content.strip().lower()
    if "positive" in sentiment:
        final_sentiment = "positive"
    elif "negative" in sentiment:
        final_sentiment = "negative"
    else:
        final_sentiment = "neutral"
    logging.info("Detected sentiment: %s", final_sentiment)
    return final_sentiment


def process_companies(csv_path: str, output_path: str) -> None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=openai_key)

    companies = load_companies(csv_path)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for company in companies:
        logging.info("Processing company: %s", company)
        news_items = fetch_company_news(company)
        processed_articles: List[ArticleResult] = []
        for item in news_items:
            link = item.get("link")
            if not link:
                logging.info("Skipping article without link: %s", item.get("title", "<no title>"))
                continue

            summary = summarize_article(client, item)
            sentiment = analyze_sentiment(client, summary)
            processed_articles.append(
                ArticleResult(
                    title=item.get("title", ""),
                    link=link,
                    source=item.get("source"),
                    summary=summary,
                    sentiment=sentiment,
                )
            )

            if len(processed_articles) >= 10:
                logging.info(
                    "Reached limit of 10 valid articles for %s", company
                )
                break

        results[company] = [asdict(article) for article in processed_articles]
        logging.info("Finished processing %s with %d articles", company, len(processed_articles))

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)
    logging.info("Results written to %s", output_path)


if __name__ == "__main__":
    process_companies("companies.csv", "news_results.json")
