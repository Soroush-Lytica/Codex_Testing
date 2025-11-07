"""Pipeline for fetching news articles, summarizing them, and analyzing sentiment."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import requests
from openai import OpenAI

SERP_API_KEY = "REPLACE_WITH_YOUR_SERP_API_KEY"
SERP_API_URL = "https://serpapi.com/search.json"


@dataclass
class ArticleResult:
    title: str
    link: str
    source: str | None
    summary: str
    sentiment: str


def load_companies(csv_path: str) -> List[str]:
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
        return companies


def fetch_company_news(company: str) -> List[Dict[str, Any]]:
    params = {
        "engine": "google_news",
        "q": company,
        "api_key": SERP_API_KEY,
        "gl": "us",
    }
    response = requests.get(SERP_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("news_results", []) or []


def summarize_article(client: OpenAI, article: Dict[str, Any]) -> str:
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
    return response.choices[0].message.content.strip()


def analyze_sentiment(client: OpenAI, summary: str) -> str:
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
        return "positive"
    if "negative" in sentiment:
        return "negative"
    return "neutral"


def process_companies(csv_path: str, output_path: str) -> None:
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=openai_key)

    companies = load_companies(csv_path)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for company in companies:
        news_items = fetch_company_news(company)
        processed_articles: List[ArticleResult] = []
        for item in news_items:
            link = item.get("link")
            if not link:
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

        results[company] = [asdict(article) for article in processed_articles]

    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_companies("companies.csv", "news_results.json")
