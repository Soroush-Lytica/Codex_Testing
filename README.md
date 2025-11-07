# Codex Testing

This repository contains tooling to fetch recent news for semiconductor manufacturers and summarise the articles with GPT.

## Requirements

* Python 3.10+
* `requests`
* `openai`
* API keys:
  * `SERP_API_KEY` – SerpAPI key for Google News results
  * `OPENAI_API_KEY` – key for GPT-4o (or compatible) summarisation
  * Optional: `SCRAPER_API_KEY` – if your scraping service requires authentication

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Prepare a CSV file with a header row containing a list of company names (see `manufacturers.csv` for an example). Then run:

```bash
export SERP_API_KEY="your-serp-api-key"
export OPENAI_API_KEY="your-openai-key"
export SCRAPER_API_URL="https://your-scraper.example.com/extract"
python news_summary.py manufacturers.csv summaries.json
```

Additional options:

* `--max-articles` – limit the number of news articles per company (default: 3)
* `--scraper-url` – override the scraping endpoint without changing the environment variable
* `--openai-model` – choose a different OpenAI model (default: `gpt-4o-mini` or the value in `OPENAI_MODEL`)

The script saves a JSON file containing the company name, article title, link, and the generated summary (up to 70 words) for each processed article.

## Workflow

For each company, the script:

1. Queries SerpAPI's Google News endpoint for the latest articles.
2. Sends every article URL to the configured scraping API to obtain the raw text content.
3. Sends the scraped text to an OpenAI model (e.g. GPT-4o) and requests a concise summary capped at 70 words.
4. Outputs the summaries in JSON format.

Errors related to missing configuration variables are surfaced immediately when the script starts. Network calls include simple retry logic when contacting the OpenAI API.
