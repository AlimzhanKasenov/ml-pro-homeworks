"""Responsible scraper for the educational site books.toscrape.com."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def fetch(session: requests.Session, url: str, delay: float) -> BeautifulSoup:
    """Download one page with a delay and explicit timeout."""
    time.sleep(delay)
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def parse_book(session: requests.Session, url: str, delay: float) -> dict:
    soup = fetch(session, url, delay)
    title = soup.select_one("div.product_main h1").get_text(strip=True)
    rating_node = soup.select_one("p.star-rating")
    rating_word = next(item for item in rating_node.get("class", []) if item in RATINGS)
    description_header = soup.find("div", id="product_description")
    description = description_header.find_next_sibling("p").get_text(" ", strip=True) if description_header else ""
    breadcrumbs = soup.select("ul.breadcrumb li a")
    category = breadcrumbs[-1].get_text(strip=True) if breadcrumbs else "Unknown"
    price_text = soup.select_one("p.price_color").get_text(strip=True).replace("£", "").replace("Â", "")
    return {
        "title": title,
        "description": description,
        "category": category,
        "price_gbp": float(price_text),
        "rating": RATINGS[rating_word],
        "url": url,
    }


def scrape_books(limit: int = 200, delay: float = 0.3) -> pd.DataFrame:
    session = requests.Session()
    session.headers["User-Agent"] = "ML-course-homework/1.0 (educational, low-rate scraper)"
    rows: list[dict] = []
    page_url = BASE_URL
    while page_url and len(rows) < limit:
        page = fetch(session, page_url, delay)
        links = [urljoin(page_url, node.get("href")) for node in page.select("article.product_pod h3 a")]
        for link in links:
            if len(rows) >= limit:
                break
            try:
                rows.append(parse_book(session, link, delay))
            except (requests.RequestException, AttributeError, StopIteration, ValueError) as error:
                print(f"Skip {link}: {error}")
        next_node = page.select_one("li.next a")
        page_url = urljoin(page_url, next_node.get("href")) if next_node else None
        print(f"Collected: {len(rows)}/{limit}")
    frame = pd.DataFrame(rows)
    frame["text"] = (frame["title"].fillna("") + ". " + frame["description"].fillna("")).str.strip()
    frame["is_high_rating"] = (frame["rating"] >= 4).astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("data/books.csv"))
    args = parser.parse_args()
    result = scrape_books(args.limit, args.delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
