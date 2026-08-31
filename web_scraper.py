"""
web_scraper.py — Web Content Scraper & Sanitizer
================================================
Fetches, cleans, and sanitizes readable text content from web URLs for Layer 4.
Strips advertisements, navigation, header/footer boilerplate, scripts, and styling.
"""

import re
import urllib.parse
from typing import Any, Dict, List, Optional
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Elements to strip
UNWANTED_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "form",
    "noscript", "svg", "iframe", "button", "input", "select", "option"
]


class WebScraper:
    """
    Scrapes and extracts readable text from web URLs.
    """

    def __init__(self, timeout: int = 8, max_chars_per_page: int = 2500):
        self.timeout = timeout
        self.max_chars_per_page = max_chars_per_page

    def scrape_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrapes a single URL and extracts readable text.

        Returns:
            Dict: {"url": str, "title": str, "domain": str, "content": str} or None if failed.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc

            try:
                resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
            except requests.exceptions.SSLError:
                # Fallback for sites with self-signed/untrusted local certs (e.g. government portals)
                resp = requests.get(url, headers=HEADERS, timeout=self.timeout, verify=False)

            if resp.status_code != 200:
                print(f"[web_scraper] URL {url} returned status code {resp.status_code}")
                return None

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove unwanted tags
            for tag in soup(UNWANTED_TAGS):
                tag.decompose()

            title = soup.title.get_text().strip() if soup.title else domain

            # Extract main content elements (article, main, or paragraphs)
            main_container = soup.find("article") or soup.find("main") or soup.find("body") or soup

            paragraphs = []
            for elem in main_container.find_all(["h1", "h2", "h3", "p", "li"]):
                text = elem.get_text(separator=" ", strip=True)
                # Ignore very short or navigation-like text
                if len(text) > 25 and not text.startswith("©") and "cookie" not in text.lower():
                    paragraphs.append(text)

            clean_text = "\n".join(paragraphs)
            # Remove excessive whitespace
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()

            if not clean_text or len(clean_text) < 100:
                return None

            truncated_text = clean_text[:self.max_chars_per_page]

            return {
                "url": url,
                "title": title,
                "domain": domain,
                "content": truncated_text
            }

        except Exception as e:
            print(f"[web_scraper] Error scraping {url}: {e}")
            return None

    def scrape_multiple(self, urls: List[Dict[str, Any]], max_pages: int = 3) -> Dict[str, Any]:
        """
        Scrapes multiple URLs and aggregates their readable content.

        Args:
            urls: List of dicts with at least "url" and optional "title".
            max_pages: Maximum number of pages to successfully scrape.

        Returns:
            Dict containing:
              - "scraped_docs": List[Dict[str, str]]
              - "formatted_context": str ready for LLM prompt
        """
        scraped_docs = []
        context_parts = []

        for item in urls:
            if len(scraped_docs) >= max_pages:
                break

            target_url = item.get("url") if isinstance(item, dict) else item
            if not target_url:
                continue

            doc = self.scrape_url(target_url)
            if doc and doc.get("content"):
                scraped_docs.append(doc)
                safe_title = doc["title"].encode("ascii", errors="replace").decode("ascii")
                safe_content = doc["content"].encode("ascii", errors="replace").decode("ascii")
                context_parts.append(
                    f"--- SOURCE [{len(scraped_docs)}]: {safe_title} ({doc['url']}) ---\n{safe_content}\n"
                )

        formatted_context = "\n".join(context_parts) if context_parts else ""

        return {
            "scraped_docs": scraped_docs,
            "formatted_context": formatted_context
        }


def scrape_urls(urls: List[Dict[str, Any]], max_pages: int = 3) -> Dict[str, Any]:
    """Helper function to scrape multiple URLs."""
    scraper = WebScraper()
    return scraper.scrape_multiple(urls, max_pages=max_pages)
