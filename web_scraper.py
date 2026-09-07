"""
web_scraper.py — Web Content Scraper & Sanitizer (PATCHED)
================================================
Fetches, cleans, and sanitizes readable text content from web URLs for Layer 4.
Strips advertisements, navigation, header/footer boilerplate, scripts, and styling.

CHANGE: scrape_url() and scrape_multiple() now accept an optional `query` param.
Instead of blindly truncating to the first N characters (which can cut off the
actual answer-bearing paragraph in favor of generic intro text), paragraphs are
scored by relevance to the query and the most relevant ones are kept, in their
original page order, up to max_chars_per_page.
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

UNWANTED_TAGS = [
    "script", "style", "nav", "header", "footer", "aside", "form",
    "noscript", "svg", "iframe", "button", "input", "select", "option"
]

# Words too common to help relevance scoring
_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "are", "of", "in", "on", "to",
    "and", "or", "for", "with", "who", "what", "when", "where", "which",
    "did", "does", "do", "how", "why", "last", "latest", "current", "recent",
    "most", "won", "win", "wins", "there", "it", "this", "that", "be"
}

# Recent-year pattern boosts paragraphs mentioning current/near-future years,
# which is usually exactly what "last/latest/current" questions need.
_YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")


class WebScraper:
    """
    Scrapes and extracts readable text from web URLs.
    """

    def __init__(self, timeout: int = 8, max_chars_per_page: int = 2500):
        self.timeout = timeout
        self.max_chars_per_page = max_chars_per_page

    def _score_paragraph(self, text: str, query_words: set) -> float:
        """
        Scores a paragraph's relevance to the query.
        Higher score = more likely to contain the direct answer.
        """
        text_lower = text.lower()
        text_words = set(re.findall(r"[a-z0-9']+", text_lower))

        # Base score: overlap with meaningful query words
        overlap = len(query_words & text_words)
        score = overlap * 2.0

        # Boost paragraphs mentioning a recent year (helps "last/latest/current" questions)
        years_found = _YEAR_RE.findall(text)
        if years_found:
            score += 1.5
            # Extra boost if it's a very recent year (likely the "latest" entry)
            if any(int(y) >= 2023 for y in years_found):
                score += 1.0

        # Slight penalty for very generic "history/stats" paragraphs that often
        # open sports/records pages (common false-positive source of confusion)
        generic_markers = ["since 19", "has won", "have won", "titles", "history of"]
        if any(m in text_lower for m in generic_markers) and not years_found:
            score -= 0.5

        return score

    def _select_relevant_text(self, paragraphs: List[str], query: Optional[str], max_chars: int) -> str:
        """
        Selects paragraphs most relevant to the query instead of naively
        truncating to the first N characters. Falls back to simple
        truncation if no query is provided.
        """
        if not query:
            joined = "\n".join(paragraphs)
            return joined[:max_chars]

        query_words = set(re.findall(r"[a-z0-9']+", query.lower())) - _STOPWORDS
        if not query_words:
            joined = "\n".join(paragraphs)
            return joined[:max_chars]

        # Score each paragraph, keep original index for stable ordering
        scored = [
            (self._score_paragraph(p, query_words), idx, p)
            for idx, p in enumerate(paragraphs)
        ]

        # Sort by score descending; ties broken by original page order
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected_indices = []
        total = 0
        for score, idx, p in scored:
            if score <= 0 and selected_indices:
                # Once we've got at least one relevant paragraph, stop pulling
                # in zero/negative-relevance filler unless we still have lots of room.
                if total > max_chars * 0.5:
                    break
            if total + len(p) > max_chars:
                continue
            selected_indices.append(idx)
            total += len(p)
            if total >= max_chars:
                break

        if not selected_indices:
            # Nothing scored — fall back to naive truncation so we never return empty
            joined = "\n".join(paragraphs)
            return joined[:max_chars]

        # Restore original page order for readability/coherence
        selected_indices.sort()
        return "\n".join(paragraphs[i] for i in selected_indices)

    def scrape_url(self, url: str, query: Optional[str] = None) -> Optional[Dict[str, str]]:
        """
        Scrapes a single URL and extracts readable text.

        Args:
            url: Target URL to scrape.
            query: Optional original user question. When provided, the most
                   query-relevant paragraphs are prioritized instead of
                   naive first-N-characters truncation.

        Returns:
            Dict: {"url": str, "title": str, "domain": str, "content": str} or None if failed.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc

            try:
                resp = requests.get(url, headers=HEADERS, timeout=self.timeout)
            except requests.exceptions.SSLError:
                resp = requests.get(url, headers=HEADERS, timeout=self.timeout, verify=False)

            if resp.status_code != 200:
                print(f"[web_scraper] URL {url} returned status code {resp.status_code}")
                return None

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(UNWANTED_TAGS):
                tag.decompose()

            title = soup.title.get_text().strip() if soup.title else domain

            main_container = soup.find("article") or soup.find("main") or soup.find("body") or soup

            paragraphs = []
            for elem in main_container.find_all(["h1", "h2", "h3", "p", "li", "td", "th"]):
                text = elem.get_text(separator=" ", strip=True)
                if len(text) > 25 and not text.startswith("©") and "cookie" not in text.lower():
                    paragraphs.append(text)

            if not paragraphs:
                return None

            # Clean up each paragraph's internal whitespace
            paragraphs = [re.sub(r"\s{2,}", " ", p).strip() for p in paragraphs]

            selected_text = self._select_relevant_text(paragraphs, query, self.max_chars_per_page)
            clean_text = re.sub(r"\n{3,}", "\n\n", selected_text).strip()

            if not clean_text or len(clean_text) < 100:
                return None

            return {
                "url": url,
                "title": title,
                "domain": domain,
                "content": clean_text
            }

        except Exception as e:
            print(f"[web_scraper] Error scraping {url}: {e}")
            return None

    def scrape_multiple(
        self,
        urls: List[Dict[str, Any]],
        max_pages: int = 3,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scrapes multiple URLs and aggregates their readable content.

        Args:
            urls: List of dicts with at least "url" and optional "title".
            max_pages: Maximum number of pages to successfully scrape.
            query: Optional original user question, passed through to scrape_url
                   for relevance-based content selection.

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

            doc = self.scrape_url(target_url, query=query)
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


def scrape_urls(urls: List[Dict[str, Any]], max_pages: int = 3, query: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to scrape multiple URLs, optionally prioritizing content relevant to `query`."""
    scraper = WebScraper()
    return scraper.scrape_multiple(urls, max_pages=max_pages, query=query)