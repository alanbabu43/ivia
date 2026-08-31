"""
google_search.py — Google Web Search Client
===========================================
Performs web searches via Google Search and extracts relevant, authoritative URLs.
Filters low-quality spam and prioritizes official, educational, and trusted sources.
"""

import os
import re
import urllib.parse
from typing import Any, Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# Common desktop User-Agent to avoid generic bot blocks
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains to prioritize
PRIORITY_DOMAINS = [".gov", ".nic.in", ".org", ".edu", "wikipedia.org", "reuters.com", "bbc.com", "thehindu.com", "indianexpress.com"]

# Domains / patterns to filter out
BLOCKED_DOMAINS = ["google.com", "youtube.com", "facebook.com", "instagram.com", "twitter.com", "x.com", "pinterest.com", "tiktok.com"]


class GoogleSearcher:
    """
    Client for performing Google searches and returning ranked URLs.
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes a Google Search query and extracts clean, relevant URLs.

        Args:
            query: The search query string.
            max_results: Max number of top URLs to return.

        Returns:
            List of dicts: [{"title": str, "url": str, "domain": str, "priority": bool}]
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        encoded_query = urllib.parse.quote_plus(clean_query)
        search_url = f"https://www.google.com/search?q={encoded_query}&num={max_results * 2}&hl=en"

        results: List[Dict[str, str]] = []
        seen_urls = set()

        try:
            response = requests.get(search_url, headers=HEADERS, timeout=self.timeout)
            if response.status_code != 200:
                print(f"[google_search] Google search returned status code {response.status_code}")
                # Fallback to duckduckgo HTML search if Google throttles/blocks
                return self._fallback_search(clean_query, max_results)

            soup = BeautifulSoup(response.text, "html.parser")

            # Google search result link containers
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]

                # Extract actual target URL from Google redirect or direct href
                target_url = None
                if href.startswith("/url?q="):
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                    target_url = parsed.get("q", [None])[0]
                elif href.startswith("http://") or href.startswith("https://"):
                    target_url = href

                if not target_url or target_url in seen_urls:
                    continue

                parsed_url = urllib.parse.urlparse(target_url)
                domain = parsed_url.netloc.lower()

                # Filter internal Google and social links
                if any(b in domain for b in BLOCKED_DOMAINS) or not domain:
                    continue

                # Extract title if available
                title_elem = a_tag.find("h3") or a_tag
                title = title_elem.get_text().strip() if title_elem else domain

                is_priority = any(p in domain for p in PRIORITY_DOMAINS)

                seen_urls.add(target_url)
                results.append({
                    "title": title or domain,
                    "url": target_url,
                    "domain": domain,
                    "priority": is_priority
                })

                if len(results) >= max_results:
                    break

        except Exception as e:
            print(f"[google_search] Error during Google Search: {e}")

        # If Google search returned 0 links (e.g. captcha/consent redirect), use fallback search
        if not results:
            return self._fallback_search(clean_query, max_results)

        # Sort priority domains first while preserving relative rank
        results.sort(key=lambda x: 0 if x.get("priority") else 1)
        return results[:max_results]

    def _fallback_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Fallback search using DuckDuckGo HTML if Google blocks or fails.
        """
        print(f"[google_search] Attempting fallback HTML search for '{query}'...")
        results: List[Dict[str, str]] = []
        seen_urls = set()

        try:
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=HEADERS,
                timeout=self.timeout
            )
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                for link in soup.select("a.result__url, a.result__snippet, a.result__a"):
                    href = link.get("href", "")
                    actual_url = None
                    if "uddg=" in href:
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        actual_url = parsed.get("uddg", [None])[0]
                    elif href.startswith("http"):
                        actual_url = href

                    if actual_url and actual_url not in seen_urls:
                        parsed_url = urllib.parse.urlparse(actual_url)
                        domain = parsed_url.netloc.lower()
                        if any(b in domain for b in BLOCKED_DOMAINS) or not domain:
                            continue
                        seen_urls.add(actual_url)
                        title = link.get_text().strip() or domain
                        is_priority = any(p in domain for p in PRIORITY_DOMAINS)
                        results.append({
                            "title": title,
                            "url": actual_url,
                            "domain": domain,
                            "priority": is_priority
                        })
                        if len(results) >= max_results:
                            break
        except Exception as ex:
            print(f"[google_search] Fallback search also encountered: {ex}")

        # Sort priority domains first
        results.sort(key=lambda x: 0 if x.get("priority") else 1)
        return results[:max_results]


def search_google(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Helper function to perform Google search."""
    searcher = GoogleSearcher()
    return searcher.search(query, max_results=max_results)
