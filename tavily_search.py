"""
tavily_search.py — Tavily Web Search Client
=============================================
Provides live internet search capability for ONLINE mode queries using the Tavily Search API.
"""

import os
import socket
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_ENDPOINT = "https://api.tavily.com/search"


def is_online(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """
    Checks internet connectivity by attempting a TCP connection to Google DNS.

    Args:
        host: IP address to ping (default: Google DNS 8.8.8.8).
        port: Port to use (default: 53 / DNS).
        timeout: Connection timeout in seconds.

    Returns:
        True if internet is reachable, False otherwise.
    """
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
        return True
    except (socket.timeout, OSError):
        return False


class TavilySearcher:
    """
    Client for interacting with the Tavily Search API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the Tavily searcher.

        Args:
            api_key: Tavily API Key. Defaults to TAVILY_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "").strip()

    def is_configured(self) -> bool:
        """Checks if Tavily API key is available."""
        return bool(self.api_key and self.api_key != "your_tavily_api_key_here")

    def search(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a web search query via Tavily Search API.

        Args:
            query: The search query string.
            search_depth: "basic" or "advanced".
            max_results: Maximum number of web search results to return.
            include_answer: Include an AI generated summary answer from Tavily if available.

        Returns:
            Dictionary containing:
                - "query": str
                - "results": list of dicts [{"title", "url", "content", "score"}]
                - "answer": optional str from Tavily
                - "formatted_context": str ready for LLM prompt
        """
        if not self.is_configured():
            raise ValueError(
                "Tavily API key is missing or not configured. "
                "Please set TAVILY_API_KEY in your .env file."
            )

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "include_answer": include_answer,
            "max_results": max_results
        }

        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                TAVILY_API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            tavily_answer = data.get("answer", "")

            formatted_context_parts = []
            if tavily_answer:
                formatted_context_parts.append(f"Tavily Web Summary: {tavily_answer}\n")

            sources_list = []
            for idx, res in enumerate(results, start=1):
                title = res.get("title", "No Title")
                url = res.get("url", "#")
                content = res.get("content", "")
                sources_list.append({
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": res.get("score", 0.0)
                })
                # Sanitize unicode to ASCII-safe for Windows console/LLM compatibility
                safe_title = title.encode("ascii", errors="replace").decode("ascii")
                safe_content = content.encode("ascii", errors="replace").decode("ascii")
                formatted_context_parts.append(
                    f"[{idx}] {safe_title}\nURL: {url}\nContent: {safe_content}\n"
                )

            formatted_context = "\n---\n".join(formatted_context_parts) if formatted_context_parts else "No live web results found."

            return {
                "query": query,
                "results": sources_list,
                "answer": tavily_answer,
                "formatted_context": formatted_context
            }

        except requests.exceptions.RequestException as e:
            print(f"[tavily_search] API Error: {e}")
            raise RuntimeError(f"Tavily Search API request failed: {str(e)}")


def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Helper function to run a Tavily search."""
    searcher = TavilySearcher()
    return searcher.search(query, max_results=max_results)
