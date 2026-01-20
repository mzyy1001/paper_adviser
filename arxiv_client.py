# arxiv_client.py
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import feedparser
import requests


ARXIV_API = "http://export.arxiv.org/api/query"


@dataclass
class ArxivPaper:
    uid: str
    title: str
    abstract: str
    authors: List[str]
    published_at: datetime
    updated_at: Optional[datetime]
    primary_category: Optional[str]
    categories: List[str]
    arxiv_url: str
    pdf_url: str


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s


def _parse_dt(dt_str: str) -> datetime:
    # arXiv returns RFC3339 like: 2025-01-20T12:34:56Z
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fetch_arxiv_latest(
    categories: List[str],
    max_results: int = 50,
    start: int = 0,
    sort_by: str = "submittedDate",
    sort_order: str = "descending",
    timeout_s: int = 15,
) -> List[ArxivPaper]:
    """
    Fetch latest papers from arXiv for given categories.
    Support pagination via `start`.
    """
    if not categories:
        raise ValueError("categories cannot be empty")

    if max_results <= 0:
        raise ValueError("max_results must be > 0")

    if start < 0:
        raise ValueError("start must be >= 0")

    cat_query = " OR ".join([f"cat:{c}" for c in categories])
    search_query = f"({cat_query})"

    params = {
        "search_query": search_query,
        "start": int(start),
        "max_results": int(max_results),
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    resp = requests.get(ARXIV_API, params=params, timeout=timeout_s)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    papers: List[ArxivPaper] = []

    for entry in feed.entries:
        arxiv_id = entry.get("id", "").split("/")[-1]  # e.g., 2501.12345v1
        title = _clean_text(entry.get("title", ""))
        abstract = _clean_text(entry.get("summary", ""))
        authors = [a.name for a in entry.get("authors", []) if hasattr(a, "name")]

        published_at = _parse_dt(entry.get("published", "1970-01-01T00:00:00Z"))
        updated_at = None
        if "updated" in entry:
            updated_at = _parse_dt(entry["updated"])

        primary_category = None
        if "arxiv_primary_category" in entry:
            primary_category = entry["arxiv_primary_category"].get("term")

        categories_all = []
        if "tags" in entry:
            categories_all = [
                t.get("term")
                for t in entry.tags
                if isinstance(t, dict) and t.get("term")
            ]

        arxiv_url = entry.get("link", entry.get("id", ""))

        # PDF link
        pdf_url = ""
        for l in entry.get("links", []):
            if l.get("type") == "application/pdf":
                pdf_url = l.get("href", "")
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        papers.append(
            ArxivPaper(
                uid=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published_at=published_at,
                updated_at=updated_at,
                primary_category=primary_category,
                categories=categories_all,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
            )
        )

    return papers
