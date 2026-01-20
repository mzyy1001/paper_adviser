# bio_client.py
from __future__ import annotations

import os
import re
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
import xml.etree.ElementTree as ET


# -----------------------------
# Unified paper schema
# -----------------------------
@dataclass
class BioPaper:
    uid: str                    # unique across all sources, e.g. "pubmed:12345"
    source: str                 # pubmed / europe_pmc / biorxiv / medrxiv
    source_id: str              # PMID / DOI / EuropePMC id
    title: str
    abstract: str
    authors: List[str]
    published_at: datetime
    updated_at: Optional[datetime]
    primary_category: Optional[str]
    categories: List[str]
    source_url: str
    pdf_url: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _parse_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_date_ymd(y: str, m: str, d: str) -> Optional[datetime]:
    if not y:
        return None
    mm, dd = 1, 1

    if m:
        m = m.strip()
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        if m.isdigit():
            mm = int(m)
        else:
            mm = month_map.get(m.lower()[:3], 1)

    if d and d.isdigit():
        dd = int(d)

    try:
        return datetime(int(y), int(mm), int(dd), tzinfo=timezone.utc)
    except Exception:
        return None


def _hit_keywords(title: str, abstract: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    t = (title or "").lower()
    a = (abstract or "").lower()
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        if k in t or k in a:
            return True
    return False


# -----------------------------
# PubMed (NCBI E-utilities)
# -----------------------------
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _ncbi_common_params() -> Dict[str, str]:
    """
    Optional env vars:
      - NCBI_TOOL
      - NCBI_EMAIL
      - NCBI_API_KEY
    """
    d = {
        "tool": os.getenv("NCBI_TOOL", "bio_paper_ingestor"),
        "email": os.getenv("NCBI_EMAIL", ""),
        "api_key": os.getenv("NCBI_API_KEY", ""),
    }
    return {k: v for k, v in d.items() if v}


def fetch_pubmed_latest(
    query: str,
    max_results: int = 50,
    start: int = 0,
    last_days: int = 14,
    timeout_s: int = 20,
    sleep_s: float = 0.0,
) -> List[BioPaper]:
    """
    PubMed:
      - ESearch: get PMIDs
      - EFetch: get title/abstract/authors/date
    Pagination:
      - start -> retstart
    """
    if not query.strip():
        raise ValueError("PubMed query cannot be empty")

    esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": int(max_results),
        "retstart": int(start),
        "datetype": "pdat",
        "reldate": int(last_days),
        "sort": "most+recent",
        **_ncbi_common_params(),
    }

    r = requests.get(esearch_url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    pmids = data.get("esearchresult", {}).get("idlist", []) or []

    if sleep_s > 0:
        time.sleep(float(sleep_s))

    if not pmids:
        return []

    efetch_url = f"{EUTILS_BASE}/efetch.fcgi"
    params2 = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        **_ncbi_common_params(),
    }
    r2 = requests.get(efetch_url, params=params2, timeout=timeout_s)
    r2.raise_for_status()

    root = ET.fromstring(r2.text)
    papers: List[BioPaper] = []

    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            continue

        title_el = art.find(".//Article/ArticleTitle")
        title = _clean_text("".join(title_el.itertext()) if title_el is not None else "")

        abs_parts = []
        for abs_el in art.findall(".//Article/Abstract/AbstractText"):
            abs_parts.append(_clean_text("".join(abs_el.itertext())))
        abstract = _clean_text(" ".join([x for x in abs_parts if x]))

        authors = []
        for au in art.findall(".//Article/AuthorList/Author"):
            coll = au.findtext("CollectiveName", default="").strip()
            if coll:
                authors.append(coll)
                continue
            ln = au.findtext("LastName", default="").strip()
            ini = au.findtext("Initials", default="").strip()
            name = (ln + (" " + ini if ini else "")).strip()
            if name:
                authors.append(name)

        journal = art.findtext(".//Article/Journal/Title", default="").strip() or None

        # date heuristics
        published_at = None
        ad = art.find(".//Article/ArticleDate")
        if ad is not None:
            y = ad.findtext("Year", default="")
            m = ad.findtext("Month", default="")
            d = ad.findtext("Day", default="")
            published_at = _parse_date_ymd(y, m, d)

        if published_at is None:
            pd = art.find(".//Article/Journal/JournalIssue/PubDate")
            if pd is not None:
                y = pd.findtext("Year", default="")
                m = pd.findtext("Month", default="")
                d = pd.findtext("Day", default="")
                published_at = _parse_date_ymd(y, m, d)

        if published_at is None:
            published_at = datetime(1970, 1, 1, tzinfo=timezone.utc)

        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        papers.append(
            BioPaper(
                uid=f"pubmed:{pmid}",
                source="pubmed",
                source_id=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                published_at=published_at,
                updated_at=None,
                primary_category="pubmed",
                categories=[],
                source_url=source_url,
                pdf_url="",     # PubMed usually doesn't give direct PDF
                doi=None,
                pmid=pmid,
                journal=journal,
            )
        )

    return papers


# -----------------------------
# Europe PMC REST API
# -----------------------------
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def fetch_europe_pmc_latest(
    query: str,
    page_size: int = 50,
    cursor_mark: str = "*",
    synonym: bool = True,
    timeout_s: int = 20,
) -> Tuple[List[BioPaper], str]:
    """
    Europe PMC search (cursor pagination).
    """
    if not query.strip():
        raise ValueError("Europe PMC query cannot be empty")

    # Europe PMC supports `sort_date:y` in query
    q = f"({query}) sort_date:y"

    params = {
        "query": q,
        "format": "json",
        "resultType": "core",
        "pageSize": int(page_size),
        "cursorMark": cursor_mark,
        "synonym": "true" if synonym else "false",
    }

    r = requests.get(EPMC_SEARCH, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    results = (data.get("resultList", {}) or {}).get("result", []) or []
    next_cursor = data.get("nextCursorMark") or cursor_mark

    papers: List[BioPaper] = []

    for it in results:
        src = (it.get("source") or "").strip()  # MED / PMC / PPR ...
        ext_id = (it.get("id") or "").strip()
        if not src or not ext_id:
            continue

        title = _clean_text(it.get("title") or "")
        abstract = _clean_text(it.get("abstractText") or "")
        author_str = _clean_text(it.get("authorString") or "")
        authors = [a.strip() for a in re.split(r",|;", author_str) if a.strip()] if author_str else []

        journal = (it.get("journalTitle") or "").strip() or None
        doi = (it.get("doi") or "").strip() or None
        pmid = (it.get("pmid") or "").strip() or None

        dt = _parse_iso(it.get("firstPublicationDate") or "") or _parse_iso(it.get("publicationDate") or "")
        if dt is None:
            y = str(it.get("pubYear") or "").strip()
            dt = _parse_date_ymd(y, "", "") or datetime(1970, 1, 1, tzinfo=timezone.utc)

        source_url = f"https://europepmc.org/article/{src}/{ext_id}"

        pdf_url = ""
        ft = it.get("fullTextUrlList") or {}
        ft_list = ft.get("fullTextUrl") or []
        if isinstance(ft_list, list):
            for u in ft_list:
                if str(u.get("documentStyle") or "").lower() == "pdf":
                    pdf_url = str(u.get("url") or "")
                    break

        papers.append(
            BioPaper(
                uid=f"epmc:{src}:{ext_id}",
                source="europe_pmc",
                source_id=f"{src}:{ext_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                published_at=dt,
                updated_at=None,
                primary_category=src,
                categories=[],
                source_url=source_url,
                pdf_url=pdf_url,
                doi=doi,
                pmid=pmid,
                journal=journal,
            )
        )

    return papers, next_cursor


# -----------------------------
# bioRxiv / medRxiv details API
# -----------------------------
BIORXIV_BASE = "https://api.biorxiv.org/details"
MEDRXIV_BASE = "https://api.medrxiv.org/details"


def _preprint_urls(server: str, doi: str, version: str) -> Tuple[str, str]:
    domain = "www.biorxiv.org" if server.lower() == "biorxiv" else "www.medrxiv.org"
    source_url = f"https://{domain}/content/{doi}v{version}"
    pdf_url = source_url + ".full.pdf"
    return source_url, pdf_url


def fetch_preprint_latest(
    server: str = "biorxiv",
    interval: str = "14d",
    cursor: int = 0,
    category: Optional[str] = None,
    timeout_s: int = 20,
) -> List[BioPaper]:
    """
    Pull latest preprints from bioRxiv/medRxiv by time window.
    Cursor increases by 100.
    """
    base = BIORXIV_BASE if server.lower() == "biorxiv" else MEDRXIV_BASE
    url = f"{base}/{server}/{interval}/{int(cursor)}/json"

    params = {}
    if category:
        params["category"] = category

    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()

    items = data.get("collection") or []
    papers: List[BioPaper] = []

    for it in items:
        doi = (it.get("doi") or "").strip()
        if not doi:
            continue

        title = _clean_text(it.get("title") or "")
        abstract = _clean_text(it.get("abstract") or "")

        authors_raw = _clean_text(it.get("authors") or "")
        if ";" in authors_raw:
            authors = [a.strip() for a in authors_raw.split(";") if a.strip()]
        elif authors_raw:
            authors = [a.strip() for a in authors_raw.split(",") if a.strip()]
        else:
            authors = []

        date_str = (it.get("date") or "").strip()
        dt = _parse_iso(date_str)
        if dt is None and len(date_str) >= 10:
            dt = _parse_date_ymd(date_str[:4], date_str[5:7], date_str[8:10])
        if dt is None:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)

        version = str(it.get("version") or "1").strip()
        cat = (it.get("category") or "").strip() or None

        source_url, pdf_url = _preprint_urls(server, doi, version)

        papers.append(
            BioPaper(
                uid=f"{server}:{doi}v{version}",
                source=server.lower(),
                source_id=doi,
                title=title,
                abstract=abstract,
                authors=authors,
                published_at=dt,
                updated_at=None,
                primary_category=cat,
                categories=[cat] if cat else [],
                source_url=source_url,
                pdf_url=pdf_url,
                doi=doi,
                pmid=None,
                journal=None,
            )
        )

    return papers


# -----------------------------
# Aggregator: multi-source batch
# -----------------------------
@dataclass
class BioFetchState:
    pubmed_start: int = 0
    epmc_cursor: str = "*"
    biorxiv_cursor: int = 0
    medrxiv_cursor: int = 0


def fetch_bio_batch(
    sources: List[str],
    query: str,
    page_size: int = 50,
    last_days: int = 14,
    state: Optional[BioFetchState] = None,
    keywords_filter: Optional[List[str]] = None,
    biorxiv_category: Optional[str] = None,
    medrxiv_category: Optional[str] = None,
) -> Tuple[List[BioPaper], BioFetchState]:
    """
    Fetch a batch of papers from multiple biology sources.

    PubMed / Europe PMC: real query search
    bioRxiv / medRxiv: time window pull + local keyword filter

    Return: (papers, next_state)
    """
    if state is None:
        state = BioFetchState()

    sources_norm = [s.strip().lower() for s in sources if s.strip()]
    if not sources_norm:
        return [], state

    k = max(1, len(sources_norm))
    per = max(10, page_size // k)

    papers_all: List[BioPaper] = []

    if "pubmed" in sources_norm:
        ps = fetch_pubmed_latest(
            query=query,
            max_results=per,
            start=state.pubmed_start,
            last_days=last_days,
            sleep_s=0.0,
        )
        papers_all.extend(ps)
        state.pubmed_start += per

    if "europe_pmc" in sources_norm or "epmc" in sources_norm:
        ps, next_cursor = fetch_europe_pmc_latest(
            query=query,
            page_size=per,
            cursor_mark=state.epmc_cursor,
            synonym=True,
        )
        state.epmc_cursor = next_cursor
        papers_all.extend(ps)

    if "biorxiv" in sources_norm:
        ps = fetch_preprint_latest(
            server="biorxiv",
            interval=f"{int(last_days)}d",
            cursor=state.biorxiv_cursor,
            category=biorxiv_category,
        )
        state.biorxiv_cursor += 100
        if keywords_filter:
            ps = [p for p in ps if _hit_keywords(p.title, p.abstract, keywords_filter)]
        papers_all.extend(ps)

    if "medrxiv" in sources_norm:
        ps = fetch_preprint_latest(
            server="medrxiv",
            interval=f"{int(last_days)}d",
            cursor=state.medrxiv_cursor,
            category=medrxiv_category,
        )
        state.medrxiv_cursor += 100
        if keywords_filter:
            ps = [p for p in ps if _hit_keywords(p.title, p.abstract, keywords_filter)]
        papers_all.extend(ps)

    # de-dup by uid
    seen = set()
    uniq: List[BioPaper] = []
    for p in papers_all:
        if p.uid in seen:
            continue
        seen.add(p.uid)
        uniq.append(p)

    return uniq, state
