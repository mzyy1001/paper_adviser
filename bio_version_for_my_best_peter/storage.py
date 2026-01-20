# storage.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import requests


DEFAULT_COLUMNS = [
    # Identity
    "uid",
    "source",          # pubmed / europe_pmc / biorxiv / medrxiv / local
    "source_id",       # PMID / DOI / EuropePMC id
    "title",
    "abstract",
    "authors",
    "published_at",
    "primary_category",
    "categories",

    # URLs
    "source_url",      # preferred
    "arxiv_url",       # deprecated but kept for backward compatibility
    "pdf_url",
    "pdf_path",

    # Extra metadata (biology-friendly)
    "doi",
    "pmid",
    "journal",

    # Stage 1: Quality
    "status",                  # pending / accepted / rejected
    "quality_score",
    "quality_reason",
    "quality_reviewed_at",

    # Stage 2: Match to user prompt
    "match_score",
    "match_reason",
    "match_summary",
    "match_prompt_hash",

    # Meta
    "added_at",
    "last_updated_at",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(uid: str) -> str:
    return uid.replace("/", "_").replace(":", "_")


def ensure_dirs(library_dir: str) -> Dict[str, str]:
    os.makedirs(library_dir, exist_ok=True)
    pdf_dir = os.path.join(library_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    return {"library_dir": library_dir, "pdf_dir": pdf_dir}


def load_csv(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for c in DEFAULT_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[DEFAULT_COLUMNS]
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


def has_reviewed(df: pd.DataFrame, uid: str) -> bool:
    """Return True if uid exists and quality_reviewed_at not null."""
    if df.empty:
        return False
    sub = df[df["uid"].astype(str) == str(uid)]
    if len(sub) == 0:
        return False
    v = sub.iloc[0].get("quality_reviewed_at")
    return pd.notna(v) and str(v).strip() != ""


def get_status(df: pd.DataFrame, uid: str) -> Optional[str]:
    if df.empty:
        return None
    sub = df[df["uid"].astype(str) == str(uid)]
    if len(sub) == 0:
        return None
    s = sub.iloc[0].get("status")
    return str(s) if pd.notna(s) else None


def upsert_rows(csv_path: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = load_csv(csv_path)
    existing = {str(x): i for i, x in enumerate(df["uid"].astype(str).tolist())}

    for r in rows:
        uid = str(r.get("uid", "")).strip()
        if not uid:
            continue

        r = dict(r)
        r["last_updated_at"] = now_iso()

        if uid in existing:
            idx = existing[uid]
            for k, v in r.items():
                if k in df.columns:
                    df.at[idx, k] = v
        else:
            new_row = {c: None for c in DEFAULT_COLUMNS}
            for k, v in r.items():
                if k in new_row:
                    new_row[k] = v
            if not new_row.get("added_at"):
                new_row["added_at"] = now_iso()
            if not new_row.get("status"):
                new_row["status"] = "pending"
            df.loc[len(df)] = new_row

    df.to_csv(csv_path, index=False)
    return df


def download_pdf(pdf_url: str, out_path: str, timeout_s: int = 30) -> bool:
    try:
        if not pdf_url or not str(pdf_url).startswith("http"):
            return False
        r = requests.get(pdf_url, timeout=timeout_s)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False
