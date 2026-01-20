# llm_match.py
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

from openai import OpenAI


@dataclass
class MatchResult:
    uid: str
    match_score: float
    match_reason: str
    match_summary: str
    prompt_hash: str


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def match_one(
    prompt: str,
    uid: str,
    title: str,
    abstract: str,
    authors: str,
    model: str = "gpt-5.2",
) -> MatchResult:
    client = OpenAI()
    prompt_hash = _hash_prompt(prompt)

    system = f"""
You are a strict paper recommender.
User needs:
{prompt}

Score this paper's FITNESS to user needs from 0 to 100.
Return ONLY valid JSON:
{{
  "uid": "...",
  "match_score": 0-100,
  "match_reason": "1-3 concise sentences",
  "match_summary": "max 6 sentences, focus on what user cares about"
}}
"""

    payload = {
        "uid": uid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    data = json.loads(resp.output_text)

    return MatchResult(
        uid=str(data["uid"]),
        match_score=float(data["match_score"]),
        match_reason=str(data["match_reason"]),
        match_summary=str(data["match_summary"]),
        prompt_hash=prompt_hash,
    )
