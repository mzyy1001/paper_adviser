# llm_quality.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from openai import OpenAI


@dataclass
class QualityJudgement:
    uid: str
    quality_score: float
    status: str          # accepted / rejected
    quality_reason: str


def judge_quality_one(
    uid: str,
    title: str,
    abstract: str,
    authors: list[str],
    primary_category: str | None,
    model: str = "gpt-5.2",
) -> QualityJudgement:
    """
    Decide ONLY paper quality. Fixed rubric. No user prompt.
    """
    client = OpenAI()

    system = """
You are an extremely strict research paper QUALITY gatekeeper.

Your job: judge whether this paper should be stored into a "high-quality library".
This is NOT about user preference. Only quality, novelty and substance.

High-quality means:
- clear problem statement
- non-trivial technical contribution
- decent methodology / experiments
- strong writing clarity
- realistic value to research community
You should be strict. Many papers should be rejected.

Return ONLY valid JSON:
{
  "uid": "...",
  "quality_score": 0-100,
  "status": "accepted" or "rejected",
  "quality_reason": "1-3 concise sentences"
}

Rules:
- If quality_score >= 75 => accepted, otherwise rejected.
- If abstract is too vague / no experiments / incremental => reject.
"""

    payload = {
        "uid": uid,
        "title": title,
        "abstract": abstract,
        "authors": authors[:8],
        "primary_category": primary_category,
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    data: Dict[str, Any] = json.loads(text)

    return QualityJudgement(
        uid=str(data["uid"]),
        quality_score=float(data["quality_score"]),
        status=str(data["status"]),
        quality_reason=str(data["quality_reason"]),
    )
