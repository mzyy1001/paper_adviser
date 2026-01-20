# run_ingest_quality.py
from __future__ import annotations

import os
from typing import List, Dict, Any

from dotenv import load_dotenv

from bio_client import fetch_bio_batch, BioFetchState
from llm_quality import judge_quality_one
from storage import ensure_dirs, load_csv, upsert_rows, now_iso, safe_filename, download_pdf

load_dotenv()

LIB_DIR = "library"
CSV_PATH = os.path.join(LIB_DIR, "papers.csv")


def paper_to_base_row(p) -> Dict[str, Any]:
    return {
        "uid": p.uid,
        "source": p.source,
        "source_id": p.source_id,

        "title": p.title,
        "abstract": p.abstract,
        "authors": "; ".join(p.authors),

        "published_at": p.published_at.isoformat(),
        "primary_category": p.primary_category or "",
        "categories": "; ".join(p.categories or []),

        "source_url": p.source_url,
        "arxiv_url": "",  # deprecated
        "pdf_url": p.pdf_url,
        "pdf_path": None,

        "doi": p.doi or "",
        "pmid": p.pmid or "",
        "journal": p.journal or "",

        "status": "pending",
        "added_at": now_iso(),
    }


def main():
    ensure_dirs(LIB_DIR)
    df = load_csv(CSV_PATH)

    # ✅ 多源：想加就加
    sources = ["pubmed", "europe_pmc", "biorxiv", "medrxiv"]

    # ✅ PubMed + EuropePMC 用这个 query（建议写准一点）
    query = '("single cell" OR scRNA-seq OR "single-cell") AND (genomics OR transcriptome OR atlas)'

    # ✅ bioRxiv/medRxiv 时间窗口拉取后，用关键词过滤更稳
    keywords_filter = ["single cell", "scRNA", "cell atlas", "genomics"]

    last_days = 14

    # ✅ 本次最多新增 review 的数量（真正调用 LLM 的次数）
    max_new_reviews = 10
    new_reviewed_cnt = 0

    # ✅ 每轮抓取量（会自动分摊到多个 sources）
    page_size = 60

    accepted_cnt = 0
    rejected_cnt = 0
    skipped_cnt = 0

    state = BioFetchState()

    print(f"[CONFIG] sources={sources}, page_size={page_size}, last_days={last_days}, max_new_reviews={max_new_reviews}")

    while new_reviewed_cnt < max_new_reviews:
        print(f"\n[FETCH] state={state}")
        papers, state = fetch_bio_batch(
            sources=sources,
            query=query,
            page_size=page_size,
            last_days=last_days,
            state=state,
            keywords_filter=keywords_filter,
            biorxiv_category=None,
            medrxiv_category=None,
        )

        if not papers:
            print("[DONE] No more papers returned.")
            break

        total = len(papers)

        for i, p in enumerate(papers, start=1):
            uid = p.uid

            if new_reviewed_cnt >= max_new_reviews:
                print(f"\n[STOP] reached max_new_reviews={max_new_reviews}")
                break

            print(f"\n[{i}/{total}] Processing: {uid} | source={p.source} | cat={p.primary_category}")
            print(f"Title: {p.title[:120]}")
            print(f"Authors: {', '.join(p.authors[:6])}{'...' if len(p.authors) > 6 else ''}")
            print(f"Published: {p.published_at.isoformat()}")
            print("Judging quality... ", end="", flush=True)

            # 1) skip if already reviewed
            sub = df[df["uid"].astype(str) == str(uid)]
            if len(sub) > 0:
                reviewed_at = sub.iloc[0].get("quality_reviewed_at")
                if reviewed_at is not None and str(reviewed_at).strip() != "" and str(reviewed_at) != "nan":
                    skipped_cnt += 1
                    print("SKIPPED (already reviewed)")
                    continue

            # 2) ensure base metadata exists
            base_row = paper_to_base_row(p)
            df = upsert_rows(CSV_PATH, [base_row])

            # 3) quality judgement (ONE paper)
            try:
                q = judge_quality_one(
                    uid=uid,
                    title=p.title,
                    abstract=p.abstract,
                    authors=p.authors,
                    primary_category=p.primary_category,
                )
            except Exception as e:
                print(f"\n❌ LLM ERROR: {repr(e)}")
                row_update = {
                    "uid": uid,
                    "status": "rejected",
                    "quality_score": 0,
                    "quality_reason": f"LLM error: {repr(e)}",
                    "quality_reviewed_at": now_iso(),
                }
                df = upsert_rows(CSV_PATH, [row_update])
                rejected_cnt += 1
                new_reviewed_cnt += 1
                continue

            # 打印判定结果
            print(f"DONE → status={q.status} | score={q.quality_score}")

            row_update = {
                "uid": uid,
                "status": q.status,
                "quality_score": q.quality_score,
                "quality_reason": q.quality_reason,
                "quality_reviewed_at": now_iso(),
            }

            # 4) if accepted, download pdf (optional)
            if q.status == "accepted":
                accepted_cnt += 1
                if (p.pdf_url or "").startswith("http"):
                    out_path = os.path.join(LIB_DIR, "pdfs", f"{safe_filename(uid)}.pdf")
                    print(f"Downloading PDF... ", end="", flush=True)
                    ok = download_pdf(p.pdf_url, out_path)
                    if ok:
                        row_update["pdf_path"] = out_path
                        print("OK ✅")
                    else:
                        print("FAILED ❌")
            else:
                rejected_cnt += 1

            # 5) write update
            df = upsert_rows(CSV_PATH, [row_update])

            # 本次新增 review +1
            new_reviewed_cnt += 1

            reason_short = (q.quality_reason or "")[:240].replace("\n", " ")
            if reason_short:
                print(f"Reason: {reason_short}{'...' if len(q.quality_reason or '') > 240 else ''}")

    print(
        f"\n[DONE] accepted={accepted_cnt}, rejected={rejected_cnt}, skipped={skipped_cnt}, "
        f"new_reviewed={new_reviewed_cnt}"
    )
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
