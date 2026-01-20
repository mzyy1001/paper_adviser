# app.py
from __future__ import annotations

import os
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from llm_quality import judge_quality_one, QualityJudgement

from arxiv_client import fetch_arxiv_latest

# ✅ Match only
from llm_match import match_one, MatchResult

from storage import (
    ensure_dirs,
    load_csv,
    upsert_rows,
    now_iso,
    download_pdf,
    safe_filename,
)

load_dotenv()

DEFAULT_LIBRARY_DIR = "library"
DEFAULT_CSV_PATH = os.path.join(DEFAULT_LIBRARY_DIR, "papers.csv")


# ----------------------------
# Utilities
# ----------------------------
def batch_quality_llm(
    rows: List[Dict[str, Any]],
    model: str,
    sleep_s: float = 0.0,
    max_retry: int = 2,
) -> List[QualityJudgement]:
    judgements: List[QualityJudgement] = []
    n = len(rows)
    progress = st.progress(0)

    for i, r in enumerate(rows):
        uid = str(r.get("uid", ""))
        title = str(r.get("title", ""))
        abstract = str(r.get("abstract", ""))
        authors = str(r.get("authors", "") or "")
        authors_list = [a.strip() for a in authors.split(";") if a.strip()]
        primary_category = str(r.get("primary_category", "") or "") or None

        last_err = None
        for _ in range(max_retry + 1):
            try:
                j = judge_quality_one(
                    uid=uid,
                    title=title,
                    abstract=abstract,
                    authors=authors_list,
                    primary_category=primary_category,
                    model=model,
                )
                judgements.append(j)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8)

        if last_err is not None:
            judgements.append(QualityJudgement(
                uid=uid,
                quality_score=0.0,
                status="rejected",
                quality_reason=f"ERROR: {str(last_err)}",
            ))

        if sleep_s > 0:
            time.sleep(float(sleep_s))

        progress.progress(int((i + 1) / max(1, n) * 100))

    return judgements


def parse_dt_safe(x: Any) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def is_today(dt: Optional[datetime], tz=timezone.utc) -> bool:
    if dt is None:
        return False
    return dt.astimezone(tz).date() == datetime.now(tz).date()


def within_last_days(dt: Optional[datetime], days: int) -> bool:
    if dt is None:
        return False
    return dt >= datetime.now(timezone.utc) - timedelta(days=days)


def compute_prompt_hash(prompt: str) -> str:
    # ✅ keep consistent with llm_match._hash_prompt (sha256[:16])
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def make_local_uid(file_bytes: bytes, filename: str) -> str:
    h = hashlib.sha1()
    h.update(file_bytes)
    h.update(filename.encode("utf-8", errors="ignore"))
    return "local:" + h.hexdigest()[:16]


def infer_source(row: pd.Series) -> str:
    uid = str(row.get("uid", "") or "")
    if uid.startswith("local:"):
        return "local"
    arxiv_url = str(row.get("arxiv_url", "") or "")
    pdf_url = str(row.get("pdf_url", "") or "")
    if "arxiv.org" in arxiv_url or "arxiv.org" in pdf_url:
        return "arxiv"
    return "unknown"


def paper_to_row_arxiv(p: Any) -> Dict[str, Any]:
    # 你 storage.py 的 schema
    return {
        "uid": p.uid,
        "title": p.title,
        "abstract": p.abstract,
        "authors": "; ".join(p.authors),
        "published_at": p.published_at.isoformat(),
        "primary_category": p.primary_category or "",
        "categories": "; ".join(p.categories or []),
        "arxiv_url": p.arxiv_url,
        "pdf_url": p.pdf_url,
        "pdf_path": None,

        # Stage 1: Quality（由 run_ingest_quality.py 负责）
        "status": "pending",
        "quality_score": None,
        "quality_reason": None,
        "quality_reviewed_at": None,

        # Stage 2: Match（由本 app 负责）
        "match_score": None,
        "match_reason": None,
        "match_summary": None,
        "match_prompt_hash": None,

        # Meta
        "added_at": now_iso(),
    }


def local_row_from_upload(uid: str, title: str, pdf_path: str, note: str = "") -> Dict[str, Any]:
    return {
        "uid": uid,
        "title": title,
        # 本地 PDF 的 abstract 暂存备注，方便 match 命中
        "abstract": note.strip(),
        "authors": "",
        "published_at": "",
        "primary_category": "",
        "categories": "",
        "arxiv_url": "",
        "pdf_url": "",
        "pdf_path": pdf_path,

        "status": "pending",
        "quality_score": None,
        "quality_reason": None,
        "quality_reviewed_at": None,

        "match_score": None,
        "match_reason": None,
        "match_summary": None,
        "match_prompt_hash": None,

        "added_at": now_iso(),
    }


def sort_for_browse(df: pd.DataFrame, score_col: str, mode: str) -> pd.DataFrame:
    df2 = df.copy()
    df2["_score"] = pd.to_numeric(df2.get(score_col), errors="coerce")
    df2["_published_dt"] = df2["published_at"].apply(parse_dt_safe)
    df2["_added_dt"] = df2["added_at"].apply(parse_dt_safe)

    if mode == "高分优先":
        df2 = df2.sort_values(["_score", "_published_dt"], ascending=[False, False], na_position="last")
    elif mode == "最新发布优先":
        df2 = df2.sort_values(["_published_dt", "_score"], ascending=[False, False], na_position="last")
    else:
        df2 = df2.sort_values(["_added_dt", "_score"], ascending=[False, False], na_position="last")

    return df2.drop(columns=["_score"], errors="ignore")


# ✅ batch match using match_one (per-paper)
def batch_match_llm(
    prompt: str,
    rows: List[Dict[str, Any]],
    model: str,
    sleep_s: float = 0.0,
    max_retry: int = 2,
) -> List[MatchResult]:
    results: List[MatchResult] = []
    n = len(rows)
    progress = st.progress(0)

    for i, r in enumerate(rows):
        uid = str(r.get("uid", ""))
        title = str(r.get("title", ""))
        abstract = str(r.get("abstract", ""))
        authors = str(r.get("authors", ""))

        last_err = None
        for _ in range(max_retry + 1):
            try:
                res = match_one(
                    prompt=prompt,
                    uid=uid,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    model=model,
                )
                results.append(res)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.8)

        if last_err is not None:
            results.append(MatchResult(
                uid=uid,
                match_score=0.0,
                match_reason=f"ERROR: {str(last_err)}",
                match_summary="",
                prompt_hash=compute_prompt_hash(prompt),
            ))

        if sleep_s > 0:
            time.sleep(float(sleep_s))

        progress.progress(int((i + 1) / max(1, n) * 100))

    return results


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Paper Recommender (Browse + Match)", layout="wide")
st.title("📚 Paper Recommender（Browse / Upload / arXiv / LLM Match）")
st.caption("Quality 请用 run_ingest_quality.py 跑；本 app 不做 Quality，只展示结果并提供 Match。")

st.sidebar.header("配置")
library_dir = st.sidebar.text_input("本地论文库目录", value=DEFAULT_LIBRARY_DIR)
csv_path = st.sidebar.text_input("CSV 路径", value=DEFAULT_CSV_PATH)
dirs = ensure_dirs(library_dir)
pdf_dir = dirs["pdf_dir"]

st.sidebar.markdown("---")
st.sidebar.subheader("LLM 设置")
use_llm = st.sidebar.checkbox("启用 GPT（仅用于 Match）", value=True)
model = st.sidebar.text_input("模型名", value="gpt-5.2")

st.sidebar.markdown("---")
st.sidebar.subheader("PDF 下载设置（Match 后可选）")
download_top_k = st.sidebar.slider("下载 Top K（按 match_score）", 0, 50, 10)
force_redownload = st.sidebar.checkbox("强制重新下载", value=False)

df_lib = load_csv(csv_path)
if len(df_lib) > 0:
    df_lib["source"] = df_lib.apply(infer_source, axis=1)
else:
    df_lib["source"] = None

tab_browse, tab_upload, tab_arxiv, tab_match = st.tabs([
    "👀 浏览（Quality/Match结果）",
    "⬆️ 批量上传 PDF",
    "🛰️ 抓取 arXiv",
    "🎯 LLM Match（按 Prompt 匹配）",
])


# ----------------------------
# Browse
# ----------------------------
with tab_browse:
    st.subheader("👀 浏览论文库（含 Quality 与 Match 字段）")

    if len(df_lib) == 0:
        st.info("你的库还是空的：先去抓 arXiv 或上传 PDF。")
    else:
        colA, colB, colC, colD, colE, colF = st.columns([1, 1, 1, 1, 1, 1])

        with colA:
            view = st.selectbox("浏览视角", ["Quality（quality_score）", "Match（match_score）"], index=0)
        with colB:
            sort_mode = st.selectbox("排序", ["高分优先", "最新发布优先", "最近添加优先"], index=0)
        with colC:
            status_filter = st.selectbox("状态", ["全部", "pending", "accepted", "rejected"], index=0)
        with colD:
            source_filter = st.selectbox("来源", ["全部", "arxiv", "local"], index=0)
        with colE:
            min_score = st.slider("最低分过滤", 70, 100, 80)
        with colF:
            only_today = st.checkbox("只看今天新发布", value=False)

        keyword = st.text_input("关键词过滤（title/abstract/quality_reason/match_summary）", value="")

        score_col = "quality_score" if view.startswith("Quality") else "match_score"

        df_show = df_lib.copy()
        df_show["_published_dt"] = df_show["published_at"].apply(parse_dt_safe)

        if status_filter != "全部":
            df_show = df_show[df_show["status"].fillna("").astype(str) == status_filter]

        if source_filter != "全部":
            df_show = df_show[df_show["source"].fillna("").astype(str) == source_filter]

        if only_today:
            df_show = df_show[df_show["_published_dt"].apply(lambda x: is_today(x, timezone.utc))]

        df_show["_score_num"] = pd.to_numeric(df_show.get(score_col), errors="coerce")
        df_scored = df_show[df_show["_score_num"].notna() & (df_show["_score_num"] >= float(min_score))].copy()
        df_unscored = df_show[df_show["_score_num"].isna()].copy()

        if keyword.strip():
            kw = keyword.strip().lower()

            def _hit(r: pd.Series) -> bool:
                fields = [
                    "title", "abstract",
                    "quality_reason", "match_reason", "match_summary",
                ]
                for f in fields:
                    if kw in str(r.get(f, "") or "").lower():
                        return True
                return False

            df_scored = df_scored[df_scored.apply(_hit, axis=1)]
            df_unscored = df_unscored[df_unscored.apply(_hit, axis=1)]

        df_scored = sort_for_browse(df_scored, score_col, sort_mode)
        df_unscored = sort_for_browse(df_unscored, score_col, "最新发布优先")

        st.markdown("### ✅ 已评分（当前视角）")

        # --- 列名去重（避免 pyarrow / streamlit duplicate columns 报错） ---
        def uniq_keep_order(cols):
            seen = set()
            out = []
            for c in cols:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        show_cols = [
            score_col,           # 当前视角分数（quality_score 或 match_score）
            "status",
            "source",
            "title",
            "primary_category",
            "published_at",
            "quality_score",
            "match_score",
            "arxiv_url",
            "pdf_path",
        ]
        show_cols = [c for c in show_cols if c in df_scored.columns]
        show_cols = uniq_keep_order(show_cols)

        # --- 分页控件 ---
        # colP1, colP2, colP3 = st.columns([1, 1, 2])
        # with colP1:
        #     page_size = st.selectbox("已评分：每页行数", [20, 50, 100, 200], index=1, key="scored_page_size")
        # total = len(df_scored)
        # max_page = max(1, (total + page_size - 1) // page_size)

        # with colP2:
        #     page = st.number_input("已评分：页码", min_value=1, max_value=max_page, value=1, step=1, key="scored_page")

        # with colP3:
        #     st.caption(f"共 {total} 条，{max_page} 页")

        # start = (page - 1) * page_size
        # end = start + page_size

        # st.dataframe(
        #     df_scored.iloc[start:end][show_cols],
        #     use_container_width=True,
        #     height=360,
        # )

        # if total > 0:
        #     st.caption(f"显示第 {start + 1} - {min(end, total)} 行 / 共 {total} 行")

        st.dataframe(df_scored[show_cols], use_container_width=True, height=320)
        
        st.markdown("### 🔍 详情查看")
        uid_pick = st.selectbox("选择 uid", options=df_show["uid"].astype(str).tolist()[:900])
        pick = df_lib[df_lib["uid"].astype(str) == str(uid_pick)]
        if len(pick) > 0:
            r = pick.iloc[0].to_dict()
            st.write(f"**Title**: {r.get('title')}")
            st.write(f"**Status**: {r.get('status')}")
            st.write(f"**Source**: {infer_source(pd.Series(r))}")
            st.write(f"**Published**: {r.get('published_at')}")
            st.write(f"**arXiv**: {r.get('arxiv_url')}")
            st.write(f"**PDF URL**: {r.get('pdf_url')}")
            st.write(f"**Local PDF**: {r.get('pdf_path')}")
            st.markdown("---")
            st.markdown("#### ⭐ Quality（来自 run_ingest_quality.py）")
            st.write(f"**quality_score**: {r.get('quality_score')}")
            st.write(f"**quality_reason**:\n\n{r.get('quality_reason')}")
            st.write(f"**quality_reviewed_at**: {r.get('quality_reviewed_at')}")
            st.markdown("#### 🎯 Match（来自本 app 的 llm_match.py）")
            st.write(f"**match_score**: {r.get('match_score')}")
            st.write(f"**match_reason**:\n\n{r.get('match_reason')}")
            st.write(f"**match_summary**:\n\n{r.get('match_summary')}")
            st.write(f"**match_prompt_hash**: {r.get('match_prompt_hash')}")
            st.markdown("#### 📝 Notes（abstract，可写备注）")
            st.write(r.get("abstract", ""))


# ----------------------------
# Upload PDFs
# ----------------------------
with tab_upload:
    st.subheader("⬆️ 批量上传 PDF（存档 + 可用于 Match）")
    st.write(f"保存目录：`{pdf_dir}`")
    st.write(f"CSV：`{csv_path}`")

    st.markdown("---")
    st.subheader("上传后自动 Quality（可选）")

    auto_quality = st.checkbox("上传完成后立即跑 LLM Quality", value=True)
    quality_model = st.text_input("Quality 模型名", value=model)
    quality_sleep = st.slider("Quality 每次调用间隔（秒）", 0.0, 1.0, 0.0, step=0.1)

    uploaded = st.file_uploader("选择 PDF（可多选）", type=["pdf"], accept_multiple_files=True)

    note = st.text_area(
        "可选：这批论文的统一备注（写关键词，Match 更准）",
        value="",
        height=100,
    )

    rename_mode = st.selectbox("标题策略", ["用文件名作为标题", "加前缀 + 文件名"], index=0)
    prefix = ""
    if rename_mode.startswith("加前缀"):
        prefix = st.text_input("标题前缀", value="My Paper: ")

    save_btn = st.button("💾 保存到论文库", type="primary")

    if save_btn:
        if not uploaded:
            st.error("你还没选 PDF。")
            st.stop()

        df_now = load_csv(csv_path)
        existing_uids = set(df_now["uid"].astype(str).tolist())

        rows: List[Dict[str, Any]] = []
        saved, skipped = 0, 0

        for f in uploaded:
            data = f.getvalue()
            uid = make_local_uid(data, f.name)

            if uid in existing_uids:
                skipped += 1
                continue

            out_name = safe_filename(uid) + ".pdf"
            out_path = os.path.join(pdf_dir, out_name)
            with open(out_path, "wb") as fp:
                fp.write(data)

            title = f.name
            if rename_mode.startswith("加前缀"):
                title = f"{prefix}{f.name}"

            rows.append(local_row_from_upload(uid, title, out_path, note))
            saved += 1

        if rows:
            upsert_rows(csv_path, rows)

        if auto_quality and rows:
            if not use_llm:
                st.warning("你在侧边栏关闭了 GPT，因此跳过 Quality。")
            else:
                with st.status(f"Quality 评估中...（{len(rows)} 篇）", expanded=True) as status:
                    judgements = batch_quality_llm(
                        rows=rows,  # 直接用刚上传的 rows（包含 uid/title/abstract）
                        model=quality_model,
                        sleep_s=float(quality_sleep),
                        max_retry=2,
                    )

                    updates = []
                    for j in judgements:
                        updates.append({
                            "uid": j.uid,
                            "quality_score": j.quality_score,
                            "status": j.status,  # accepted / rejected
                            "quality_reason": j.quality_reason,
                            "quality_reviewed_at": now_iso(),
                        })

                    if updates:
                        upsert_rows(csv_path, updates)

                    status.update(label="✅ Quality 写回完成", state="complete")

                st.success("✅ 本地上传论文已完成 Quality 评分与筛选。")


        st.success(f"✅ 上传完成：保存 {saved} 篇，跳过重复 {skipped} 篇")


# ----------------------------
# Fetch arXiv
# ----------------------------
with tab_arxiv:
    st.subheader("🛰️ 抓取 arXiv 最新论文（只写入 CSV，Quality 交给脚本）")

    cats = st.multiselect(
        "arXiv 分类",
        options=["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"],
        default=["cs.AI", "cs.LG"],
    )
    fetch_n = st.slider("抓取数量", 10, 500, 150, step=10)
    last_days = st.slider("只保留最近 N 天", 1, 60, 14)

    run_fetch = st.button("🚀 抓取并写入 CSV", type="primary")

    if run_fetch:
        if not cats:
            st.error("至少选择一个分类。")
            st.stop()

        with st.status("抓取中...", expanded=True) as status:
            papers = fetch_arxiv_latest(categories=cats, max_results=fetch_n)
            papers = [p for p in papers if within_last_days(p.published_at, last_days)]
            rows = [paper_to_row_arxiv(p) for p in papers]
            upsert_rows(csv_path, rows)
            status.update(label=f"✅ 写入 {len(rows)} 篇到 {csv_path}", state="complete")

        st.success("完成！接下来你跑 run_ingest_quality.py 做 accepted/rejected 即可。")


# ----------------------------
# Match (llm_match.py)
# ----------------------------
with tab_match:
    st.subheader("🎯 LLM Match（按 Prompt 匹配）")
    st.caption("本页只做 Match：调用 llm_match.match_one 写回 match_*。Quality 不在 app 内执行。")

    df_now = load_csv(csv_path)
    if len(df_now) == 0:
        st.info("库为空。先抓 arXiv 或上传 PDF。")
    else:
        colA, colB, colC, colD, colE = st.columns([1, 1, 1, 1, 1])

        with colA:
            match_scope = st.selectbox("匹配范围", ["只匹配 accepted", "匹配全部（不管 status）"], index=0)
        with colB:
            max_candidates = st.slider("最多参与匹配篇数", 10, 1000, 200, step=10)
        with colC:
            only_new_prompt = st.checkbox("只处理没跑过该 prompt 的论文（hash不同）", value=True)
        with colD:
            min_match_download = st.slider("match_score ≥ X 才自动下载 PDF（arXiv）", 0, 100, 85)
        with colE:
            per_call_sleep = st.slider("每次调用间隔（秒，防限速）", 0.0, 1.0, 0.0, step=0.1)

        prompt = st.text_area(
            "你的需求（prompt）",
            value="我想找关于 agent + RL 工具链（训练/评估/部署）的最新工作，偏工程落地、可复现，最好有代码。",
            height=140,
        )

        run_match = st.button("🚀 运行 Match", type="primary")

        if run_match:
            if not use_llm:
                st.error("你在侧边栏关闭了 GPT。")
                st.stop()

            prompt_hash = compute_prompt_hash(prompt)

            df_scope = df_now.copy()
            df_scope["source"] = df_scope.apply(infer_source, axis=1)

            if match_scope.startswith("只匹配 accepted"):
                df_scope = df_scope[df_scope["status"].fillna("").astype(str) == "accepted"]

            if only_new_prompt:
                df_scope = df_scope[
                    df_scope["match_prompt_hash"].isna()
                    | (df_scope["match_prompt_hash"].astype(str).str.strip() != prompt_hash)
                ]

            # 优先今天新、再最新发布
            df_scope["_published_dt"] = df_scope["published_at"].apply(parse_dt_safe)
            df_scope["is_today"] = df_scope["_published_dt"].apply(lambda x: is_today(x, timezone.utc))
            df_scope = df_scope.sort_values(["is_today", "_published_dt"], ascending=[False, False], na_position="last")
            df_scope = df_scope.head(max_candidates)

            rows_for_match: List[Dict[str, Any]] = []
            for _, r in df_scope.iterrows():
                rows_for_match.append({
                    "uid": str(r.get("uid", "")),
                    "title": str(r.get("title", "")),
                    "abstract": str(r.get("abstract", "")),
                    "authors": str(r.get("authors", "")),
                })

            with st.status(f"LLM Match 匹配中...（{len(rows_for_match)} 篇）", expanded=True) as status:
                results = batch_match_llm(
                    prompt=prompt,
                    rows=rows_for_match,
                    model=model,
                    sleep_s=float(per_call_sleep),
                    max_retry=2,
                )

                updates = []
                for rr in results:
                    updates.append({
                        "uid": rr.uid,
                        "match_score": rr.match_score,
                        "match_reason": rr.match_reason,
                        "match_summary": rr.match_summary,
                        "match_prompt_hash": rr.prompt_hash,
                    })

                if updates:
                    upsert_rows(csv_path, updates)

                status.update(label="✅ Match 写回完成", state="complete")

            # 自动下载：按 match_score
            if download_top_k > 0:
                df_after = load_csv(csv_path)
                df_after["source"] = df_after.apply(infer_source, axis=1)
                df_after["match_num"] = pd.to_numeric(df_after.get("match_score"), errors="coerce")

                df_dl = df_after[
                    (df_after["source"] == "arxiv")
                    & (df_after["match_num"].notna())
                    & (df_after["match_num"] >= float(min_match_download))
                ].copy()

                df_dl = df_dl.sort_values("match_num", ascending=False).head(int(download_top_k))

                ok_cnt = 0
                updated_rows = []
                for _, r in df_dl.iterrows():
                    uid = str(r["uid"])
                    pdf_url = str(r.get("pdf_url", "") or "")
                    if not pdf_url.startswith("http"):
                        continue

                    out_path = os.path.join(pdf_dir, f"{safe_filename(uid)}.pdf")
                    if os.path.exists(out_path) and not force_redownload:
                        updated_rows.append({"uid": uid, "pdf_path": out_path})
                        continue

                    ok = download_pdf(pdf_url, out_path)
                    if ok:
                        ok_cnt += 1
                        updated_rows.append({"uid": uid, "pdf_path": out_path})

                if updated_rows:
                    upsert_rows(csv_path, updated_rows)

                st.info(f"📥 自动下载完成：{ok_cnt} 篇（按 match_score）")

            st.success("✅ Match 完成！去『浏览』页切 Match 视角刷你真正要的论文。")
