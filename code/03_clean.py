# -*- coding: utf-8 -*-
"""
Data cleaning: merge raw data from multiple platforms → denoise → deduplicate → drop ads → drop copied scenic-area blurbs → unify fields.

Input: every .csv / .xlsx under data/raw/ (column names detected automatically)
Output: data/clean/comments_clean.csv
        outputs/logs/清洗记录.csv
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import content_words, split_sentences, norm_text

# ============================================================
# Column-name aliases: whichever platform or tool exported the file, it is detected automatically
# ============================================================
COL_ALIASES = {
    "content": ["content", "comment", "text", "review", "body", "评论内容", "评价内容", "点评内容",
                "评论", "评价", "点评", "内容", "评论正文", "游记内容", "正文", "commentContent",
                "reviewContent", "comment_content"],
    "rating": ["rating", "score", "star", "stars", "评分", "星级", "打分", "分数", "点评星级",
               "overallScore", "commentScore", "满意"],
    "date": ["date", "time", "publishTime", "publishDate", "createTime", "commentTime", "评论时间",
             "点评时间", "发表时间", "时间", "日期", "publishdate", "publish_time"],
    "poi_name": ["poi_name", "poi", "site", "scenic", "景点", "景区", "景点名称", "景区名称",
                 "标题", "title", "source_page", "页面"],
    "comment_id": ["comment_id", "id", "commentId", "reviewId", "评论id", "序号"],
    "platform": ["platform", "平台", "来源", "source"],
    "useful": ["useful", "usefulCount", "点赞", "有用", "有用数", "赞同数"],
}

# These files are not comment data, skip them while cleaning
SKIP_FILES = {"site_attributes_amap.csv", "site_attributes_manual.csv"}


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    low = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for std, alts in COL_ALIASES.items():
        for a in alts:
            if a.lower() in low:
                rename[low[a.lower()]] = std
                break
    df = df.rename(columns=rename)
    if "content" not in df.columns:
        return pd.DataFrame()
    for col in ("rating", "date", "poi_name", "comment_id", "platform", "useful"):
        if col not in df.columns:
            df[col] = ""
    return df[["platform", "poi_name", "comment_id", "content",
               "rating", "date", "useful"]]


def read_csv_compat(path):
    """
    Read a CSV robustly:
      1) try the UTF-8 / UTF-8-BOM / GBK encodings automatically
         -- Excel's "Save as CSV" and Octoparse exports default to GBK (ANSI) on Chinese
            Windows, where hard-coding utf-8 raises UnicodeDecodeError outright
      2) support the bad-line argument of both new and old pandas (on_bad_lines vs error_bad_lines)
    """
    import io
    text = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            text = path.read_bytes().decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = path.read_bytes().decode("utf-8", errors="replace")
        print(f"  [WARNING] the encoding of {path.name} could not be detected, decoded forcibly, Chinese may be garbled; "
              f"suggest saving it as UTF-8 in Notepad")

    buf = io.StringIO(text)
    try:
        return pd.read_csv(buf, on_bad_lines="skip", dtype=str)
    except TypeError:                                  # pandas < 1.3
        buf.seek(0)
        return pd.read_csv(buf, error_bad_lines=False, warn_bad_lines=False, dtype=str)


def load_raw() -> pd.DataFrame:
    files = sorted(list(C.DATA_RAW.glob("*.csv")) + list(C.DATA_RAW.glob("*.xlsx")))
    files = [f for f in files if f.name not in SKIP_FILES and not f.name.startswith("~$")]
    if not files:
        print(f"[ERROR] no csv/xlsx file found in {C.DATA_RAW}.")
        print("       Finish the stage 3 data collection first, or put the Octoparse exports there.")
        sys.exit(1)

    frames = []
    for f in files:
        try:
            df = pd.read_excel(f) if f.suffix.lower() == ".xlsx" else read_csv_compat(f)
            if df.empty:
                continue
            raw_cols = list(df.columns)
            df = map_columns(df)
            if df.empty:
                print(f"  [skip] {f.name}: no comment-content column found (existing columns {raw_cols})")
                continue
            df["source_file"] = f.name
            frames.append(df)
            print(f"  [read] {f.name}: {len(df)} rows")
        except Exception as e:
            print(f"  [skip] {f.name}: {type(e).__name__} {e}")
    if not frames:
        print("[ERROR] none of the files has a usable comment-content column. Check the column names, or tell me your column names.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def normalize(text: str) -> str:
    t = str(text)
    t = re.sub(r"<[^>]+>", " ", t)                       # HTML tags
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)          # links
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", " ", t)  # emoji
    t = t.replace("\u3000", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_date(s):
    s = str(s).strip()
    if not s or s.lower() in ("nan", "nat", "none"):
        return pd.NaT
    s = s.replace("年", "-").replace("月", "-").replace("日", "")
    s = re.sub(r"[./]", "-", s)
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})-(\d{1,2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), 1)
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def parse_rating(s):
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 < v <= 5 else None


def drop_repeated_sentences(texts, log):
    """
    Follows the original study: a sentence repeated >= 3 times with >= 6 words after
    stop-word removal is taken as a scenic-area blurb copied by tourists, so it is deleted.
    """
    cnt = Counter()
    for t in texts:
        for s in split_sentences(t):
            n = norm_text(s)
            if len(content_words(s)) >= C.DUP_SENT_MIN_WORDS and len(n) >= 8:
                cnt[n] += 1
    bad = {k for k, v in cnt.items() if v >= C.DUP_SENT_MIN_COUNT}
    log.append(("疑似复制的句子种类数", len(bad)))
    print(f"  found {len(bad)} distinct sentences that were copied repeatedly (each one will be removed)")
    for k, _ in cnt.most_common(5):
        if cnt[k] >= C.DUP_SENT_MIN_COUNT:
            print(f"    appears {cnt[k]:>4} times: {k[:40]}")

    out = []
    removed = 0
    for t in texts:
        keep = [s for s in split_sentences(t) if norm_text(s) not in bad]
        new = "。".join(keep)
        if not keep:
            removed += 1
        out.append(new)
    log.append(("因简介复制而清空的评论数", removed))
    return out


def main():
    log = []
    print("=" * 60)
    print("Reading the raw data...")
    df = load_raw()
    log.append(("原始评论总数", len(df)))
    print(f"{len(df)} rows in total before cleaning")

    # ---- 1. Basic cleaning ----
    df["content"] = df["content"].astype(str).map(normalize)
    df = df[df["content"].str.len() >= C.MIN_LEN]
    log.append((f"去除少于 {C.MIN_LEN} 字的评论后", len(df)))
    print(f"{len(df)} rows after dropping comments that are too short")

    df["content"] = df["content"].str.slice(0, C.MAX_LEN)

    # ---- 2. Drop ads ----
    pat = "|".join(re.escape(w) for w in C.AD_WORDS)
    adv = df["content"].str.contains(pat, case=False, na=False, regex=True)
    df = df[~adv]
    log.append(("去除广告/引流评论后", len(df)))
    print(f"{len(df)} rows after dropping ads ({int(adv.sum())} removed)")

    # ---- 3. Deduplicate ----
    df["_norm"] = df["content"].map(norm_text)
    before = len(df)
    df = df[df["_norm"].str.len() > 0]
    df = df.drop_duplicates(subset=["_norm"])
    log.append(("完全重复评论删除数", before - len(df)))
    print(f"{len(df)} rows after deduplication ({before - len(df)} removed)")

    # The same user copy-pasting across platforms: near-deduplicate on the first 60 characters
    before = len(df)
    df["_key"] = df["_norm"].str.slice(0, 60)
    df = df.drop_duplicates(subset=["_key"], keep="first")
    log.append(("近似重复评论删除数", before - len(df)))
    print(f"{len(df)} rows after near-deduplication")

    # ---- 4. Remove copied scenic-area blurbs ----
    df["content"] = drop_repeated_sentences(df["content"].tolist(), log)
    df = df[df["content"].str.len() >= C.MIN_LEN]
    log.append(("删除简介复制句后", len(df)))
    print(f"{len(df)} rows after removing copied blurb sentences")

    # ---- 5. Field normalization ----
    df["date"] = df["date"].map(parse_date)
    df["rating"] = df["rating"].map(parse_rating)
    df["platform"] = df["platform"].replace("", pd.NA).fillna("未知")
    df["poi_name"] = df["poi_name"].fillna("").astype(str)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["ym"] = df["date"].dt.to_period("M").astype(str).replace("NaT", "")
    df["n_char"] = df["content"].str.len()
    df = df.drop(columns=["_norm", "_key"]).reset_index(drop=True)
    df["cid"] = ["C%06d" % i for i in range(len(df))]

    # ---- 6. Output ----
    df.to_csv(C.F_COMMENTS_CLEAN, index=False, encoding="utf-8-sig")
    pd.DataFrame(log, columns=["清洗环节", "剩余/删除数量"]).to_csv(
        C.OUT_LOG / "清洗记录.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"Finally {len(df)} rows → {C.F_COMMENTS_CLEAN}")
    print(f"Columns: {list(df.columns)}")
    print(f"Comment length: median {df['n_char'].median():.0f} characters, mean {df['n_char'].mean():.0f} characters")
    if df["date"].notna().any():
        print(f"Time range: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"Platform distribution:\n{df['platform'].value_counts().to_string()}")
    print(f"Scenic-area label distribution:\n{df['poi_name'].value_counts().head(10).to_string()}")
    print("\nTip: if the comment count collapses, MIN_LEN or the ad-word filter is probably too aggressive; tune it in config.py.")


if __name__ == "__main__":
    main()
