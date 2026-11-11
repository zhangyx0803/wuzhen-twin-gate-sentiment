# -*- coding: utf-8 -*-
"""
Sentiment analysis (two-layer design).

[Review level] Calls Baidu AI Cloud NLP sentiment classification (or falls back to offline SnowNLP)
    Outputs 0=negative / 1=neutral / 2=positive + confidence + positive/negative probabilities
[Aspect level] Scores sentence by sentence with a local sentiment lexicon (consumes no API quota)
    Outputs a continuous sentiment value in -1 ~ +1

Engineering features:
    * Resumable: re-running after an interruption never calls the API twice (results are flushed to jsonl in real time)
    * Automatic rate limiting: respects QPS and waits to retry when rate limited
    * Stops with a message when the quota runs out, so no work is wasted
    * Over-long text is truncated safely by bytes (never splits a Chinese character in half)

Input: data/clean/comments_clean.csv, data/clean/sentences_clf_label.csv
Output: data/clean/comments_sentiment.csv
      data/clean/sentences_sentiment.csv
      outputs/tables/情感分析_一致性检验.txt (if you did a manual spot check)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import tokenize

CACHE_COMMENT = C.DATA_CLEAN / ".baidu_cache.jsonl"
TOKEN_CACHE = C.DATA_RAW / ".baidu_token.json"

SENT_LABEL = {0: "消极", 1: "中立", 2: "积极"}


# ============================================================
# 1. Baidu NLP
# ============================================================
def get_token():
    import requests
    if TOKEN_CACHE.exists():
        try:
            d = json.loads(TOKEN_CACHE.read_text(encoding="utf-8"))
            if d.get("expire_at", 0) > time.time() + 3600:
                return d["access_token"]
        except Exception:
            pass
    if not C.BAIDU_API_KEY or not C.BAIDU_SECRET_KEY:
        raise RuntimeError("Please fill in BAIDU_API_KEY / BAIDU_SECRET_KEY in config.py first")
    r = requests.post("https://aip.baidubce.com/oauth/2.0/token",
                      params={"grant_type": "client_credentials",
                              "client_id": C.BAIDU_API_KEY,
                              "client_secret": C.BAIDU_SECRET_KEY},
                      timeout=20)
    js = r.json()
    if "access_token" not in js:
        raise RuntimeError(f"Failed to obtain token: {js}")
    TOKEN_CACHE.write_text(json.dumps({
        "access_token": js["access_token"],
        "expire_at": time.time() + int(js.get("expires_in", 2592000)) - 600,
    }), encoding="utf-8")
    return js["access_token"]


def safe_truncate(text: str, max_bytes: int) -> str:
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    return b[:max_bytes].decode("utf-8", errors="ignore")


def baidu_sentiment(text: str, token: str, session):
    """Return a dict, or None on failure"""
    body = {"text": safe_truncate(str(text), C.BAIDU_MAX_BYTES)}
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify"
    for attempt in range(3):
        try:
            r = session.post(url, params={"charset": "UTF-8", "access_token": token},
                             json=body, timeout=20)
            js = r.json()
            if "error_code" in js:
                code = js["error_code"]
                if code in (18,):                       # QPS limit exceeded
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if code in (17, 19):                    # daily/total quota exhausted
                    raise QuotaExhausted(f"Quota exhausted (error_code={code}: {js.get('error_msg')})")
                if code in (110, 111):                  # token invalid
                    if TOKEN_CACHE.exists():            # Python 3.7 compatibility
                        TOKEN_CACHE.unlink()
                    token = get_token()
                    continue
                print(f"    [API error] {js}")
                return None
            item = js["items"][0]
            return {"sentiment": int(item["sentiment"]),
                    "confidence": float(item["confidence"]),
                    "positive_prob": float(item["positive_prob"]),
                    "negative_prob": float(item["negative_prob"])}
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"    [Request exception {type(e).__name__}] {e}")
            time.sleep(1.0 + attempt)
    return None


class QuotaExhausted(Exception):
    pass


# ============================================================
# 2. SnowNLP offline fallback
# ============================================================
_SNOW_OK = True
try:
    from snownlp import SnowNLP
except Exception:
    _SNOW_OK = False


def snownlp_sentiment(text: str):
    if not _SNOW_OK:
        return None
    try:
        p = float(SnowNLP(str(text)).sentiments)   # 0~1, the larger the more positive
    except Exception:
        return None
    if p >= 0.6:
        s = 2
    elif p <= 0.4:
        s = 0
    else:
        s = 1
    return {"sentiment": s, "confidence": abs(p - 0.5) * 2,
            "positive_prob": p, "negative_prob": 1 - p}


# ============================================================
# 3. Local sentiment lexicon (aspect level)
# ============================================================
NEG_WINDOW = 3


def dict_sentiment(sentence: str):
    """
    Score a sentence from -1 to +1 with a sentiment lexicon + degree adverbs + negation rules.
    Returns (score, n_hits).
    """
    toks = tokenize(str(sentence))
    if not toks:
        return 0.0, 0
    score, hits = 0.0, 0
    for i, w in enumerate(toks):
        base = 0.0
        if w in C.POS_WORDS:
            base = C.POS_WORDS[w]
        elif w in C.NEG_WORDS:
            base = -C.NEG_WORDS[w]
        else:
            continue
        hits += 1
        deg = 1.0
        if i > 0 and toks[i - 1] in C.DEGREE_WORDS:
            deg = C.DEGREE_WORDS[toks[i - 1]]
        neg = False
        for j in range(max(0, i - NEG_WINDOW), i):
            if any(toks[j].startswith(n) for n in C.NEGATION_WORDS):
                neg = True
                break
        v = base * deg
        score += (-v) if neg else v
    if hits == 0:
        return 0.0, 0
    # Normalize by hits for length, then compress non-linearly to -1~1
    raw = score / np.sqrt(hits)
    return float(np.tanh(raw / 2.0)), hits


# ============================================================
# 4. Main pipeline
# ============================================================
def run_comment_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    done = {}
    if CACHE_COMMENT.exists():
        for line in CACHE_COMMENT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                done[d["cid"]] = d
            except Exception:
                pass
        print(f"  Found {len(done)} cached records; skipping them (resume)")

    method = C.SENTIMENT_METHOD
    todo = [r for r in df.itertuples(index=False) if r.cid not in done]
    print(f"  {len(todo)} records to analyze (method: {method})")

    sess = None
    token = None
    if method == "baidu" and todo:
        import requests                     # requests is only needed when actually calling the API
        token = get_token()
        sess = requests.Session()
    elif method == "snownlp" and not _SNOW_OK:
        print("  SnowNLP is not installed, and SENTIMENT_METHOD='snownlp'.")
        print("  Fix: pip install snownlp   or   change SENTIMENT_METHOD in config.py to 'baidu'")
        sys.exit(1)

    fout = CACHE_COMMENT.open("a", encoding="utf-8")
    ok = fail = 0
    try:
        for k, r in enumerate(todo, 1):
            if method == "baidu":
                res = baidu_sentiment(r.content, token, sess)
                time.sleep(C.BAIDU_QPS_SLEEP)
            else:
                res = snownlp_sentiment(r.content)
            rec = {"cid": r.cid, "method": method}
            if res:
                rec.update(res)
                ok += 1
            else:
                rec.update({"sentiment": None, "confidence": None,
                            "positive_prob": None, "negative_prob": None})
                fail += 1
            done[r.cid] = rec
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if k % 200 == 0:
                fout.flush()
                print(f"    Progress {k}/{len(todo)}  ok {ok}  failed {fail}")
    except QuotaExhausted as e:
        fout.flush()
        print(f"\n!! Baidu API quota exhausted: {e}")
        print(f"   {ok} records have been processed and saved, nothing is lost. You have three options:")
        print("   1) Wait for the quota to reset the next day and re-run this script (it resumes from the checkpoint)")
        print("   2) Change SENTIMENT_METHOD in config.py to 'snownlp' for the remainder")
        print("   3) See route D in section 6.3 of the manual: batch-label with an LLM API")
    finally:
        fout.close()

    # ---------- Merge ----------
    s = pd.DataFrame(list(done.values()))
    out = df.merge(s, on="cid", how="left")
    out["sentiment_label"] = out["sentiment"].map(SENT_LABEL)
    out["is_positive"] = (out["sentiment"] == 2).astype("Int64")
    return out


def run_sentence_sentiment():
    src = C.F_SENT_CLF if C.F_SENT_CLF.exists() else C.F_SENT_LABEL_RULE
    if not src.exists():
        print("  Sentence-level file not found; skipping aspect-level sentiment. Run scripts 04/05/07 first.")
        return None
    sdf = pd.read_csv(src, encoding="utf-8-sig")
    print(f"  Sentence-level sentiment (local lexicon): {len(sdf)} sentences")
    res = [dict_sentiment(s) for s in sdf["sentence"].astype(str)]
    sdf["sent_score"] = [round(a, 4) for a, _ in res]
    sdf["sent_hits"] = [b for _, b in res]
    sdf["sent_polar"] = np.where(sdf["sent_score"] > 0.1, 1,
                                 np.where(sdf["sent_score"] < -0.1, -1, 0))
    sdf.to_csv(C.F_SENT_SENTIMENT, index=False, encoding="utf-8-sig")
    return sdf


def consistency_check(df):
    """Compute Cohen's Kappa automatically if you did a manual sentiment spot check"""
    fp = C.DATA_LABEL / "sentiment_check.xlsx"
    if not fp.exists():
        print("\n[Tip] To strengthen the paper, consider a manual spot check of the sentiment analysis:")
        print("       Randomly sample 300 reviews from comments_sentiment.csv and label them positive/neutral/negative by hand,")
        print(f"       save them as {fp} (columns: cid, 人工情感); this script computes Kappa automatically.")
        return
    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score
    except ImportError:
        return
    chk = pd.read_excel(fp)
    m = chk.merge(df[["cid", "sentiment_label"]], on="cid", how="left")
    m = m.dropna(subset=["人工情感", "sentiment_label"])
    if len(m) < 30:
        print("[Consistency check] Valid sample <30, skipping")
        return
    acc = accuracy_score(m["人工情感"], m["sentiment_label"])
    kap = cohen_kappa_score(m["人工情感"], m["sentiment_label"])
    txt = (f"情感分析人工抽检一致性\n"
           f"{'=' * 40}\n"
           f"抽检样本数 : {len(m)}\n"
           f"准确率     : {acc:.4f}\n"
           f"Cohen's Kappa: {kap:.4f}\n\n"
           f"解释：Kappa > 0.8 几乎完全一致；0.6-0.8 高度一致；0.4-0.6 中等一致。\n"
           f"论文里写：'随机抽取 {len(m)} 条评论进行人工情感标注，与模型结果的一致率为 "
           f"{acc * 100:.1f}%，Cohen's Kappa 为 {kap:.3f}，表明自动情感分析结果可靠。'\n")
    (C.OUT_TABLE / "情感分析_一致性检验.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


def main():
    if not C.F_COMMENTS_CLEAN.exists():
        print("comments_clean.csv not found; run scripts 03 and 04 first.")
        sys.exit(1)
    df = pd.read_csv(C.F_COMMENTS_CLEAN, encoding="utf-8-sig")

    print("=" * 60)
    print("[1/2] Review-level sentiment analysis")
    out = run_comment_sentiment(df)
    out.to_csv(C.F_SENTIMENT, index=False, encoding="utf-8-sig")

    print("\nSentiment distribution:")
    print(out["sentiment_label"].value_counts(dropna=False).to_string())
    print("\nSentiment distribution by scenic area (shares):")
    if "site" in out.columns:
        tab = pd.crosstab(out["site"], out["sentiment_label"], normalize="index")
        print((tab * 100).round(2).to_string())

    print("\n[2/2] Sentence-level aspect sentiment (local lexicon)")
    run_sentence_sentiment()

    consistency_check(out)

    print("=" * 60)
    print(f"Review-level sentiment → {C.F_SENTIMENT}")
    print(f"Sentence-level sentiment   → {C.F_SENT_SENTIMENT}")


if __name__ == "__main__":
    main()
