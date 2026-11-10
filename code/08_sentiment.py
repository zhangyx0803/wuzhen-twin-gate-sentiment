# -*- coding: utf-8 -*-
"""
情感分析（两层设计）。

【评论级】调用百度智能云 NLP 情感倾向分析（或用 SnowNLP 离线兜底）
    输出 0=消极 / 1=中立 / 2=积极 + 置信度 + 正负概率
【属性级】本地情感词典逐句打分（不消耗任何 API 额度）
    输出 -1 ~ +1 的连续情感值

工程特性：
    * 断点续跑：中断后重跑不会重复调用 API（结果实时落盘到 jsonl）
    * 自动限速：遵守 QPS，触发限流自动等待重试
    * 额度用尽自动停止并提示，不会白跑
    * 超长文本按字节安全截断（避免把中文截成半个字）

输入：data/clean/comments_clean.csv、data/clean/sentences_clf_label.csv
输出：data/clean/comments_sentiment.csv
      data/clean/sentences_sentiment.csv
      outputs/tables/情感分析_一致性检验.txt（如果你做了人工抽检）
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
# 一、百度 NLP
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
        raise RuntimeError("请先在 config.py 里填写 BAIDU_API_KEY / BAIDU_SECRET_KEY")
    r = requests.post("https://aip.baidubce.com/oauth/2.0/token",
                      params={"grant_type": "client_credentials",
                              "client_id": C.BAIDU_API_KEY,
                              "client_secret": C.BAIDU_SECRET_KEY},
                      timeout=20)
    js = r.json()
    if "access_token" not in js:
        raise RuntimeError(f"获取 token 失败：{js}")
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
    """返回 dict 或 None（失败时）"""
    body = {"text": safe_truncate(str(text), C.BAIDU_MAX_BYTES)}
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify"
    for attempt in range(3):
        try:
            r = session.post(url, params={"charset": "UTF-8", "access_token": token},
                             json=body, timeout=20)
            js = r.json()
            if "error_code" in js:
                code = js["error_code"]
                if code in (18,):                       # QPS 超限
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if code in (17, 19):                    # 日/总量额度用尽
                    raise QuotaExhausted(f"额度用尽 (error_code={code}: {js.get('error_msg')})")
                if code in (110, 111):                  # token 失效
                    if TOKEN_CACHE.exists():            # 兼容 Python 3.7
                        TOKEN_CACHE.unlink()
                    token = get_token()
                    continue
                print(f"    [接口错误] {js}")
                return None
            item = js["items"][0]
            return {"sentiment": int(item["sentiment"]),
                    "confidence": float(item["confidence"]),
                    "positive_prob": float(item["positive_prob"]),
                    "negative_prob": float(item["negative_prob"])}
        except QuotaExhausted:
            raise
        except Exception as e:
            print(f"    [请求异常 {type(e).__name__}] {e}")
            time.sleep(1.0 + attempt)
    return None


class QuotaExhausted(Exception):
    pass


# ============================================================
# 二、SnowNLP 离线兜底
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
        p = float(SnowNLP(str(text)).sentiments)   # 0~1，越大越积极
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
# 三、本地情感词典（属性级）
# ============================================================
NEG_WINDOW = 3


def dict_sentiment(sentence: str):
    """
    基于情感词典 + 程度副词 + 否定词规则，给句子打 -1 ~ +1 的情感分。
    返回 (score, n_hits)
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
    # 用 hits 做长度归一，再做非线性压缩到 -1~1
    raw = score / np.sqrt(hits)
    return float(np.tanh(raw / 2.0)), hits


# ============================================================
# 四、主流程
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
        print(f"  发现缓存 {len(done)} 条，将跳过这些 (断点续跑)")

    method = C.SENTIMENT_METHOD
    todo = [r for r in df.itertuples(index=False) if r.cid not in done]
    print(f"  待分析 {len(todo)} 条（方式：{method}）")

    sess = None
    token = None
    if method == "baidu" and todo:
        import requests                     # 只有真正调 API 时才需要 requests
        token = get_token()
        sess = requests.Session()
    elif method == "snownlp" and not _SNOW_OK:
        print("  SnowNLP 未安装，且 SENTIMENT_METHOD='snownlp'。")
        print("  解决：pip install snownlp   或   把 config.py 的 SENTIMENT_METHOD 改成 'baidu'")
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
                print(f"    进度 {k}/{len(todo)}  成功 {ok} 失败 {fail}")
    except QuotaExhausted as e:
        fout.flush()
        print(f"\n!! 百度接口额度用尽：{e}")
        print(f"   已处理 {ok} 条并落盘，不会丢。你有三个选择：")
        print("   ① 等到次日额度刷新后重跑本脚本（会从断点继续）")
        print("   ② 把 config.py 的 SENTIMENT_METHOD 改成 'snownlp' 兜底剩余部分")
        print("   ③ 见手册 6.3 路线 D：用大模型 API 批量打标")
    finally:
        fout.close()

    # ---------- 合并 ----------
    s = pd.DataFrame(list(done.values()))
    out = df.merge(s, on="cid", how="left")
    out["sentiment_label"] = out["sentiment"].map(SENT_LABEL)
    out["is_positive"] = (out["sentiment"] == 2).astype("Int64")
    return out


def run_sentence_sentiment():
    src = C.F_SENT_CLF if C.F_SENT_CLF.exists() else C.F_SENT_LABEL_RULE
    if not src.exists():
        print("  找不到句级文件，跳过属性级情感。请先运行 04/05/07 脚本。")
        return None
    sdf = pd.read_csv(src, encoding="utf-8-sig")
    print(f"  句级情感（本地词典）: {len(sdf)} 句")
    res = [dict_sentiment(s) for s in sdf["sentence"].astype(str)]
    sdf["sent_score"] = [round(a, 4) for a, _ in res]
    sdf["sent_hits"] = [b for _, b in res]
    sdf["sent_polar"] = np.where(sdf["sent_score"] > 0.1, 1,
                                 np.where(sdf["sent_score"] < -0.1, -1, 0))
    sdf.to_csv(C.F_SENT_SENTIMENT, index=False, encoding="utf-8-sig")
    return sdf


def consistency_check(df):
    """如果你做了人工情感抽检，这里自动算 Kappa 一致度"""
    fp = C.DATA_LABEL / "sentiment_check.xlsx"
    if not fp.exists():
        print("\n[提示] 想提高论文可信度，建议做一次情感分析人工抽检：")
        print("       从 comments_sentiment.csv 随机抽 300 条，人工标 积极/中立/消极，")
        print(f"       保存为 {fp}（列名：cid, 人工情感），本脚本会自动算 Kappa。")
        return
    try:
        from sklearn.metrics import cohen_kappa_score, accuracy_score
    except ImportError:
        return
    chk = pd.read_excel(fp)
    m = chk.merge(df[["cid", "sentiment_label"]], on="cid", how="left")
    m = m.dropna(subset=["人工情感", "sentiment_label"])
    if len(m) < 30:
        print("[一致性检验] 有效样本 <30，跳过")
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
        print("找不到 comments_clean.csv，请先运行 03、04 脚本。")
        sys.exit(1)
    df = pd.read_csv(C.F_COMMENTS_CLEAN, encoding="utf-8-sig")

    print("=" * 60)
    print("【1/2】评论级情感分析")
    out = run_comment_sentiment(df)
    out.to_csv(C.F_SENTIMENT, index=False, encoding="utf-8-sig")

    print("\n情感分布：")
    print(out["sentiment_label"].value_counts(dropna=False).to_string())
    print("\n分景区情感分布（占比）：")
    if "site" in out.columns:
        tab = pd.crosstab(out["site"], out["sentiment_label"], normalize="index")
        print((tab * 100).round(2).to_string())

    print("\n【2/2】句级属性情感（本地词典）")
    run_sentence_sentiment()

    consistency_check(out)

    print("=" * 60)
    print(f"评论级情感 → {C.F_SENTIMENT}")
    print(f"句级情感   → {C.F_SENT_SENTIMENT}")


if __name__ == "__main__":
    main()
