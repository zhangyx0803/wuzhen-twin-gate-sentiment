# -*- coding: utf-8 -*-
"""
Rule-based classifier built on a domain lexicon (baseline + sampling basis for annotation).

Input: data/clean/sentences.csv
Output: data/clean/sentences_rule_label.csv
        data/clean/comments_rule_label.csv
        outputs/tables/词典命中统计.csv
        outputs/logs/词典样例_<类别>.txt (30 random rows per category, to check the lexicon quality)
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

random.seed(C.RANDOM_STATE)

# Precompile the regexes; far faster than checking word by word with `in`
PATTERNS = {cat: re.compile("|".join(re.escape(w) for w in words))
            for cat, words in C.CATEGORIES.items()}
IRR_PAT = re.compile("|".join(re.escape(w) for w in C.IRRELEVANT_WORDS))
CATS = list(C.CATEGORIES.keys())


def label_sentence(s: str):
    s = str(s)
    hits = [c for c in CATS if PATTERNS[c].search(s)]
    return hits, bool(IRR_PAT.search(s))


def main():
    if not C.F_SENTENCES.exists():
        print("Cannot find data/clean/sentences.csv, please run 04_segment.py first")
        sys.exit(1)

    sdf = pd.read_csv(C.F_SENTENCES, encoding="utf-8-sig")
    print(f"Read {len(sdf)} sentences")

    cats_col, irr_col, hit_col = [], [], []
    for s in sdf["sentence"].astype(str):
        hits, irr = label_sentence(s)
        cats_col.append("|".join(hits))
        irr_col.append(int(irr))
        hit_col.append(len(hits))
    sdf["cats_rule"] = cats_col
    sdf["is_irrelevant"] = irr_col
    sdf["n_cats"] = hit_col

    sdf.to_csv(C.F_SENT_LABEL_RULE, index=False, encoding="utf-8-sig")

    # ---------- Comment-level aggregation ----------
    g = sdf.groupby("cid")
    cmt = pd.DataFrame({
        "cats_rule": g["cats_rule"].apply(
            lambda s: "|".join(sorted({c for v in s for c in str(v).split("|") if c}))),
        "n_sent_kept": g.size(),
        "n_irrelevant_sent": g["is_irrelevant"].sum(),
    }).reset_index()

    base = pd.read_csv(C.F_COMMENTS_CLEAN, encoding="utf-8-sig")
    cmt = base.merge(cmt, on="cid", how="left")
    cmt["cats_rule"] = cmt["cats_rule"].fillna("")
    for c in CATS:
        cmt["men_" + c] = cmt["cats_rule"].str.contains(re.escape(c)).astype(int)
    cmt.to_csv(C.F_CMT_LABEL_RULE, index=False, encoding="utf-8-sig")

    # ---------- Statistics ----------
    stats = []
    for c in CATS:
        n_s = int(sdf["cats_rule"].str.contains(re.escape(c)).sum())
        n_c = int(cmt["men_" + c].sum())
        stats.append({"类别": c, "命中句数": n_s, "命中率_句": round(n_s / len(sdf), 4),
                      "命中评论数": n_c, "命中率_评论": round(n_c / len(cmt), 4)})
    st = pd.DataFrame(stats)
    st.to_csv(C.OUT_TABLE / "词典命中统计.csv", index=False, encoding="utf-8-sig")
    print("\nLexicon hit statistics (sorted by comment hit rate, descending):")
    print(st.sort_values("命中率_评论", ascending=False).to_string(index=False))

    # ---------- Per-category samples, for manual lexicon review ----------
    samp_dir = C.OUT_LOG / "词典样例"
    samp_dir.mkdir(exist_ok=True)
    for c in CATS:
        sub = sdf[sdf["cats_rule"].str.contains(re.escape(c))]
        if sub.empty:
            print(f"  !! {c} matched no sentence at all, the lexicon needs more words")
            continue
        take = sub.sample(min(30, len(sub)), random_state=C.RANDOM_STATE)
        (samp_dir / f"{c}.txt").write_text(
            "\n".join(f"- {t}" for t in take["sentence"]), encoding="utf-8")
    print(f"\nPer-category samples written → {samp_dir}")

    # ---------- Unmatched sentences, to see what the lexicon is missing ----------
    nomatch = sdf[(sdf["n_cats"] == 0) & (sdf["n_char"] >= 10)]
    if len(nomatch):
        take = nomatch.sample(min(200, len(nomatch)), random_state=C.RANDOM_STATE)
        (C.OUT_LOG / "词典未命中样例.txt").write_text(
            "\n".join(f"- {t}" for t in take["sentence"]), encoding="utf-8")
        print(f"Unmatched samples ({len(nomatch)} sentences, 200 sampled) → {C.OUT_LOG / '词典未命中样例.txt'}")
        print("   → Open that file and add the recurring new words to CATEGORIES in config.py")

    print("=" * 60)
    print(f"Sentence-level results → {C.F_SENT_LABEL_RULE}")
    print(f"Comment-level results → {C.F_CMT_LABEL_RULE}")


if __name__ == "__main__":
    main()
