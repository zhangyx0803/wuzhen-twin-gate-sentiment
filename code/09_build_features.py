# -*- coding: utf-8 -*-
"""
Build the feature matrix used by the random forest.

Review-level features (the "independent variables" of the original study):
    men_Cxx : whether the review mentions aspect category xx (0/1)                ← what people focus on
    sen_Cxx : mean sentiment of category-xx sentences in the review (-1~1)        ← how they feel about it
    n_cat   : number of categories mentioned
    log_len : log of the review character count (control variable)
    rating  : star rating (if available)
    month / platform / site

Dependent variables:
    is_positive : positive or not (1/0) -- main model
    y_3cls      : 0 negative / 1 neutral / 2 positive -- robustness
    pos_prob    : positive probability (continuous) -- robustness

Input: comments_sentiment.csv, sentences_sentiment.csv
Output: data/feature/features_comment.csv
      data/feature/features_sentence.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CATS = list(C.CATEGORIES.keys())


def main():
    if not C.F_SENTIMENT.exists():
        print("comments_sentiment.csv not found; run 08_sentiment.py first")
        sys.exit(1)

    cmt = pd.read_csv(C.F_SENTIMENT, encoding="utf-8-sig")
    sdf = pd.read_csv(C.F_SENT_SENTIMENT, encoding="utf-8-sig")
    print(f"{len(cmt)} reviews, {len(sdf)} sentences")

    # Choose the label column: prefer the trained model, fall back to the lexicon rules
    label_col = "cats_clf" if "cats_clf" in sdf.columns else "cats_rule"
    print(f"Using sentence-level category label column: {label_col}")

    # If the classifier emitted per-category probabilities and the user changed the threshold, re-decide by threshold
    have_prob = [c for c in CATS if f"prob_{c}" in sdf.columns]
    if have_prob and C.CLF_PROB_THRESHOLD != 0.5:
        P = sdf[[f"prob_{c}" for c in have_prob]].fillna(0).to_numpy()
        mask = P >= C.CLF_PROB_THRESHOLD
        names = np.array(have_prob)
        sdf["labels"] = ["|".join(names[row]) for row in mask]
        label_col = f"thr{C.CLF_PROB_THRESHOLD}"
        print(f"  Re-decided categories using probability threshold {C.CLF_PROB_THRESHOLD}"
              f" (covering {len(have_prob)} categories)")

    # ---------- Sentence-level features ----------
    sdf["labels"] = sdf[label_col].fillna("").astype(str)
    srows = []
    for i, r in enumerate(sdf.itertuples(index=False)):
        cats = [c for c in str(r.labels).split("|") if c]
        row = {"sent_id": r.sent_id, "cid": r.cid, "site": r.site_sent,
               "sentence": r.sentence, "date": r.date, "ym": r.ym,
               "sent_score": r.sent_score, "sent_polar": r.sent_polar,
               "n_cats": len(cats)}
        for c in CATS:
            row["is_" + c] = int(c in cats)
        srows.append(row)
    fs = pd.DataFrame(srows)
    fs.to_csv(C.F_FEATURES_SENT, index=False, encoding="utf-8-sig")
    print(f"Sentence-level features {fs.shape} → {C.F_FEATURES_SENT}")

    # ---------- Review-level features ----------
    agg = {}
    tmp = pd.DataFrame({"cid": sdf["cid"], "score": sdf["sent_score"]})
    for c in CATS:
        hit = sdf["labels"].str.contains(re.escape(c), regex=True).fillna(False)
        tmp["_hit"] = hit.astype(int)
        tmp["_sc"] = tmp["score"].where(hit)
        g = tmp.groupby("cid")
        men = g["_hit"].max().rename("men_" + c)
        sen = g["_sc"].mean().rename("sen_" + c)
        agg[c] = (men, sen)

    feats = cmt[["cid", "site", "platform", "date", "ym", "year", "month",
                 "rating", "n_char", "sentiment", "sentiment_label",
                 "is_positive", "positive_prob", "n_sent"]].copy()
    for c, (men, sen) in agg.items():
        feats = feats.merge(men.reset_index(), on="cid", how="left")
        feats = feats.merge(sen.reset_index(), on="cid", how="left")
    feats["log_len"] = np.log1p(feats["n_char"])
    feats["n_cat"] = feats[["men_" + c for c in CATS]].sum(axis=1)
    feats["y_3cls"] = feats["sentiment"]

    feats.to_csv(C.F_FEATURES, index=False, encoding="utf-8-sig")
    print(f"Review-level features {feats.shape} → {C.F_FEATURES}")

    # ---------- Overview ----------
    print("\nSample size and positive-sentiment share by scenic area:")
    g = feats.groupby("site").agg(
        评论数=("cid", "count"),
        积极数=("is_positive", "sum"),
    )
    g["积极占比"] = (g["积极数"] / g["评论数"] * 100).round(2)
    print(g.to_string())

    print("\nMention rate of each aspect category (by scenic area):")
    rows = []
    for c in CATS:
        r = {"类别": c}
        for s in C.SITES:
            sub = feats[feats["site"] == s]
            r[s + "_提及率"] = round(float(sub["men_" + c].mean() * 100), 2) if len(sub) else np.nan
            r[s + "_属性情感"] = round(float(sub["sen_" + c].mean()), 3) if sub["sen_" + c].notna().any() else np.nan
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(C.OUT_TABLE / "属性提及率与情感_概览.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))

    # ---------- Diagnostics ----------
    low, high = [], []
    for c in CATS:
        n = int(feats["men_" + c].sum())
        rate = n / max(len(feats), 1)
        if n < 50:
            low.append((c, n))
        if rate > 0.90:
            high.append((c, rate))
    if low:
        print("\n[Warning] The following categories are mentioned in fewer than 50 reviews; interpret group comparisons with caution:")
        for c, n in low:
            print(f"    {c}: {n} reviews")
    if high:
        print("\n[Diagnostic] The following categories have a mention rate above 90%, usually caused by the classifier 'over-predicting':")
        for c, r in high:
            print(f"    {c}: {r * 100:.1f}%")
        print("    → Fix: raise CLF_PROB_THRESHOLD in config.py to 0.6 or 0.7, "
              "then re-run script 09")
        print("    → Or go back to script 07 to check this category's precision (usually low), add labels and retrain")


if __name__ == "__main__":
    main()
