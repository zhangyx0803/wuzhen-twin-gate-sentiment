# -*- coding: utf-8 -*-
"""
构建随机森林用的特征矩阵。

评论级特征（对应原研究的"自变量"）：
    men_Cxx : 该评论是否提及第 xx 类属性（0/1）              ← 关注什么
    sen_Cxx : 该评论中第 xx 类句子的平均情感值（-1~1）        ← 怎么看
    n_cat   : 提及的类别数量
    log_len : 评论字数对数（控制变量）
    rating  : 星级评分（若有）
    month / platform / site

因变量：
    is_positive : 是否积极（1/0）—— 主模型
    y_3cls      : 0 消极 / 1 中立 / 2 积极 —— 稳健性
    pos_prob    : 积极概率（连续）—— 稳健性

输入：comments_sentiment.csv、sentences_sentiment.csv
输出：data/feature/features_comment.csv
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
        print("找不到 comments_sentiment.csv，请先运行 08_sentiment.py")
        sys.exit(1)

    cmt = pd.read_csv(C.F_SENTIMENT, encoding="utf-8-sig")
    sdf = pd.read_csv(C.F_SENT_SENTIMENT, encoding="utf-8-sig")
    print(f"评论 {len(cmt)} 条，句子 {len(sdf)} 句")

    # 选标签列：优先用训练好的模型，没有就用词典规则
    label_col = "cats_clf" if "cats_clf" in sdf.columns else "cats_rule"
    print(f"使用句级类别标签列：{label_col}")

    # 若分类模型输出了各类别概率，且用户调整了阈值，则按阈值重新判定
    have_prob = [c for c in CATS if f"prob_{c}" in sdf.columns]
    if have_prob and C.CLF_PROB_THRESHOLD != 0.5:
        P = sdf[[f"prob_{c}" for c in have_prob]].fillna(0).to_numpy()
        mask = P >= C.CLF_PROB_THRESHOLD
        names = np.array(have_prob)
        sdf["labels"] = ["|".join(names[row]) for row in mask]
        label_col = f"thr{C.CLF_PROB_THRESHOLD}"
        print(f"  已按概率阈值 {C.CLF_PROB_THRESHOLD} 重新判定类别"
              f"（覆盖 {len(have_prob)} 个类别）")

    # ---------- 句级特征 ----------
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
    print(f"句级特征 {fs.shape} → {C.F_FEATURES_SENT}")

    # ---------- 评论级特征 ----------
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
    print(f"评论级特征 {feats.shape} → {C.F_FEATURES}")

    # ---------- 概览 ----------
    print("\n各景区样本量与积极情感占比：")
    g = feats.groupby("site").agg(
        评论数=("cid", "count"),
        积极数=("is_positive", "sum"),
    )
    g["积极占比"] = (g["积极数"] / g["评论数"] * 100).round(2)
    print(g.to_string())

    print("\n各类属性提及率（按景区）：")
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

    # ---------- 诊断 ----------
    low, high = [], []
    for c in CATS:
        n = int(feats["men_" + c].sum())
        rate = n / max(len(feats), 1)
        if n < 50:
            low.append((c, n))
        if rate > 0.90:
            high.append((c, rate))
    if low:
        print("\n[提醒] 以下类别被提及的评论少于 50 条，分组比较的结论要谨慎：")
        for c, n in low:
            print(f"    {c}: {n} 条")
    if high:
        print("\n[诊断] 以下类别的提及率超过 90%，通常是分类模型「过度预测」造成的：")
        for c, r in high:
            print(f"    {c}: {r * 100:.1f}%")
        print("    → 处理办法：把 config.py 里的 CLF_PROB_THRESHOLD 调到 0.6 或 0.7，"
              "然后重跑 09 脚本")
        print("    → 或者回到 07 脚本查看这一类别的 precision（多半偏低），补充标注后重训")


if __name__ == "__main__":
    main()
