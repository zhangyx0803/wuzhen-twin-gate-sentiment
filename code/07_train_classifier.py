# -*- coding: utf-8 -*-
"""
训练 AI 语料分类模型（多标签文本分类）。

模型：TF-IDF（字级 1-2 gram）+ 逻辑回归（OvR 多标签）
为什么选它：中文不需要分词、几千条标注就能到较高精度、可解释（能导出每类的关键词）、
           CPU 上几十秒训完。原研究用的是"信息量最高的 2000 个特征词"，
           TF-IDF 自动完成这件事。

输入：data/label/label_done.xlsx
输出：data/feature/clf_model.joblib
      outputs/tables/分类模型评估.txt
      outputs/tables/分类模型_各类指标.csv
      outputs/tables/分类模型_关键词.csv
      outputs/tables/分类模型_错误样例.csv
      data/clean/sentences_clf_label.csv（对全量句子的预测结果）
"""
import difflib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import norm_text

CATS = list(C.CATEGORIES.keys())

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (classification_report, f1_score,
                                 accuracy_score, hamming_loss)
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:
    print("缺少 scikit-learn：python -m pip install scikit-learn")
    sys.exit(1)

try:
    import joblib
    dump, load = joblib.dump, joblib.load
except ImportError:
    import pickle
    dump = lambda o, p: pickle.dump(o, open(p, "wb"))
    load = lambda p: pickle.load(open(p, "rb"))
    print("[提示] 未安装 joblib，改用 pickle（功能相同）")


# ============================================================
# 解析人工标注
# ============================================================
SPLIT_RE = re.compile(r"[|｜、,，;；/\s]+")


def parse_labels(s):
    """把 'C01|C04' / '夜景灯光,氛围' / '无' 解析成标准类别名列表"""
    if pd.isna(s):
        return []
    txt = str(s).strip()
    if txt in ("", "无", "none", "None", "0", "N/A", "-"):
        return []
    out = []
    for part in SPLIT_RE.split(txt):
        part = part.strip()
        if not part:
            continue
        if part in CATS:                                   # 全名
            out.append(part)
            continue
        code = part.upper()
        hit = [c for c in CATS if c.startswith(code)]       # C01 / C1
        if hit:
            out.append(hit[0])
            continue
        hit = [c for c in CATS if part in c]                # 部分包含
        if hit:
            out.append(hit[0])
            continue
        near = difflib.get_close_matches(part, CATS, n=1, cutoff=0.6)
        if near:
            out.append(near[0])
        else:
            print(f"  [无法识别] '{part}'（原始内容：{txt[:40]}）")
    return sorted(set(out))


def main():
    src = C.F_LABEL_DONE if C.F_LABEL_DONE.exists() else C.F_LABEL_SHEET
    if not src.exists():
        print("找不到标注文件。请先运行 06_make_label_sheet.py 并完成人工标注。")
        sys.exit(1)
    if src == C.F_LABEL_SHEET:
        print(f"[注意] 没找到 {C.F_LABEL_DONE.name}，用未完成的 {C.F_LABEL_SHEET.name} 试跑。")
        print("       论文里的正式结果必须用人工标注后的 label_done.xlsx。")

    df = pd.read_excel(src, sheet_name="标注")
    df = df[df["句子"].notna()].copy()
    df["labels"] = df["你的标注"].map(parse_labels)
    labeled = df[df["你的标注"].notna() & (df["你的标注"].astype(str).str.strip() != "")]
    print(f"读入 {len(df)} 句，其中已标注 {len(labeled)} 句")

    if len(labeled) < 200:
        print("!! 已标注样本太少（<200），模型会不可靠。请至少标 1000 句。")
        print("   如果时间实在不够：可以直接用 05 的词典分类结果作为主结果，")
        print("   在论文里说明『采用领域词典规则分类』，并做人工抽样验证准确率。")
        sys.exit(1)

    texts = labeled["句子"].astype(str).tolist()
    Y = labeled["labels"].tolist()

    # 去掉标注里从没出现过的类别（没法训练）
    from collections import Counter
    cnt = Counter(c for L in Y for c in L)
    active = [c for c in CATS if cnt.get(c, 0) >= 5]
    dropped = [c for c in CATS if c not in active]
    if dropped:
        print(f"[提示] 以下类别标注样本 <5，本次不训练：{dropped}")
    Y = [[c for c in L if c in active] for L in Y]

    mlb = MultiLabelBinarizer(classes=active)
    Ybin = mlb.fit_transform(Y)
    print(f"标签矩阵：{Ybin.shape[0]} 样本 × {Ybin.shape[1]} 类别")
    print("各类标注数量：")
    print(pd.Series(Ybin.sum(axis=0), index=active).sort_values(ascending=False).to_string())

    # ---------- 特征 ----------
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                          min_df=2, max_features=60000,
                          sublinear_tf=True, norm="l2")
    X = vec.fit_transform(texts)
    print(f"特征矩阵：{X.shape}")

    # ---------- 划分（同时保留原始下标，便于回查错误样例的原句）----------
    all_idx = np.arange(len(texts))
    itr, ite = train_test_split(all_idx, test_size=C.TEST_SIZE,
                                random_state=C.RANDOM_STATE)
    Xtr, Xte, Ytr, Yte = X[itr], X[ite], Ybin[itr], Ybin[ite]
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                             solver="liblinear" if len(active) <= 2 else "lbfgs")
    clf.fit(Xtr, Ytr)
    Ypred = clf.predict(Xte)

    # ---------- 评估 ----------
    lines = []
    lines.append("=" * 70)
    lines.append("AI 语料分类模型 · 评估报告")
    lines.append("=" * 70)
    lines.append(f"标注样本数        : {len(texts)}")
    lines.append(f"类别数            : {len(active)}")
    lines.append(f"特征维度(TF-IDF)  : {X.shape[1]}")
    lines.append(f"训练/验证划分     : {int((1 - C.TEST_SIZE) * 100)}% / {int(C.TEST_SIZE * 100)}%")
    lines.append("")
    lines.append(f"完全匹配准确率(子集): {accuracy_score(Yte, Ypred):.4f}")
    lines.append(f"F1-micro          : {f1_score(Yte, Ypred, average='micro', zero_division=0):.4f}")
    lines.append(f"F1-macro          : {f1_score(Yte, Ypred, average='macro', zero_division=0):.4f}")
    lines.append(f"Hamming loss      : {hamming_loss(Yte, Ypred):.4f}")
    lines.append("")
    try:
        cv = cross_val_score(clf, X, Ybin, cv=C.CV_FOLDS, scoring="f1_micro")
        lines.append(f"{C.CV_FOLDS} 折交叉验证 F1-micro: {cv.mean():.4f} (+/- {cv.std():.4f})")
        lines.append(f"   各折: {[round(v, 4) for v in cv]}")
    except Exception as e:
        lines.append(f"(交叉验证跳过: {e})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("各类别指标（precision / recall / f1 / support）")
    lines.append("-" * 70)
    rep = classification_report(Yte, Ypred, target_names=active,
                                zero_division=0, digits=4)
    lines.append(rep)
    report_txt = "\n".join(lines)
    (C.OUT_TABLE / "分类模型评估.txt").write_text(report_txt, encoding="utf-8")
    print("\n" + report_txt)

    # 各类指标表
    repd = classification_report(Yte, Ypred, target_names=active,
                                 zero_division=0, output_dict=True)
    pd.DataFrame(repd).T.to_csv(C.OUT_TABLE / "分类模型_各类指标.csv", encoding="utf-8-sig")

    # ---------- 每类关键词（可解释性，论文里很好用） ----------
    feats = np.array(vec.get_feature_names_out())
    rows = []
    for i, c in enumerate(active):
        w = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
        top = np.argsort(w)[-20:][::-1]
        rows.append({"类别": c,
                     "高权重特征词": "、".join(feats[top[:15]]),
                     "低权重特征词": "、".join(feats[np.argsort(w)[:8]])})
    pd.DataFrame(rows).to_csv(C.OUT_TABLE / "分类模型_关键词.csv",
                              index=False, encoding="utf-8-sig")

    # ---------- 错误样例（迭代改进用） ----------
    errs = []
    for pos, orig in enumerate(ite):
        true_set = set(np.array(active)[Yte[pos] == 1])
        pred_set = set(np.array(active)[Ypred[pos] == 1])
        if true_set != pred_set:
            errs.append({"句子": texts[orig],
                         "真实": "|".join(sorted(true_set)),
                         "预测": "|".join(sorted(pred_set))})
    if errs:
        pd.DataFrame(errs).to_csv(C.OUT_TABLE / "分类模型_错误样例.csv",
                                  index=False, encoding="utf-8-sig")
        print(f"错误样例 {len(errs)} 条 → outputs/tables/分类模型_错误样例.csv")
        print("   → 看完这些错误，回到 06 补充标注（重点标错误的类型），再重训一轮")

    # ---------- 保存模型 ----------
    dump({"vec": vec, "clf": clf, "mlb": mlb, "active": active}, C.F_CLF_MODEL)

    # ---------- 全量预测 ----------
    if C.F_SENT_LABEL_RULE.exists():
        sdf = pd.read_csv(C.F_SENT_LABEL_RULE, encoding="utf-8-sig")
        Xall = vec.transform(sdf["sentence"].astype(str))
        P = clf.predict(Xall)
        try:
            Prob = clf.predict_proba(Xall)
        except Exception:
            Prob = None
        sdf["cats_clf"] = ["|".join(np.array(active)[row == 1]) for row in P]
        sdf["n_cats_clf"] = P.sum(axis=1)
        if Prob is not None:
            for j, c in enumerate(active):
                sdf["prob_" + c] = np.round(Prob[:, j], 4)
        sdf.to_csv(C.F_SENT_CLF, index=False, encoding="utf-8-sig")
        print(f"全量 {len(sdf)} 句预测完成 → {C.F_SENT_CLF}")

    print("=" * 70)
    print(f"模型已保存 → {C.F_CLF_MODEL}")
    print(f"精度达标线：F1-micro >= 0.85（原研究报告 86% 精度）")


if __name__ == "__main__":
    main()
