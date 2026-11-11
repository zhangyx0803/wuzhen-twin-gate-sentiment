# -*- coding: utf-8 -*-
"""
Train the AI corpus classification model (multi-label text classification).

Model: TF-IDF (character-level 1-2 grams) + logistic regression (OvR multi-label)
Why this choice: Chinese needs no tokenization, a few thousand labeled sentences reach decent accuracy,
           it is interpretable (per-category keywords can be exported) and trains in seconds on a CPU.
           The original study used the 2000 most informative feature words; TF-IDF does that automatically.

Input: data/label/label_done.xlsx
Output: data/feature/clf_model.joblib
        outputs/tables/分类模型评估.txt
        outputs/tables/分类模型_各类指标.csv
        outputs/tables/分类模型_关键词.csv
        outputs/tables/分类模型_错误样例.csv
        data/clean/sentences_clf_label.csv (predictions for all sentences)
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
    print("Missing scikit-learn: python -m pip install scikit-learn")
    sys.exit(1)

try:
    import joblib
    dump, load = joblib.dump, joblib.load
except ImportError:
    import pickle
    dump = lambda o, p: pickle.dump(o, open(p, "wb"))
    load = lambda p: pickle.load(open(p, "rb"))
    print("[note] joblib is not installed, falling back to pickle (same functionality)")


# ============================================================
# Parse the manual annotations
# ============================================================
SPLIT_RE = re.compile(r"[|｜、,，;；/\s]+")


def parse_labels(s):
    """Parse 'C01|C04' / '夜景灯光,氛围' / '无' into a list of standard category names"""
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
        if part in CATS:                                   # full name
            out.append(part)
            continue
        code = part.upper()
        hit = [c for c in CATS if c.startswith(code)]       # C01 / C1
        if hit:
            out.append(hit[0])
            continue
        hit = [c for c in CATS if part in c]                # partial containment
        if hit:
            out.append(hit[0])
            continue
        near = difflib.get_close_matches(part, CATS, n=1, cutoff=0.6)
        if near:
            out.append(near[0])
        else:
            print(f"  [unrecognized] '{part}' (raw content: {txt[:40]})")
    return sorted(set(out))


def main():
    src = C.F_LABEL_DONE if C.F_LABEL_DONE.exists() else C.F_LABEL_SHEET
    if not src.exists():
        print("Cannot find the annotation file. Please run 06_make_label_sheet.py first and complete the manual annotation.")
        sys.exit(1)
    if src == C.F_LABEL_SHEET:
        print(f"[note] {C.F_LABEL_DONE.name} not found, doing a trial run with the unfinished {C.F_LABEL_SHEET.name}.")
        print("       Formal results in the paper must use the manually annotated label_done.xlsx.")

    df = pd.read_excel(src, sheet_name="标注")
    df = df[df["句子"].notna()].copy()
    df["labels"] = df["你的标注"].map(parse_labels)
    labeled = df[df["你的标注"].notna() & (df["你的标注"].astype(str).str.strip() != "")]
    print(f"Read {len(df)} sentences, {len(labeled)} of them annotated")

    if len(labeled) < 200:
        print("!! Too few annotated samples (<200), the model would be unreliable. Please label at least 1000 sentences.")
        print("   If you are really short of time: use the lexicon classification from 05 as the main result,")
        print("   state in the paper that you used domain-lexicon rule-based classification, and validate the accuracy on a manual sample.")
        sys.exit(1)

    texts = labeled["句子"].astype(str).tolist()
    Y = labeled["labels"].tolist()

    # Drop categories that never appear in the annotations (they cannot be trained)
    from collections import Counter
    cnt = Counter(c for L in Y for c in L)
    active = [c for c in CATS if cnt.get(c, 0) >= 5]
    dropped = [c for c in CATS if c not in active]
    if dropped:
        print(f"[note] these categories have <5 annotated samples and are not trained this time: {dropped}")
    Y = [[c for c in L if c in active] for L in Y]

    mlb = MultiLabelBinarizer(classes=active)
    Ybin = mlb.fit_transform(Y)
    print(f"Label matrix: {Ybin.shape[0]} samples × {Ybin.shape[1]} categories")
    print("Annotated count per category:")
    print(pd.Series(Ybin.sum(axis=0), index=active).sort_values(ascending=False).to_string())

    # ---------- Features ----------
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2),
                          min_df=2, max_features=60000,
                          sublinear_tf=True, norm="l2")
    X = vec.fit_transform(texts)
    print(f"Feature matrix: {X.shape}")

    # ---------- Split (keeping the original indices so error samples can be traced back) ----------
    all_idx = np.arange(len(texts))
    itr, ite = train_test_split(all_idx, test_size=C.TEST_SIZE,
                                random_state=C.RANDOM_STATE)
    Xtr, Xte, Ytr, Yte = X[itr], X[ite], Ybin[itr], Ybin[ite]
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced",
                             solver="liblinear" if len(active) <= 2 else "lbfgs")
    clf.fit(Xtr, Ytr)
    Ypred = clf.predict(Xte)

    # ---------- Evaluation ----------
    lines = []
    lines.append("=" * 70)
    lines.append("AI corpus classification model · evaluation report")
    lines.append("=" * 70)
    lines.append(f"Labeled samples      : {len(texts)}")
    lines.append(f"Categories           : {len(active)}")
    lines.append(f"Feature dim (TF-IDF) : {X.shape[1]}")
    lines.append(f"Train/validation     : {int((1 - C.TEST_SIZE) * 100)}% / {int(C.TEST_SIZE * 100)}%")
    lines.append("")
    lines.append(f"Exact-match accuracy (subset): {accuracy_score(Yte, Ypred):.4f}")
    lines.append(f"F1-micro          : {f1_score(Yte, Ypred, average='micro', zero_division=0):.4f}")
    lines.append(f"F1-macro          : {f1_score(Yte, Ypred, average='macro', zero_division=0):.4f}")
    lines.append(f"Hamming loss      : {hamming_loss(Yte, Ypred):.4f}")
    lines.append("")
    try:
        cv = cross_val_score(clf, X, Ybin, cv=C.CV_FOLDS, scoring="f1_micro")
        lines.append(f"{C.CV_FOLDS}-fold cross-validation F1-micro: {cv.mean():.4f} (+/- {cv.std():.4f})")
        lines.append(f"   per fold: {[round(v, 4) for v in cv]}")
    except Exception as e:
        lines.append(f"(cross-validation skipped: {e})")
    lines.append("")
    lines.append("-" * 70)
    lines.append("Per-category metrics (precision / recall / f1 / support)")
    lines.append("-" * 70)
    rep = classification_report(Yte, Ypred, target_names=active,
                                zero_division=0, digits=4)
    lines.append(rep)
    report_txt = "\n".join(lines)
    (C.OUT_TABLE / "分类模型评估.txt").write_text(report_txt, encoding="utf-8")
    print("\n" + report_txt)

    # Per-category metrics table
    repd = classification_report(Yte, Ypred, target_names=active,
                                 zero_division=0, output_dict=True)
    pd.DataFrame(repd).T.to_csv(C.OUT_TABLE / "分类模型_各类指标.csv", encoding="utf-8-sig")

    # ---------- Per-category keywords (interpretability, very useful in a paper) ----------
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

    # ---------- Error samples (for iterative improvement) ----------
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
        print(f"Error samples: {len(errs)} → outputs/tables/分类模型_错误样例.csv")
        print("   → After reviewing these errors, go back to 06 and annotate more (focus on the error types), then retrain")

    # ---------- Save the model ----------
    dump({"vec": vec, "clf": clf, "mlb": mlb, "active": active}, C.F_CLF_MODEL)

    # ---------- Full prediction ----------
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
        print(f"Predicted all {len(sdf)} sentences → {C.F_SENT_CLF}")

    print("=" * 70)
    print(f"Model saved → {C.F_CLF_MODEL}")
    print(f"Accuracy target: F1-micro >= 0.85 (the original study reported 86% accuracy)")


if __name__ == "__main__":
    main()
