# -*- coding: utf-8 -*-
"""
Random forest + SHAP: quantify how much each aspect contributes to tourists' positive sentiment, and compare the East and West Scenic Areas.

Model setup:
    Dependent variable y    : is_positive (1=positive, 0=not positive) -- main model
    Independent variables X : "mention" (men_) of 21 categories + "aspect sentiment" (sen_) of 21 categories
                              + control variables log_len / rating / month
    Split                   : 80% train / 20% validation (same as the original study)
    Evaluation              : 5-fold cross-validated AUC / F1
    Interpretation          : permutation importance + SHAP values + partial dependence plots

Robustness checks (run automatically):
    R1 dependent variable switched to the 3-class y_3cls
    R2 model switched to logistic regression
    R3 both groups downsampled to the same sample size
    R4 long reviews only (>= 30 characters)

Input: data/feature/features_comment.csv
Output: outputs/tables/随机森林_*.csv / .txt
      outputs/figures/图4_*.png, 图5_*.png, 图6_*.png
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import set_chinese_font

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             classification_report)
from sklearn.inspection import permutation_importance
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAS_SK = True
except Exception:
    HAS_SK = False

CATS = list(C.CATEGORIES.keys())
E, W = C.SITE_EAST, C.SITE_WEST


def build_xy(df, ycol="is_positive"):
    df = df[df[ycol].notna()].copy()
    men = ["men_" + c for c in CATS if "men_" + c in df.columns]
    sen = ["sen_" + c for c in CATS if "sen_" + c in df.columns]
    ctrl = [c for c in ["log_len", "rating", "month"] if c in df.columns]
    X = df[men + sen + ctrl].copy()
    # A missing sen_ means "the review did not mention this aspect"; fill 0 (0 = neutral / not mentioned)
    X[sen] = X[sen].fillna(0.0)
    if "rating" in X.columns:
        X["rating"] = X["rating"].fillna(X["rating"].median())
    if "month" in X.columns:
        X["month"] = X["month"].fillna(0)
    keep = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
    dropped = [c for c in X.columns if c not in keep]
    if dropped:
        print(f"    (dropped {len(dropped)} constant features)")
    return X[keep], df[ycol].astype(int), keep


def fit_one(df, tag, ycol="is_positive", model="rf"):
    X, y, feats = build_xy(df, ycol)
    if len(y) < 100 or y.nunique() < 2:
        print(f"  [{tag}] insufficient samples or classes, skipping")
        return None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                          random_state=C.RANDOM_STATE, stratify=y)
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=C.RF_N_ESTIMATORS, min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
            class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1)
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]

    res = {
        "组": tag, "模型": model, "样本量": len(y), "特征数": X.shape[1],
        "训练集": len(ytr), "验证集": len(yte),
        "验证集准确率": round(accuracy_score(yte, pred), 4),
        "验证集F1": round(f1_score(yte, pred, zero_division=0), 4),
        "验证集AUC": round(roc_auc_score(yte, prob), 4),
        "积极率%": round(y.mean() * 100, 2),
    }
    try:
        cv = cross_val_score(clf, X, y, cv=StratifiedKFold(C.CV_FOLDS, shuffle=True,
                                                           random_state=C.RANDOM_STATE),
                             scoring="roc_auc")
        res["5折CV_AUC均值"] = round(cv.mean(), 4)
        res["5折CV_AUC标准差"] = round(cv.std(), 4)
    except Exception:
        res["5折CV_AUC均值"] = np.nan
        res["5折CV_AUC标准差"] = np.nan

    print(f"  [{tag}/{model}] n={len(y)} AUC={res['验证集AUC']} "
          f"F1={res['验证集F1']} CV-AUC={res.get('5折CV_AUC均值')}")

    # ---- Permutation importance (more reliable than built-in importance) ----
    pi = permutation_importance(clf, Xte, yte, n_repeats=20,
                                random_state=C.RANDOM_STATE, n_jobs=-1,
                                scoring="roc_auc")
    imp = pd.DataFrame({
        "特征": X.columns,
        "置换重要性": pi.importances_mean.round(5),
        "标准差": pi.importances_std.round(5),
    }).sort_values("置换重要性", ascending=False).reset_index(drop=True)
    imp["排名"] = imp.index + 1
    imp["组"] = tag

    if model == "rf":
        imp["内置重要性"] = clf.feature_importances_.round(5)

    return {"res": res, "imp": imp, "clf": clf, "X": X, "y": y,
            "Xte": Xte, "yte": yte, "prob": prob}


def main():
    if not C.F_FEATURES.exists():
        print("features_comment.csv not found; run 09_build_features.py first")
        sys.exit(1)

    df = pd.read_csv(C.F_FEATURES, encoding="utf-8-sig")
    df = df[df["site"].isin([E, W])].copy()
    print(f"Main analysis sample {len(df)} reviews ({E} {int((df['site'] == E).sum())} | "
          f"{W} {int((df['site'] == W).sum())})\n")

    results, imps = [], []

    print("[Main model] Fit models by group")
    for s in (E, W):
        r = fit_one(df[df["site"] == s], s)
        if r:
            results.append(r["res"])
            imps.append(r["imp"])

    print("\n[Pooled model] Add the scenic-area variable as a control")
    d2 = df.copy()
    d2["is_west"] = (d2["site"] == W).astype(int)
    r = fit_one(d2, "全样本")
    if r:
        results.append(r["res"])
        imps.append(r["imp"])

    summary = pd.DataFrame(results)
    print("\n" + summary.to_string(index=False))
    summary.to_csv(C.OUT_TABLE / "随机森林_模型表现.csv", index=False, encoding="utf-8-sig")

    imp_all = pd.concat(imps, ignore_index=True)
    imp_all.to_csv(C.OUT_TABLE / "随机森林_特征重要性.csv", index=False, encoding="utf-8-sig")

    # ---------- Comparison of importance rankings between the two scenic areas ----------
    pe = imp_all[imp_all["组"] == E].set_index("特征")["置换重要性"]
    pw = imp_all[imp_all["组"] == W].set_index("特征")["置换重要性"]
    cmp_df = pd.DataFrame({"东栅重要性": pe, "西栅重要性": pw}).fillna(0.0)
    cmp_df["东栅排名"] = cmp_df["东栅重要性"].rank(ascending=False).astype(int)
    cmp_df["西栅排名"] = cmp_df["西栅重要性"].rank(ascending=False).astype(int)
    cmp_df["排名差(东-西)"] = cmp_df["东栅排名"] - cmp_df["西栅排名"]
    cmp_df = cmp_df.reset_index().rename(columns={"index": "特征"})
    cmp_df = cmp_df.sort_values("西栅重要性", ascending=False)
    cmp_df.to_csv(C.OUT_TABLE / "随机森林_两栅重要性对比.csv",
                  index=False, encoding="utf-8-sig")

    print("\nTop 12 features in the two scenic areas:")
    print(cmp_df.head(12)[["特征", "东栅重要性", "西栅重要性",
                           "东栅排名", "西栅排名", "排名差(东-西)"]].to_string(index=False))

    # Rank consistency
    from scipy import stats as st
    rho, pval = st.spearmanr(cmp_df["东栅重要性"], cmp_df["西栅重要性"])
    top_e = set(cmp_df.nsmallest(10, "东栅排名")["特征"])
    top_w = set(cmp_df.nsmallest(10, "西栅排名")["特征"])
    overlap = sorted(top_e & top_w)
    txt = (f"东栅 / 西栅 驱动结构对比\n"
           f"{'=' * 56}\n"
           f"特征重要性排名的 Spearman 相关: rho = {rho:.4f}, p = {pval:.4g}\n"
           f"  → rho 高（>0.7）说明两栅的驱动结构相似；低（<0.4）说明存在结构性差异，可支撑 H3\n\n"
           f"两栅共同的前 10 关键因素（{len(overlap)} 个）: {overlap}\n\n"
           f"仅在东栅前 10 的因素: {sorted(top_e - top_w)}\n"
           f"仅在西栅前 10 的因素: {sorted(top_w - top_e)}\n")
    (C.OUT_TABLE / "随机森林_驱动结构对比.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)

    # ---------- Robustness checks ----------
    print("[Robustness checks]")
    rob = []
    for s in (E, W):
        sub = df[df["site"] == s]
        # R1 three-class target
        try:
            X, y3, _ = build_xy(sub, "y_3cls")
            Xtr, Xte, ytr, yte = train_test_split(
                X, y3, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y3)
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=C.RF_MIN_SAMPLES_LEAF,
                                       class_weight="balanced",
                                       random_state=C.RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
            rob.append({"组": s, "检验": "R1 因变量=三分类", "指标": "F1-macro",
                        "取值": round(f1_score(yte, m.predict(Xte), average="macro",
                                           zero_division=0), 4)})
        except Exception as ex:
            rob.append({"组": s, "检验": "R1 因变量=三分类", "指标": "失败", "取值": str(ex)[:40]})
        # R2 logistic regression
        r2 = fit_one(sub, s, model="lr")
        if r2:
            rob.append({"组": s, "检验": "R2 模型=逻辑回归", "指标": "验证集AUC",
                        "取值": r2["res"]["验证集AUC"]})
            pe2 = r2["imp"].set_index("特征")["置换重要性"]
            pe1 = cmp_df.set_index("特征")[f"{s}重要性"]
            common = pe1.index.intersection(pe2.index)
            rho2, _ = st.spearmanr(pe1[common], pe2[common])
            rob.append({"组": s, "检验": "R2 模型=逻辑回归", "指标": "与RF的排名相关性",
                        "取值": round(float(rho2), 4)})
        # R4 long reviews
        r4 = fit_one(sub[sub["n_char"] >= 30], s + "-长评论")
        if r4:
            rob.append({"组": s, "检验": "R4 仅长评论(>=30字)", "指标": "验证集AUC",
                        "取值": r4["res"]["验证集AUC"]})
    # R3 equal sample size
    ne = min(len(df[df["site"] == E]), len(df[df["site"] == W]))
    for s in (E, W):
        sub = df[df["site"] == s].sample(n=ne, random_state=C.RANDOM_STATE)
        r3 = fit_one(sub, s + "-等样本")
        if r3:
            rob.append({"组": s, "检验": f"R3 下采样至 n={ne}", "指标": "验证集AUC",
                        "取值": r3["res"]["验证集AUC"]})
    rob_df = pd.DataFrame(rob)
    rob_df.to_csv(C.OUT_TABLE / "随机森林_稳健性检验.csv", index=False, encoding="utf-8-sig")
    print(rob_df.to_string(index=False))

    # ---------- SHAP ----------
    print("\n[SHAP interpretation]")
    try:
        import shap
        plt = set_chinese_font()
        make_plots = plt is not None
    except ImportError:
        print("  shap is not installed; skipping SHAP analysis. Install with: pip install shap")
        shap = None
        make_plots = False

    if shap is not None:
        shap_rows = []
        for s in (E, W):
            sub = df[df["site"] == s]
            r = fit_one(sub, s)
            if not r:
                continue
            clf, X, y = r["clf"], r["X"], r["y"]
            Xs = X.sample(min(800, len(X)), random_state=C.RANDOM_STATE)
            ex = shap.TreeExplainer(clf)
            sv = ex.shap_values(Xs)
            if isinstance(sv, list):
                sv_pos = sv[1] if len(sv) > 1 else sv[0]
            elif getattr(sv, "ndim", 2) == 3:
                sv_pos = sv[:, :, 1]
            else:
                sv_pos = sv
            mean_abs = np.abs(sv_pos).mean(axis=0)
            dfx = pd.DataFrame({"特征": Xs.columns, "SHAP平均绝对值": mean_abs.round(5),
                                "组": s}).sort_values("SHAP平均绝对值", ascending=False)
            dfx["SHAP排名"] = range(1, len(dfx) + 1)
            shap_rows.append(dfx)
            if make_plots:
                plt.figure(figsize=(9, 8))
                shap.summary_plot(sv_pos, Xs, feature_names=list(Xs.columns),
                                  show=False, max_display=18)
                fig = plt.gcf()
                fig.suptitle(f"{s} 游客积极情感的 SHAP 特征贡献", fontsize=14)
                plt.tight_layout()
                fig.savefig(C.OUT_FIG / f"图5_SHAP摘要_{s}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  SHAP summary plot → 图5_SHAP摘要_{s}.png")
        if shap_rows:
            sh = pd.concat(shap_rows, ignore_index=True)
            sh.to_csv(C.OUT_TABLE / "随机森林_SHAP重要性.csv", index=False, encoding="utf-8-sig")
            print("\nTop 8 mean absolute SHAP contributions:")
            for s in (E, W):
                top = sh[sh["组"] == s].head(8)
                print(f"  {s}: " + ", ".join(f"{r[0]}({r[1]:.3f})"
                                             for r in top.itertuples(index=False)))
            print("\nTip: if the SHAP importance ranking broadly agrees with permutation importance, the conclusions are robust;")
            print("      if they differ substantially, report the features highlighted by both as the core findings in the paper.")

    # ---------- Partial dependence plots ----------
    if shap is not None and make_plots:
        from sklearn.inspection import PartialDependenceDisplay
        for s in (E, W):
            sub = df[df["site"] == s]
            r = fit_one(sub, s)
            if not r:
                continue
            top = r["imp"].head(6)["特征"].tolist()
            top = [f for f in top if r["X"][f].nunique() > 1][:4]
            if not top:
                continue
            fig, ax = plt.subplots(1, len(top), figsize=(4.2 * len(top), 3.6))
            if len(top) == 1:
                ax = [ax]
            PartialDependenceDisplay.from_estimator(
                r["clf"], r["X"], top, ax=ax, kind="average",
                grid_resolution=30, random_state=C.RANDOM_STATE)
            fig.suptitle(f"{s}：关键特征的偏依赖图", fontsize=13)
            plt.tight_layout()
            fig.savefig(C.OUT_FIG / f"图6_偏依赖图_{s}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  Partial dependence plot → 图6_偏依赖图_{s}.png")

    print("=" * 60)
    print("Done. When reporting results in the paper, remember to state clearly:")
    print("  · how feature importance was measured (permutation importance + SHAP cross-check)")
    print("  · the corresponding AUC / F1 (model reliability)")
    print("  · whether the 4 robustness settings support the main conclusions")


if __name__ == "__main__":
    main()
