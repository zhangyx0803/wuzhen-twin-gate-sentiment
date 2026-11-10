# -*- coding: utf-8 -*-
"""
东栅 vs 西栅 的统计检验。

包含：
  1. 描述统计（样本量、评分、情感、评论长度）
  2. 积极情感比例差异：卡方检验 + Cramér's V 效应量
  3. 评分 / 情感分值差异：Mann-Whitney U 检验 + Cliff's delta 效应量
  4. 每个属性类别的提及率差异：卡方检验 + BH-FDR 多重比较校正
  5. 每个属性类别的属性情感差异：Mann-Whitney U + FDR 校正

输入：data/feature/features_comment.csv
输出：outputs/tables/对比检验.xlsx（多工作表）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

CATS = list(C.CATEGORIES.keys())
E, W = C.SITE_EAST, C.SITE_WEST


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正，返回校正后的 q 值"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        i = n - rank
        val = min(prev, p[idx] * n / i)
        q[idx] = val
        prev = val
    return q


def cramers_v(tab):
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.values.sum()
    r, k = tab.shape
    if n == 0 or min(r - 1, k - 1) == 0:
        return np.nan
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # 用秩和近似，避免 O(n*m)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    if not C.F_FEATURES.exists():
        print("找不到 features_comment.csv，请先运行 09_build_features.py")
        sys.exit(1)

    df = pd.read_csv(C.F_FEATURES, encoding="utf-8-sig")
    sub = df[df["site"].isin([E, W])].copy()
    a = sub[sub["site"] == E]
    b = sub[sub["site"] == W]
    print(f"{E}: {len(a)} 条 | {W}: {len(b)} 条")
    if len(a) < 30 or len(b) < 30:
        print("!! 某一组样本过少，检验结果不可靠。")
        sys.exit(1)

    out = {}

    # ---------- 1. 描述统计 ----------
    desc = []
    for name, g in [(E, a), (W, b)]:
        desc.append({
            "景区": name, "样本量": len(g),
            "评论字数_均值": round(g["n_char"].mean(), 1),
            "评论字数_中位数": round(g["n_char"].median(), 1),
            "平均评分": round(g["rating"].mean(), 3) if g["rating"].notna().any() else np.nan,
            "积极评论数": int(g["is_positive"].sum()),
            "积极占比%": round(g["is_positive"].mean() * 100, 2),
            "中立占比%": round((g["sentiment"] == 1).mean() * 100, 2),
            "消极占比%": round((g["sentiment"] == 0).mean() * 100, 2),
            "平均积极概率": round(g["positive_prob"].mean(), 4),
            "平均提及类别数": round(g["n_cat"].mean(), 2),
        })
    out["1_描述统计"] = pd.DataFrame(desc)
    print("\n" + out["1_描述统计"].to_string(index=False))

    # ---------- 2. 积极情感比例：卡方 ----------
    tab = pd.crosstab(sub["site"], sub["sentiment_label"])
    tab = tab.reindex([E, W])
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    v = cramers_v(tab)
    out["2_情感分布卡方"] = pd.DataFrame([{
        "检验": "积极/中立/消极 分布差异", "卡方值": round(chi2, 4), "自由度": dof,
        "p值": p, "显著性": stars(p), "Cramér's_V": round(v, 4),
        "结论": f"两组情感分布{'存在' if p < 0.05 else '不存在'}显著差异",
    }])
    print("\n情感分布卡方检验：")
    print(tab.to_string())
    print(out["2_情感分布卡方"].to_string(index=False))

    # ---------- 3. 评分 / 情感分值 ----------
    rows = []
    for col, label in [("rating", "星级评分"), ("positive_prob", "积极概率"),
                       ("n_char", "评论字数"), ("n_cat", "提及类别数")]:
        x, y = a[col].dropna(), b[col].dropna()
        if len(x) < 10 or len(y) < 10:
            continue
        u, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"变量": label, "东栅均值": round(x.mean(), 4),
                     "西栅均值": round(y.mean(), 4), "U统计量": round(u, 1),
                     "p值": pv, "显著性": stars(pv),
                     "Cliff's_delta": round(cliffs_delta(x, y), 4),
                     "方向": "东栅更高" if x.mean() > y.mean() else "西栅更高"})
    out["3_连续变量差异"] = pd.DataFrame(rows)
    print("\n连续变量差异（Mann-Whitney U）：")
    print(out["3_连续变量差异"].to_string(index=False))

    # ---------- 4. 属性提及率差异 ----------
    rows = []
    for c in CATS:
        col = "men_" + c
        t = pd.crosstab(sub["site"], sub[col]).reindex([E, W]).fillna(0)
        if t.shape[1] < 2:
            t[1 - t.columns[0]] = 0
        t = t[[0, 1]]
        try:
            chi2, pv, dof, _ = stats.chi2_contingency(t)
        except Exception:
            chi2, pv = np.nan, np.nan
        rows.append({
            "类别": c,
            "东栅提及数": int(t.loc[E, 1]), "东栅提及率%": round(t.loc[E, 1] / max(t.loc[E].sum(), 1) * 100, 2),
            "西栅提及数": int(t.loc[W, 1]), "西栅提及率%": round(t.loc[W, 1] / max(t.loc[W].sum(), 1) * 100, 2),
            "提及率差(西-东)%": round((t.loc[W, 1] / max(t.loc[W].sum(), 1) -
                                   t.loc[E, 1] / max(t.loc[E].sum(), 1)) * 100, 2),
            "卡方值": round(chi2, 3) if not pd.isna(chi2) else np.nan,
            "p值": pv,
        })
    t4 = pd.DataFrame(rows)
    t4["q值(FDR)"] = bh_fdr(t4["p值"].fillna(1))
    t4["显著性"] = t4["q值(FDR)"].map(stars)
    t4["关注度更高方"] = np.where(t4["提及率差(西-东)%"] > 0, "西栅", "东栅")
    t4 = t4.sort_values("q值(FDR)")
    out["4_属性提及率差异"] = t4
    print("\n属性提及率差异（FDR 校正后）：")
    print(t4[["类别", "东栅提及率%", "西栅提及率%", "提及率差(西-东)%", "q值(FDR)", "显著性"]]
          .to_string(index=False))

    # ---------- 5. 属性情感差异 ----------
    rows = []
    for c in CATS:
        col = "sen_" + c
        x = a[col].dropna()
        y = b[col].dropna()
        if len(x) < 20 or len(y) < 20:
            rows.append({"类别": c, "东栅属性情感": np.nan, "西栅属性情感": np.nan,
                         "p值": np.nan, "样本不足": "是"})
            continue
        u, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({
            "类别": c, "东栅属性情感": round(x.mean(), 4), "西栅属性情感": round(y.mean(), 4),
            "情感差(西-东)": round(y.mean() - x.mean(), 4),
            "东栅样本": len(x), "西栅样本": len(y),
            "U统计量": round(u, 1), "p值": pv,
            "Cliff's_delta": round(cliffs_delta(x, y), 4),
            "样本不足": "",
        })
    t5 = pd.DataFrame(rows)
    t5["q值(FDR)"] = bh_fdr(t5["p值"].fillna(1))
    t5["显著性"] = t5["q值(FDR)"].map(stars)
    t5 = t5.sort_values("q值(FDR)")
    out["5_属性情感差异"] = t5
    print("\n属性情感差异（FDR 校正后）：")
    print(t5[["类别", "东栅属性情感", "西栅属性情感", "情感差(西-东)", "q值(FDR)", "显著性"]]
          .to_string(index=False))

    # ---------- 6. H2 专项：商业化在东栅是否更负面 ----------
    rows = []
    for c in ["C09_商业化程度", "C04_夜景灯光", "C12_住宿民宿",
              "C16_服务态度", "C06_历史底蕴", "C15_门票性价比"]:
        col = "sen_" + c
        if col not in sub.columns:
            continue
        x = a[sub["men_" + c] == 1][col].dropna()
        y = b[sub["men_" + c] == 1][col].dropna()
        if len(x) < 20 or len(y) < 20:
            continue
        u, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        rows.append({"类别": c, "东栅属性情感": round(x.mean(), 4), "西栅属性情感": round(y.mean(), 4),
                     "差(西-东)": round(y.mean() - x.mean(), 4), "p值": pv, "显著性": stars(pv),
                     "解读": ("西栅更正面" if y.mean() > x.mean() else "东栅更正面")})
    if rows:
        out["6_关键属性专项"] = pd.DataFrame(rows)
        print("\n关键属性专项对比（H2 检验）：")
        print(out["6_关键属性专项"].to_string(index=False))

    # ---------- 输出 ----------
    fp = C.OUT_TABLE / "对比检验.xlsx"
    with pd.ExcelWriter(fp, engine="openpyxl") as w:
        for k, v in out.items():
            v.to_excel(w, sheet_name=k[:31], index=False)
    print("=" * 60)
    print(f"全部检验结果 → {fp}")
    print("\n论文里怎么报：卡方检验报 χ²、p、Cramér's V；")
    print("组间比较报 U、p、Cliff's delta；多重比较必须说明用了 BH-FDR 校正。")


if __name__ == "__main__":
    main()
