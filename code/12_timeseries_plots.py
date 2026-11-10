# -*- coding: utf-8 -*-
"""
时间序列分析 + 全部论文级图表输出（300 dpi）。

产出图（outputs/figures/）：
    图1_情感分布对比.png        堆叠柱状图
    图2_属性提及率对比.png      双栅横向对比条形图（按差值排序）
    图3_属性情感对比.png        哑铃图（同一属性在两栅的情感差异）
    图7_月度情感趋势.png        月度积极占比折线（含 3 个月滚动平均）
    图8_年-月热力图.png         年份 × 月份 的积极情感占比热力图
    图9_评论量与评分趋势.png    月度评论量柱 + 平均评分折线
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import set_chinese_font

plt = set_chinese_font()
if plt is None:
    print("matplotlib 不可用，请先安装：python -m pip install matplotlib")
    sys.exit(1)

CATS = list(C.CATEGORIES.keys())
E, W = C.SITE_EAST, C.SITE_WEST
SHORT = {c: c.split("_", 1)[1] for c in CATS}
COLORS = {E: "#C0504D", W: "#4F81BD"}


def save(fig, name):
    fp = C.OUT_FIG / name
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {fp.name}")


def fig1_sentiment(df):
    tab = pd.crosstab(df["site"], df["sentiment_label"])
    tab = tab.reindex(columns=["消极", "中立", "积极"]).fillna(0)
    tab = tab.loc[[s for s in (E, W) if s in tab.index]]
    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bottom = np.zeros(len(pct))
    for col, color in zip(["消极", "中立", "积极"], ["#C0504D", "#BFBFBF", "#4F81BD"]):
        ax.bar(pct.index, pct[col], bottom=bottom, label=col, color=color, width=0.5)
        for i, v in enumerate(pct[col]):
            if v > 4:
                ax.text(i, bottom[i] + v / 2, f"{v:.1f}%", ha="center", va="center",
                        color="white", fontsize=10)
        bottom += pct[col].values
    ax.set_ylabel("占比 (%)")
    ax.set_title("东栅与西栅游客情感分布对比")
    ax.legend(loc="lower right", ncol=3, fontsize=9, framealpha=0.9)
    ax.set_ylim(0, 100)
    save(fig, "图1_情感分布对比.png")


def fig2_mention(df):
    rows = []
    for c in CATS:
        r = {"类别": SHORT[c]}
        for s in (E, W):
            sub = df[df["site"] == s]
            r[s] = sub["men_" + c].mean() * 100 if len(sub) else np.nan
        r["差"] = r[W] - r[E]
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("差")
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t[E], height=0.4, color=COLORS[E], label=E)
    ax.barh(y + 0.2, t[W], height=0.4, color=COLORS[W], label=W)
    ax.set_yticks(y)
    ax.set_yticklabels(t["类别"], fontsize=9)
    ax.set_xlabel("提及率 (%)")
    ax.set_title("东栅与西栅各属性类别提及率对比")
    ax.legend()
    save(fig, "图2_属性提及率对比.png")


def fig3_aspect_sentiment(df):
    rows = []
    for c in CATS:
        r = {"类别": SHORT[c]}
        for s in (E, W):
            sub = df[(df["site"] == s) & (df["men_" + c] == 1)]
            r[s] = sub["sen_" + c].mean() if len(sub) >= 20 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).dropna()
    if t.empty:
        print("  (属性情感样本不足，跳过图3)")
        return
    t = t.sort_values(E)
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    y = np.arange(len(t))
    for i, r in enumerate(t.itertuples(index=False)):
        ax.plot([r[1], r[2]], [i, i], color="#BBBBBB", zorder=1, lw=2)
    ax.scatter(t[E], y, color=COLORS[E], s=55, label=E, zorder=2)
    ax.scatter(t[W], y, color=COLORS[W], s=55, label=W, zorder=2)
    ax.axvline(0, color="#888888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(t["类别"], fontsize=9)
    ax.set_xlabel("属性情感均值（-1 消极 ~ +1 积极）")
    ax.set_title("同一属性在东栅与西栅的情感倾向差异")
    ax.legend()
    save(fig, "图3_属性情感对比.png")


def fig7_monthly(df):
    d = df[df["ym"].astype(str).str.len() >= 7].copy()
    if d.empty:
        print("  (无有效月份，跳过时间序列图)")
        return
    g = d.groupby(["site", "ym"]).agg(积极占比=("is_positive", "mean"),
                                      评论数=("cid", "count")).reset_index()
    g["积极占比"] *= 100
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for s in (E, W):
        sub = g[g["site"] == s].sort_values("ym")
        if sub.empty:
            continue
        ax.plot(sub["ym"], sub["积极占比"], marker="o", ms=3.5, lw=1.2,
                color=COLORS[s], label=f"{s} 月度积极占比")
        roll = sub["积极占比"].rolling(3, min_periods=2).mean()
        ax.plot(sub["ym"], roll, lw=2.4, ls="--", color=COLORS[s], alpha=0.85,
                label=f"{s} 3个月滚动平均")
    ticks = [t for i, t in enumerate(sorted(g["ym"].unique())) if i % 6 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("积极情感占比 (%)")
    ax.set_title("东栅与西栅游客积极情感的月度变化")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "图7_月度情感趋势.png")


def fig8_heatmap(df):
    d = df[df["year"].notna() & df["month"].notna()].copy()
    if d.empty:
        return
    d["year"] = d["year"].astype(int)
    d["month"] = d["month"].astype(int)
    for s in (E, W):
        sub = d[d["site"] == s]
        if len(sub) < 50:
            continue
        piv = sub.pivot_table(index="year", columns="month", values="is_positive",
                              aggfunc="mean") * 100
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        im = ax.imshow(piv.values, cmap="RdYlBu_r", aspect="auto", vmin=40, vmax=100)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{m}月" for m in piv.columns], fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=9)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                            color="black")
        fig.colorbar(im, ax=ax, label="积极情感占比 (%)")
        ax.set_title(f"{s}：年份 × 月份 积极情感占比热力图")
        save(fig, f"图8_年月热力图_{s}.png")


def fig9_volume(df):
    d = df[df["ym"].astype(str).str.len() >= 7]
    if d.empty:
        return
    g = d.groupby(["site", "ym"]).agg(评论数=("cid", "count"),
                                      平均评分=("rating", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.4
    yms = sorted(g["ym"].unique())
    x = np.arange(len(yms))
    for k, s in enumerate((E, W)):
        sub = g[g["site"] == s].set_index("ym").reindex(yms)
        ax.bar(x + (k - 0.5) * width, sub["评论数"].fillna(0), width,
               color=COLORS[s], label=f"{s} 评论量", alpha=0.85)
    ax.set_ylabel("评论数量")
    ax.set_xticks([i for i in x if i % 6 == 0])
    ax.set_xticklabels([yms[i] for i in x if i % 6 == 0], rotation=45, ha="right", fontsize=8)
    ax2 = ax.twinx()
    for s in (E, W):
        sub = g[g["site"] == s].set_index("ym").reindex(yms)
        ax2.plot(x, sub["平均评分"], color=COLORS[s], lw=1.8, ls=":")
    ax2.set_ylabel("平均评分")
    ax2.set_ylim(1, 5)
    ax.set_title("月度评论量与平均评分（虚线为评分）")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "图9_评论量与评分趋势.png")


def fig_importance():
    fp = C.OUT_TABLE / "随机森林_两栅重要性对比.csv"
    if not fp.exists():
        print("  (还没有随机森林结果，跳过重要性对比图。先跑 11_rf_shap.py)")
        return
    t = pd.read_csv(fp)
    t["类别"] = t["特征"].str.replace(r"^(men_|sen_)", "", regex=True).str.split("_").str[-1]
    t["类型"] = np.where(t["特征"].str.startswith("men_"), "提及", "属性情感")
    t = t.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["东栅重要性"], height=0.4, color=COLORS[E], label=E)
    ax.barh(y + 0.2, t["西栅重要性"], height=0.4, color=COLORS[W], label=W)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}（{k}）" for c, k in zip(t["类别"], t["类型"])], fontsize=9)
    ax.set_xlabel("置换重要性（对 AUC 的贡献）")
    ax.set_title("驱动游客积极情感的关键因素：东栅 vs 西栅")
    ax.legend()
    save(fig, "图4_随机森林特征重要性对比.png")


def main():
    if not C.F_FEATURES.exists():
        print("找不到 features_comment.csv，请先运行 09_build_features.py")
        sys.exit(1)
    df = pd.read_csv(C.F_FEATURES, encoding="utf-8-sig")
    df = df[df["site"].isin([E, W])].copy()
    df["ym"] = df["ym"].astype(str)
    print(f"绘图数据 {len(df)} 条\n")
    print("生成图表：")
    fig1_sentiment(df)
    fig2_mention(df)
    fig3_aspect_sentiment(df)
    fig_importance()
    fig7_monthly(df)
    fig8_heatmap(df)
    fig9_volume(df)
    print(f"\n全部图表 → {C.OUT_FIG}")
    print("注意：matplotlib 默认字体若无中文会显示方框，请确认 00_check_env.py 报了字体 OK。")


if __name__ == "__main__":
    main()
