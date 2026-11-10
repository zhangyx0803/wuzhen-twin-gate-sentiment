# -*- coding: utf-8 -*-
"""
地点归属 + 分句。

地点归属规则（按优先级）：
  1) 评论所在页面的景点标签（poi_name）—— 最可靠，因为评论是挂在具体景点页下的
  2) 评论文本中出现"东栅""西栅"关键词
  3) 都没有 → 未提及

输入：data/clean/comments_clean.csv
输出：data/clean/comments_clean.csv（新增 site 等列，就地更新）
      data/clean/sentences.csv（句级数据）
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from textutil import split_sentences

E = C.SITE_EAST
W = C.SITE_WEST


def site_from_text(text: str) -> str:
    t = str(text)
    has_e = bool(re.search(r"东栅|东栅景区|东大街", t))
    has_w = bool(re.search(r"西栅|西栅景区|西市河", t))
    if has_e and has_w:
        return C.SITE_BOTH
    if has_e:
        return E
    if has_w:
        return W
    return ""


def site_from_poi(poi: str) -> str:
    p = str(poi)
    has_e = "东栅" in p
    has_w = "西栅" in p
    if has_e and has_w:
        return C.SITE_BOTH
    if has_e:
        return E
    if has_w:
        return W
    return ""


def main():
    if not C.F_COMMENTS_CLEAN.exists():
        print("找不到 data/clean/comments_clean.csv，请先运行 03_clean.py")
        sys.exit(1)

    df = pd.read_csv(C.F_COMMENTS_CLEAN, encoding="utf-8-sig", dtype={"comment_id": str})
    print(f"读入 {len(df)} 条评论")

    # ---------- 评论级地点 ----------
    from_text = df["content"].map(site_from_text)
    from_poi = df["poi_name"].map(site_from_poi)
    df["site_src"] = ""
    df["site"] = ""
    m = from_poi != ""
    df.loc[m, "site"] = from_poi[m]
    df.loc[m, "site_src"] = "页面标签"
    m2 = (~m) & (from_text != "")
    df.loc[m2, "site"] = from_text[m2]
    df.loc[m2, "site_src"] = "文本关键词"
    m3 = (~m) & (~m2)
    df.loc[m3, "site"] = from_text[m3].where(from_text[m3] != "", C.SITE_UNK)
    df.loc[m3, "site_src"] = "未识别"

    # 若文本明确说"两个都去了"，即使用页面标签也要修正为双栅
    fix = (from_text == C.SITE_BOTH)
    df.loc[fix, "site"] = C.SITE_BOTH
    df.loc[fix, "site_src"] = "文本明确双栅"

    print("\n评论级地点归属：")
    print(df["site"].value_counts().to_string())
    print("\n归属依据：")
    print(df["site_src"].value_counts().to_string())

    # ---------- 分句 ----------
    rows = []
    for r in df.itertuples(index=False):
        sents = split_sentences(r.content)
        for i, s in enumerate(sents):
            st = site_from_text(s)
            rows.append({
                "cid": r.cid,
                "sent_id": f"{r.cid}_S{i:02d}",
                "sent_idx": i,
                "sentence": s,
                "site_comment": r.site,
                "site_sent": st or r.site,        # 句子里没提就继承评论级地点
                "site_from_text": st,
                "n_char": len(s),
                "date": r.date,
                "ym": r.ym,
                "year": r.year,
                "month": r.month,
                "rating": r.rating,
                "platform": r.platform,
            })
    sdf = pd.DataFrame(rows)
    # 丢掉过短的句子（1-2 个字的碎片）
    sdf = sdf[sdf["n_char"] >= 3].reset_index(drop=True)
    df["n_sent"] = df["cid"].map(sdf.groupby("cid").size()).fillna(0).astype(int)

    df.to_csv(C.F_COMMENTS_CLEAN, index=False, encoding="utf-8-sig")
    sdf.to_csv(C.F_SENTENCES, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"句级数据 {len(sdf)} 句 → {C.F_SENTENCES}")
    print(f"平均每条约 {len(sdf) / max(len(df), 1):.1f} 句")
    print("\n句级地点分布：")
    print(sdf["site_sent"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("【重要】主分析样本 = 纯东栅 + 纯西栅；双栅评论留作稳健性检验")
    for s in (E, W, C.SITE_BOTH, C.SITE_UNK):
        n_c = int((df["site"] == s).sum())
        n_s = int((sdf["site_sent"] == s).sum())
        print(f"  {s:<5} 评论 {n_c:>6} 条 | 句子 {n_s:>7} 句")
    need = 5000
    for s in (E, W):
        n = int((df["site"] == s).sum())
        if n < need:
            print(f"\n!! 警告：{s} 只有 {n} 条评论（建议 >= {need}）。")
            print("   补救：① 回到阶段 3 多采几个页面；② 把时间范围往前提；")
            print("        ③ 把主分析单元换成「句级」（句级样本量通常是评论级的 5-8 倍）。")


if __name__ == "__main__":
    main()
