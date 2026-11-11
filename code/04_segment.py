# -*- coding: utf-8 -*-
"""
Site attribution + sentence splitting.

Site attribution rules (in priority order):
  1) The scenic-area label of the comment's page (poi_name) -- the most reliable, since comments hang under a specific scenic-area page
  2) The keywords 东栅 / 西栅 appearing in the comment text
  3) Neither -> not mentioned

Input: data/clean/comments_clean.csv
Output: data/clean/comments_clean.csv (adds a site column and others, updated in place)
        data/clean/sentences.csv (sentence-level data)
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
        print("Cannot find data/clean/comments_clean.csv, please run 03_clean.py first")
        sys.exit(1)

    df = pd.read_csv(C.F_COMMENTS_CLEAN, encoding="utf-8-sig", dtype={"comment_id": str})
    print(f"Read {len(df)} comments")

    # ---------- Comment-level site ----------
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

    # If the text explicitly says "went to both", override the page label with 双栅 too
    fix = (from_text == C.SITE_BOTH)
    df.loc[fix, "site"] = C.SITE_BOTH
    df.loc[fix, "site_src"] = "文本明确双栅"

    print("\nComment-level site attribution:")
    print(df["site"].value_counts().to_string())
    print("\nBasis of attribution:")
    print(df["site_src"].value_counts().to_string())

    # ---------- Sentence splitting ----------
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
                "site_sent": st or r.site,        # inherit the comment-level site if the sentence mentions none
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
    # Drop sentences that are too short (1-2 character fragments)
    sdf = sdf[sdf["n_char"] >= 3].reset_index(drop=True)
    df["n_sent"] = df["cid"].map(sdf.groupby("cid").size()).fillna(0).astype(int)

    df.to_csv(C.F_COMMENTS_CLEAN, index=False, encoding="utf-8-sig")
    sdf.to_csv(C.F_SENTENCES, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"Sentence-level data: {len(sdf)} sentences → {C.F_SENTENCES}")
    print(f"About {len(sdf) / max(len(df), 1):.1f} sentences per comment on average")
    print("\nSentence-level site distribution:")
    print(sdf["site_sent"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("[IMPORTANT] main analysis sample = pure 东栅 + pure 西栅; 双栅 comments are kept for robustness checks")
    for s in (E, W, C.SITE_BOTH, C.SITE_UNK):
        n_c = int((df["site"] == s).sum())
        n_s = int((sdf["site_sent"] == s).sum())
        print(f"  {s:<5} comments {n_c:>6} | sentences {n_s:>7}")
    need = 5000
    for s in (E, W):
        n = int((df["site"] == s).sum())
        if n < need:
            print(f"\n!! WARNING: {s} has only {n} comments (recommended >= {need}).")
            print("   Remedies: (1) go back to stage 3 and collect more pages; (2) move the time range earlier;")
            print("        (3) switch the main analysis unit to sentence level (the sentence sample is usually 5-8x the comment sample).")


if __name__ == "__main__":
    main()
