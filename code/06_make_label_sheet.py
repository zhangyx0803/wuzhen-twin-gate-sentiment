# -*- coding: utf-8 -*-
"""
Generate the manual annotation sheet (Excel).

Sampling strategy (stratified):
  60 sentences per category with a lexicon hit (true positives)
  + 300 sentences without any lexicon hit (hard negatives, the most likely to expose model flaws)
  + 200 fully random sentences (to cover the long tail)
The target total is about 1500-1800 sentences, 3-4 hours of annotation.

Input: data/clean/sentences_rule_label.csv
Output: data/label/label_sheet.xlsx
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

random.seed(C.RANDOM_STATE)
CATS = list(C.CATEGORIES.keys())

N_PER_CAT = 60
N_HARD_NEG = 300
N_RANDOM = 200


def main():
    if not C.F_SENT_LABEL_RULE.exists():
        print("Cannot find sentences_rule_label.csv, please run 05_rule_classify.py first")
        sys.exit(1)

    sdf = pd.read_csv(C.F_SENT_LABEL_RULE, encoding="utf-8-sig")
    sdf = sdf[sdf["n_char"] >= 6].copy()
    print(f"{len(sdf)} candidate sentences")

    picked = set()
    rows = []

    def add(sub, tag):
        for r in sub.itertuples(index=False):
            if r.sent_id in picked:
                continue
            picked.add(r.sent_id)
            rows.append({"sent_id": r.sent_id, "句子": r.sentence,
                         "site": r.site_sent, "抽样来源": tag,
                         "规则预测": r.cats_rule, "你的标注": "", "备注": ""})

    for c in CATS:
        sub = sdf[sdf["cats_rule"].str.contains(re.escape(c))]
        if len(sub) == 0:
            print(f"  {c}: no matching sample, skipped")
            continue
        add(sub.sample(min(N_PER_CAT, len(sub)), random_state=C.RANDOM_STATE), f"命中_{c}")

    hard = sdf[sdf["n_cats"] == 0]
    add(hard.sample(min(N_HARD_NEG, len(hard)), random_state=C.RANDOM_STATE), "难负例")

    rest = sdf[~sdf["sent_id"].isin(picked)]
    add(rest.sample(min(N_RANDOM, len(rest)), random_state=C.RANDOM_STATE), "随机")

    sheet = pd.DataFrame(rows).sample(frac=1, random_state=C.RANDOM_STATE).reset_index(drop=True)
    sheet.insert(0, "序号", range(1, len(sheet) + 1))

    # ---------- Instruction sheet ----------
    guide = [
        ["一、这份表怎么用", ""],
        ["", "1. 打开『标注』工作表，只看『句子』列，在『你的标注』列填写类别名称。"],
        ["", "2. 多个类别用竖线 | 分隔，例如：C04_夜景灯光|C20_氛围放松"],
        ["", "3. 完全不属于任何类别（寒暄、天气、无关内容）→ 填：无"],
        ["", "4. 不确定的可以在『备注』里写个问号，我会重点看这些样本。"],
        ["", "5. 标完另存为 data/label/label_done.xlsx（文件名必须是这个）。"],
        ["", ""],
        ["二、标注原则", ""],
        ["", "- 按『这句话主要在说什么』判断，不要按关键词表面意思。"],
        ["", "- 一句话可以说多件事，就标多个类别（这是多标签任务）。"],
        ["", "- 只标确实提到的属性，不要脑补。"],
        ["", "- 类别名称必须与下面清单完全一致（可以只写编号 C01）。"],
        ["", ""],
        ["三、21 个类别清单", ""],
    ]
    for c in CATS:
        guide.append([c, "、".join(C.CATEGORIES[c][:12])])
    guide += [
        ["", ""],
        ["四、标注量参考", f"总样本 {len(sheet)} 句；建议至少标完 1200 句，耗时约 3-5 小时。"],
    ]
    guide_df = pd.DataFrame(guide, columns=["说明", "内容"])

    with pd.ExcelWriter(C.F_LABEL_SHEET, engine="openpyxl") as w:
        guide_df.to_excel(w, sheet_name="说明", index=False)
        sheet.to_excel(w, sheet_name="标注", index=False)
        ws = w.sheets["标注"]
        widths = {"A": 6, "B": 22, "C": 60, "D": 8, "E": 12, "F": 32, "G": 24, "H": 14}
        for col, wd in widths.items():
            ws.column_dimensions[col].width = wd
        ws.freeze_panes = "C2"

    print("=" * 60)
    print(f"Annotation sheet generated → {C.F_LABEL_SHEET}")
    print(f"{len(sheet)} sentences in total, source distribution:")
    print(sheet["抽样来源"].value_counts().head(25).to_string())
    print("\nNext: open it in Excel and fill in the 『你的标注』 column,")
    print(f"then save it as {C.F_LABEL_DONE}")


if __name__ == "__main__":
    main()
