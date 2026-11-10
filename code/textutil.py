# -*- coding: utf-8 -*-
"""
中文文本公共工具：分词 / 分句 / 停用词 / 中文字体。
被 03、04、05、07、11 等脚本共用。
"""
import re
from pathlib import Path

import config as C

# ---------- jieba 可选 ----------
try:
    import jieba
    jieba.setLogLevel(60)
    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False


_DEFAULT_STOPWORDS = """
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 他 她 它 我们 你们 他们 这个 那个 这些 那些 什么 怎么 为什么 因为 所以
但是 而且 然后 还是 还有 就是 可以 应该 觉得 感觉 真的 非常 特别 一下 一点 有点 太
更 最 已经 曾经 一直 总是 其实 可能 或者 如果 虽然 不过 而是 只是 而已 呢 吗 吧 啊
呀 哇 嗯 哦 哈 呵 啦 嘛 之 与 及 等 等等 被 把 让 给 从 向 对 于 为 以 并 或 中 里
内 外 前 后 时 时候 现在 今天 昨天 明天 大家 一起 地方 东西 这样 那样 如何 多少 几
个 只 条 张 次 遍 趟 天 年 月 日 点 分 秒 元 块 米 公里 之类 之类的话 一般 比较 觉得
建议 希望 下次 终于 居然 竟然 简直 实在 确实 当然 于是 于是乎 反正 总之 大概 也许
之后 之前 当时 后来 其中 以及 无法 不会 不能 不要 不用 不是 没啥 没什么 一次
""".split()


def load_stopwords() -> set:
    sw = set(_DEFAULT_STOPWORDS)
    for p in [C.DOCS / "词表" / "stopwords_zh.txt",
              C.DOCS / "stopwords_zh.txt"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):
                    sw.add(w)
    return sw


STOPWORDS = load_stopwords()


def tokenize(text: str):
    """分词。有 jieba 用 jieba；没有则用「单字 + 双字组合」兜底。"""
    text = re.sub(r"\s+", "", text)
    if not text:
        return []
    if HAS_JIEBA:
        return [w for w in jieba.lcut(text) if w.strip()]
    # 兜底：中文按字，英文/数字按词，并补充相邻字的二元组
    chars = re.findall(r"[\u4e00-\u9fa5]", text)
    words = re.findall(r"[A-Za-z0-9]+", text)
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return chars + bigrams + words


def content_words(text: str):
    """去掉停用词后的实词"""
    return [w for w in tokenize(text)
            if w not in STOPWORDS and len(w.strip()) > 1 or
            (len(w) == 1 and re.match(r"[\u4e00-\u9fa5]", w) and w not in STOPWORDS)]


def split_sentences(text: str, strong=None, weak=None, max_len=None):
    """
    分句：先按强标点（。！？；）切；
    切出来的句子若仍超过 max_len 字，再按弱标点（，、）继续切。
    """
    strong = strong or C.STRONG_PUNCT
    weak = weak or C.WEAK_PUNCT
    max_len = max_len or C.MAX_SENT_LEN
    text = re.sub(r"[\r\n\t]+", "。", str(text))
    raw = re.split(f"[{re.escape(strong)}]+", text)
    out = []
    for seg in raw:
        seg = seg.strip()
        if not seg:
            continue
        if len(seg) <= max_len:
            out.append(seg)
            continue
        buf = ""
        for piece in re.split(f"[{re.escape(weak)}]+", seg):
            piece = piece.strip()
            if not piece:
                continue
            if buf and len(buf) + len(piece) + 1 > max_len:
                out.append(buf)
                buf = piece
            else:
                buf = f"{buf}，{piece}" if buf else piece
        if buf:
            out.append(buf)
    return [s for s in out if s]


def norm_text(t: str) -> str:
    """强归一化：用于去重比对"""
    t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", t)
    return t.lower()


def set_chinese_font():
    """给 matplotlib 设置中文字体，避免图上出现方框"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "Noto Sans CJK SC"]:
            if cand in names:
                plt.rcParams["font.sans-serif"] = [cand]
                break
        plt.rcParams["axes.unicode_minus"] = False
        return plt
    except Exception as e:
        print(f"[警告] 设置中文字体失败: {e}")
        return None
