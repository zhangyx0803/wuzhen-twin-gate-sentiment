# -*- coding: utf-8 -*-
"""
携程评论采集器（cURL 驱动版）

设计思路：
  平台接口经常变，与其硬编码接口，不如让你把浏览器里"能跑通的请求"原样粘过来。
  本脚本解析 cURL 文本 → 变成 Python 请求 → 自动翻页 → 保存 JSON + CSV。
  接口一变，你只要重新复制一次 cURL，脚本不用改。

用法：
  1) 把 Copy as cURL 的内容保存到  data/raw/ctrip_curl.txt
  2) 探测模式（先跑这个！）：
         python code/01_collect_ctrip.py --probe
     它会请求第一页并把字段结构打印出来，同时存一份样例 JSON。
  3) 正式采集：
         python code/01_collect_ctrip.py --site 西栅 --max-pages 200
     支持断点续采：中断后重跑，会跳过已经采过的页。
"""
import argparse
import json
import random
import re
import shlex
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

try:
    import requests
except ImportError:
    print("缺少 requests 库：python -m pip install requests")
    sys.exit(1)


# ============================================================
# 一、解析 cURL
# ============================================================
def read_text_compat(path) -> str:
    """
    自动识别文本编码。
    记事本「另存为」在中文 Windows 上默认是 ANSI(GBK)，直接用 utf-8 读会报
    UnicodeDecodeError 或把中文读成乱码，导致 cURL 里的中文参数发错。
    """
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_curl(text: str):
    """把 Copy as cURL (bash) 的文本解析成 (url, headers, body_dict)"""
    # 清理换行续行符
    text = text.replace("\\\n", " ").replace("^\n", " ").replace("`\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 提取 URL
    m = re.search(r"curl\s+(?:-X\s+\w+\s+)?['\"]([^'\"]+)['\"]", text)
    if not m:
        m = re.search(r"curl\s+([^\s'\"-][^\s'\"]*)", text)
    if not m:
        raise ValueError("没解析出 URL，请确认粘贴的是完整的 cURL（以 curl ' 开头）")
    url = m.group(1)

    # 提取 headers
    headers = {}
    for hm in re.finditer(r"-H\s+'([^:']+):\s*([^']*)'", text):
        k, v = hm.group(1).strip(), hm.group(2).strip()
        if k.startswith(":"):          # http2 伪头，丢掉
            continue
        if k.lower() in ("content-length", "host", "accept-encoding", "connection"):
            continue
        headers[k] = v

    # 提取 body
    body_raw = None
    for pat in (r"--data-raw\s+'([^']*)'", r"--data\s+'([^']*)'",
                r"--data-raw\s+\"([^\"]*)\"", r"--data\s+\"([^\"]*)\""):
        bm = re.search(pat, text)
        if bm:
            body_raw = bm.group(1)
            break

    body = None
    if body_raw:
        try:
            body = json.loads(body_raw)
        except Exception:
            body = dict(re.findall(r"([A-Za-z0-9_]+)=([^&]*)", body_raw)) or None

    if not any(k.lower() == "user-agent" for k in headers):
        headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/124.0 Safari/537.36")
    return url, headers, body


PAGE_KEYS = ("pageindex", "page", "pageno", "pagenum", "pageidx", "pagenumber")
SIZE_KEYS = ("pagesize", "limit", "rows", "count")


def find_page_field(body, _path=()):
    """
    递归查找分页字段，返回 (page_path, size_path)。
    path 是「从根到该字段」的键路径，例如 ['arg', 'pageIndex']。
    携程等平台常把分页参数藏在 arg / head 这类外层对象里，只查顶层会找不到。
    """
    if not isinstance(body, dict):
        return None, None
    for k in body:
        if str(k).lower() in PAGE_KEYS:
            page_path = list(_path) + [k]
            size_path = None
            for k2 in body:
                if k2 != k and str(k2).lower() in SIZE_KEYS:
                    size_path = list(_path) + [k2]
                    break
            return page_path, size_path
    for k, v in body.items():
        if isinstance(v, dict):
            p, s = find_page_field(v, _path + (k,))
            if p:
                return p, s
    return None, None


def set_nested(d: dict, path, value):
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def get_nested(d: dict, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# ============================================================
# 二、从返回 JSON 里挖评论（结构自适应）
# ============================================================
TEXT_KEYS = ["content", "commentContent", "text", "reviewContent", "comment", "body"]
TIME_KEYS = ["publishTime", "publishDate", "createTime", "commentTime", "date", "publishTimeStr"]
SCORE_KEYS = ["score", "rating", "star", "commentScore", "overallScore"]
ID_KEYS = ["commentId", "id", "reviewId", "commentID"]


def _walk_lists(obj, path=""):
    """递归找出所有"元素是 dict 且含文本字段"的列表"""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _walk_lists(v, f"{path}.{k}")
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        keys = set(obj[0].keys())
        if keys & set(TEXT_KEYS):
            found.append((path, obj))
        for i, item in enumerate(obj[:1]):
            found += _walk_lists(item, f"{path}[{i}]")
    return found


def pick(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    # 大小写不敏感兜底
    low = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in low and low[k.lower()] not in (None, ""):
            return low[k.lower()]
    return default


def extract_comments(resp_json, platform="携程", poi=""):
    rows = []
    for _path, lst in _walk_lists(resp_json):
        for it in lst:
            txt = pick(it, TEXT_KEYS)
            if not txt or not isinstance(txt, str):
                continue
            rows.append({
                "platform": platform,
                "poi_name": poi,
                "comment_id": pick(it, ID_KEYS, ""),
                "content": re.sub(r"<[^>]+>", " ", txt).strip(),
                "rating": pick(it, SCORE_KEYS, ""),
                "date": pick(it, TIME_KEYS, ""),
            })
    return rows


# ============================================================
# 三、主流程
# ============================================================
CURL_FILE = C.DATA_RAW / "ctrip_curl.txt"
RAW_DIR = C.DATA_RAW / "ctrip_json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
CKPT = C.DATA_RAW / "ctrip_checkpoint.json"
OUT_CSV = C.DATA_RAW / "ctrip_comments.csv"


def load_ckpt():
    if CKPT.exists():
        return json.loads(CKPT.read_text(encoding="utf-8"))
    return {"done_pages": [], "rows": []}


def save_ckpt(ck):
    CKPT.write_text(json.dumps(ck, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="西栅", help="页面所属景点，写进 poi_name 字段")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=2.5, help="每页间隔基准秒数")
    ap.add_argument("--probe", action="store_true", help="只请求第一页并打印字段结构")
    ap.add_argument("--platform", default="携程")
    args = ap.parse_args()

    if not CURL_FILE.exists():
        print(f"找不到 {CURL_FILE}\n请按手册 3.2 把 Copy as cURL 的内容保存到该文件。")
        sys.exit(1)

    url, headers, body = parse_curl(read_text_compat(CURL_FILE))
    page_path, size_path = find_page_field(body)
    print("=" * 60)
    print(f"URL        : {url}")
    print(f"请求体字段 : {list(body.keys()) if isinstance(body, dict) else '(非 JSON / 无 body)'}")
    if page_path:
        print(f"分页字段   : {'.'.join(map(str, page_path))}"
              f"  (每页条数: {'.'.join(map(str, size_path)) if size_path else '未识别'})")
    else:
        print("分页字段   : (未识别 → 只会采到第 1 页)")
        print("   排查：在 F12 的 Payload 里看看分页参数叫什么名字，")
        print("         并把名字告诉我；或手动把 cURL 里的 pageIndex 从 1 改成 2 再复制一次。")
    print("=" * 60)

    sess = requests.Session()
    sess.headers.update(headers)

    def fetch(page):
        payload = json.loads(json.dumps(body)) if isinstance(body, dict) else body
        if isinstance(payload, dict) and page_path:
            set_nested(payload, page_path, page)
            if size_path and not get_nested(payload, size_path):
                set_nested(payload, size_path, 10)
        for attempt in range(4):
            try:
                r = sess.post(url, json=payload, timeout=25)
                if r.status_code == 200:
                    return r.json()
                print(f"  第 {page} 页 HTTP {r.status_code}，第 {attempt+1} 次重试…")
            except Exception as e:
                print(f"  第 {page} 页异常 {type(e).__name__}: {e}，第 {attempt+1} 次重试…")
            time.sleep(3 + attempt * 5)
        return None

    # ---------- 探测模式 ----------
    if args.probe:
        js = fetch(1)
        if js is None:
            print("请求失败。常见原因：Cookie 过期、URL 复制的是别的请求、需要代理。")
            sys.exit(1)
        (RAW_DIR / "probe_page1.json").write_text(
            json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始返回已保存: {RAW_DIR / 'probe_page1.json'}")
        lists = _walk_lists(js)
        if not lists:
            print("!! 没在返回里找到含文本字段的列表。请打开 probe_page1.json 人工看一眼：")
            print("   如果返回的是 HTML 或错误提示，说明 cURL 复制错了（可能复制成了页面请求）")
            print("   如果有评论但字段名很特殊，请把字段名告诉我，我帮你改 TEXT_KEYS")
        for path, lst in lists:
            print(f"  找到评论列表: {path}  长度 {len(lst)}")
            print(f"  首条字段: {list(lst[0].keys())}")
            print(f"  首条样例: {json.dumps(lst[0], ensure_ascii=False)[:300]}")
        rows = extract_comments(js, args.platform, args.site)
        print(f"  按默认字段规则能提取出 {len(rows)} 条")
        return

    # ---------- 正式采集 ----------
    ck = load_ckpt()
    done = set(ck["done_pages"])
    rows = ck["rows"]
    seen = {r["comment_id"] for r in rows if r.get("comment_id")}
    empty_streak = 0

    for page in range(1, args.max_pages + 1):
        if page in done:
            continue
        js = fetch(page)
        if js is None:
            print(f"第 {page} 页连续失败，停止。已采 {len(rows)} 条，重跑可续采。")
            break
        new = extract_comments(js, args.platform, args.site)
        added = 0
        for r in new:
            key = r.get("comment_id") or r["content"][:80]
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
            added += 1
        done.add(page)
        ck["done_pages"], ck["rows"] = sorted(done), rows
        save_ckpt(ck)
        (RAW_DIR / f"page_{page:04d}.json").write_text(
            json.dumps(js, ensure_ascii=False), encoding="utf-8")
        print(f"第 {page:>4} 页: 抓到 {len(new):>3} 条, 新增 {added:>3} 条, 累计 {len(rows)} 条")

        empty_streak = empty_streak + 1 if len(new) == 0 else 0
        if empty_streak >= 3:
            print("连续 3 页没有新评论，认为已到末尾，停止。")
            break
        time.sleep(args.sleep + random.random() * 1.5)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print("=" * 60)
        print(f"完成：共 {len(df)} 条 → {OUT_CSV}")
        print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
    else:
        print("一条都没采到，请先跑 --probe 排查。")


if __name__ == "__main__":
    main()
