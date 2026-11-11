# -*- coding: utf-8 -*-
"""
Ctrip comment collector (cURL-driven version)

Design rationale:
  Platform APIs change all the time. Rather than hard-coding an endpoint, paste the request that
  works in your browser verbatim. The script parses the cURL text -> builds Python requests ->
  paginates automatically -> saves JSON + CSV; when the API changes, just copy the cURL again.

Usage:
  1) Save the "Copy as cURL" content to  data/raw/ctrip_curl.txt
  2) Probe mode (run this first!):
         python code/01_collect_ctrip.py --probe
     It requests the first page, prints the field structure and stores a sample JSON.
  3) Actual collection:
         python code/01_collect_ctrip.py --site 西栅 --max-pages 200
     Resumable: after an interruption, rerun it and already-collected pages are skipped.
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
    print("Missing the requests library: python -m pip install requests")
    sys.exit(1)


# ============================================================
# 1. Parsing cURL
# ============================================================
def read_text_compat(path) -> str:
    """
    Detect the text encoding automatically.
    On Chinese Windows, Notepad's "Save as" defaults to ANSI (GBK), so reading it as utf-8
    raises UnicodeDecodeError or garbles Chinese and makes the cURL parameters wrong.
    """
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_curl(text: str):
    """Parse "Copy as cURL (bash)" text into (url, headers, body_dict)"""
    # Normalize line continuations
    text = text.replace("\\\n", " ").replace("^\n", " ").replace("`\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # Extract the URL
    m = re.search(r"curl\s+(?:-X\s+\w+\s+)?['\"]([^'\"]+)['\"]", text)
    if not m:
        m = re.search(r"curl\s+([^\s'\"-][^\s'\"]*)", text)
    if not m:
        raise ValueError("could not parse the URL; make sure you pasted a complete cURL (starting with curl ')")
    url = m.group(1)

    # Extract headers
    headers = {}
    for hm in re.finditer(r"-H\s+'([^:']+):\s*([^']*)'", text):
        k, v = hm.group(1).strip(), hm.group(2).strip()
        if k.startswith(":"):          # http2 pseudo-header, drop it
            continue
        if k.lower() in ("content-length", "host", "accept-encoding", "connection"):
            continue
        headers[k] = v

    # Extract the body
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
    Recursively locate the pagination fields and return (page_path, size_path).
    path is the key path from the root to that field, e.g. ['arg', 'pageIndex'].
    Ctrip often hides pagination in outer objects like arg / head, so a top-level lookup fails.
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
# 2. Mining comments out of the returned JSON (structure-adaptive)
# ============================================================
TEXT_KEYS = ["content", "commentContent", "text", "reviewContent", "comment", "body"]
TIME_KEYS = ["publishTime", "publishDate", "createTime", "commentTime", "date", "publishTimeStr"]
SCORE_KEYS = ["score", "rating", "star", "commentScore", "overallScore"]
ID_KEYS = ["commentId", "id", "reviewId", "commentID"]


def _walk_lists(obj, path=""):
    """Recursively find every list whose elements are dicts containing a text field"""
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
    # Case-insensitive fallback
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
# 3. Main flow
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
    ap.add_argument("--site", default="西栅", help="scenic area this page belongs to, written into the poi_name field")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=2.5, help="base number of seconds between pages")
    ap.add_argument("--probe", action="store_true", help="only request the first page and print the field structure")
    ap.add_argument("--platform", default="携程")
    args = ap.parse_args()

    if not CURL_FILE.exists():
        print(f"Cannot find {CURL_FILE}\nSave the 'Copy as cURL' content to that file as described in handbook section 3.2.")
        sys.exit(1)

    url, headers, body = parse_curl(read_text_compat(CURL_FILE))
    page_path, size_path = find_page_field(body)
    print("=" * 60)
    print(f"URL         : {url}")
    print(f"Body fields : {list(body.keys()) if isinstance(body, dict) else '(not JSON / no body)'}")
    if page_path:
        print(f"Page field  : {'.'.join(map(str, page_path))}"
              f"  (items per page: {'.'.join(map(str, size_path)) if size_path else 'not detected'})")
    else:
        print("Page field  : (not detected -> only page 1 will be collected)")
        print("   Diagnose: look at the Payload in F12 to see what the pagination parameter is called,")
        print("             and tell me the name; or change pageIndex from 1 to 2 in the cURL and copy it again.")
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
                print(f"  page {page} returned HTTP {r.status_code}, retry #{attempt+1}...")
            except Exception as e:
                print(f"  page {page} raised {type(e).__name__}: {e}, retry #{attempt+1}...")
            time.sleep(3 + attempt * 5)
        return None

    # ---------- Probe mode ----------
    if args.probe:
        js = fetch(1)
        if js is None:
            print("Request failed. Common causes: expired Cookie, the URL belongs to a different request, a proxy is required.")
            sys.exit(1)
        (RAW_DIR / "probe_page1.json").write_text(
            json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Raw response saved: {RAW_DIR / 'probe_page1.json'}")
        lists = _walk_lists(js)
        if not lists:
            print("!! No list containing a text field was found in the response. Open probe_page1.json and look at it manually:")
            print("   If the response is HTML or an error message, the cURL was copied wrong (probably a page request)")
            print("   If there are comments but the field names are unusual, tell me the names and I will adjust TEXT_KEYS")
        for path, lst in lists:
            print(f"  comment list found: {path}  length {len(lst)}")
            print(f"  first item fields: {list(lst[0].keys())}")
            print(f"  first item sample: {json.dumps(lst[0], ensure_ascii=False)[:300]}")
        rows = extract_comments(js, args.platform, args.site)
        print(f"  {len(rows)} comments can be extracted with the default field rules")
        return

    # ---------- Actual collection ----------
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
            print(f"page {page} failed repeatedly, stopping. {len(rows)} collected so far; rerun to resume.")
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
        print(f"page {page:>4}: fetched {len(new):>3}, new {added:>3}, total {len(rows)}")

        empty_streak = empty_streak + 1 if len(new) == 0 else 0
        if empty_streak >= 3:
            print("3 consecutive pages had no new comments; assuming the end, stopping.")
            break
        time.sleep(args.sleep + random.random() * 1.5)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print("=" * 60)
        print(f"Done: {len(df)} comments in total → {OUT_CSV}")
        print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    else:
        print("No comment was collected at all; run --probe first to diagnose.")


if __name__ == "__main__":
    main()
