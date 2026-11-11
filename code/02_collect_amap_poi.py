# -*- coding: utf-8 -*-
"""
Collect objective attributes (POI density) around 东栅/西栅 with the Amap Web Service API.

Output:
  data/raw/site_attributes_amap.csv   number of POIs per scenic area and per POI category
  It also prints the coordinates of both scenic areas, for the "study area" part of your paper

Usage:
  1) Put your Key into AMAP_KEY below
  2) python code/02_collect_amap_poi.py
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

AMAP_KEY = ""          # ← put your Amap Web Service Key here
RADIUS = 1200          # buffer radius in meters, adjustable

SITES = {
    "东栅": {"address": "乌镇东栅景区", "city": "嘉兴"},
    "西栅": {"address": "乌镇西栅景区", "city": "嘉兴"},
}

# Amap POI category codes (see https://lbs.amap.com/api/webservice/download)
POI_TYPES = {
    "餐饮服务": "050000",
    "住宿服务": "100000",
    "购物服务": "060000",
    "风景名胜": "110000",
    "科教文化服务": "140000",
    "体育休闲服务": "080000",
    "公共设施": "200000",
}

BASE_GEO = "https://restapi.amap.com/v3/geocode/geo"
BASE_AROUND = "https://restapi.amap.com/v3/place/around"


def geocode(address, city):
    r = requests.get(BASE_GEO, params={"address": address, "city": city,
                                       "key": AMAP_KEY}, timeout=20).json()
    if r.get("status") != "1" or not r.get("geocodes"):
        raise RuntimeError(f"geocoding failed: {r}")
    g = r["geocodes"][0]
    return g["location"], g.get("formatted_address", "")


def count_poi(location, types, radius=RADIUS):
    r = requests.get(BASE_AROUND, params={
        "location": location, "types": types, "radius": radius,
        "offset": 1, "page": 1, "key": AMAP_KEY}, timeout=20).json()
    if r.get("status") != "1":
        print(f"    query failed: {r.get('info')}")
        return None
    return int(r.get("count", 0))


def main():
    if not AMAP_KEY:
        print("Please fill in AMAP_KEY in this script first (see handbook section 3.6 for how to apply).")
        sys.exit(1)

    out = []
    for name, cfg in SITES.items():
        print(f"[{name}] geocoding...")
        loc, addr = geocode(cfg["address"], cfg["city"])
        print(f"  coordinates {loc}  address {addr}")
        row = {"site": name, "location": loc, "address": addr, "radius_m": RADIUS}
        for tname, tcode in POI_TYPES.items():
            cnt = count_poi(loc, tcode)
            row[tname] = cnt
            print(f"  {tname:<10} {cnt}")
            time.sleep(0.6)
        out.append(row)

    df = pd.DataFrame(out)
    fp = C.DATA_RAW / "site_attributes_amap.csv"
    df.to_csv(fp, index=False, encoding="utf-8-sig")
    print(f"\nSaved → {fp}")

    # Also generate the template that you have to fill in by hand
    tpl = C.DATA_RAW / "site_attributes_manual.csv"
    if not tpl.exists():
        pd.DataFrame([
            {"site": "东栅", "area_ha": "", "ticket_price": "", "open_hours": "",
             "night_open": "", "annual_visitors_wan": "", "source_note": ""},
            {"site": "西栅", "area_ha": "", "ticket_price": "", "open_hours": "",
             "night_open": "", "annual_visitors_wan": "", "source_note": ""},
        ]).to_csv(tpl, index=False, encoding="utf-8-sig")
        print(f"Manual attribute template generated → {tpl} (fill it in from the official scenic-area website / statistical yearbook and cite the source)")


if __name__ == "__main__":
    main()
