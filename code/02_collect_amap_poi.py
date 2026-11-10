# -*- coding: utf-8 -*-
"""
用高德 Web 服务 API 采集东栅/西栅周边的客观属性（POI 密度）。

产出：
  data/raw/site_attributes_amap.csv   每个景区、每类 POI 的数量
  另外会打印两个景区的经纬度，供你在论文里写"研究范围"

用法：
  1) 把 Key 填到下面 AMAP_KEY
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

AMAP_KEY = ""          # ← 填你的高德 Web 服务 Key
RADIUS = 1200          # 缓冲区半径（米），可改

SITES = {
    "东栅": {"address": "乌镇东栅景区", "city": "嘉兴"},
    "西栅": {"address": "乌镇西栅景区", "city": "嘉兴"},
}

# 高德 POI 分类编码（详见 https://lbs.amap.com/api/webservice/download）
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
        raise RuntimeError(f"地理编码失败: {r}")
    g = r["geocodes"][0]
    return g["location"], g.get("formatted_address", "")


def count_poi(location, types, radius=RADIUS):
    r = requests.get(BASE_AROUND, params={
        "location": location, "types": types, "radius": radius,
        "offset": 1, "page": 1, "key": AMAP_KEY}, timeout=20).json()
    if r.get("status") != "1":
        print(f"    查询失败: {r.get('info')}")
        return None
    return int(r.get("count", 0))


def main():
    if not AMAP_KEY:
        print("请先在脚本里填写 AMAP_KEY（申请方式见手册 3.6）。")
        sys.exit(1)

    out = []
    for name, cfg in SITES.items():
        print(f"[{name}] 地理编码…")
        loc, addr = geocode(cfg["address"], cfg["city"])
        print(f"  坐标 {loc}  地址 {addr}")
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
    print(f"\n已保存 → {fp}")

    # 顺手生成人工需要填的模板
    tpl = C.DATA_RAW / "site_attributes_manual.csv"
    if not tpl.exists():
        pd.DataFrame([
            {"site": "东栅", "area_ha": "", "ticket_price": "", "open_hours": "",
             "night_open": "", "annual_visitors_wan": "", "source_note": ""},
            {"site": "西栅", "area_ha": "", "ticket_price": "", "open_hours": "",
             "night_open": "", "annual_visitors_wan": "", "source_note": ""},
        ]).to_csv(tpl, index=False, encoding="utf-8-sig")
        print(f"已生成人工属性模板 → {tpl}（请查景区官网/统计年鉴填写，并注明来源）")


if __name__ == "__main__":
    main()
