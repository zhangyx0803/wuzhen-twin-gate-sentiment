# -*- coding: utf-8 -*-
"""环境自检：跑通这个脚本，说明环境没问题。"""
import sys
import platform

def check(name, optional=False):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK]   {name:<14} {ver}")
        return True
    except Exception as e:
        flag = "[可选]" if optional else "[缺失]"
        print(f"  {flag} {name:<14} {type(e).__name__}: {e}")
        return optional

def main():
    print("=" * 60)
    print("乌镇双栅研究 - 环境自检")
    print("=" * 60)
    print(f"Python 版本 : {sys.version.split()[0]}")
    print(f"操作系统    : {platform.system()} {platform.release()}")
    print(f"Python 路径 : {sys.executable}")
    if sys.version_info < (3, 9):
        print("!" * 62)
        print("!! 严重警告：你的 Python 版本低于 3.9，本项目的核心库装不上。")
        print("!! Python 3.7 已于 2023 年 6 月停止维护，请二选一：")
        print("!!   [推荐] 安装 Anaconda（自带 Python 3.12 和全部科学计算库）")
        print("!!           https://www.anaconda.com/download")
        print("!!           装完后从「开始菜单 → Anaconda Prompt」进入，再 cd 到项目目录")
        print("!!   [备选] 只想先试一下：pip install -r requirements-py37.txt")
        print("!" * 62)
    print("-" * 60)

    core = ["pandas", "numpy", "sklearn", "jieba", "matplotlib"]
    opt = ["seaborn", "openpyxl", "requests", "tqdm", "shap",
           "statsmodels", "joblib", "snownlp"]

    ok = True
    print("核心库：")
    for m in core:
        ok &= check(m)
    print("推荐/可选库：")
    for m in opt:
        check(m, optional=True)

    print("-" * 60)
    # 中文字体检查（Windows 上必须设置，否则图上全是方框）
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]:
            if cand in names:
                print(f"  [OK]   中文字体可用: {cand}")
                break
        else:
            print("  [警告] 未找到常见中文字体，图表中文会显示为方框")
    except Exception as e:
        print(f"  [警告] 字体检查失败: {e}")

    print("-" * 60)
    # 目录检查
    try:
        from config import (DATA_RAW, DATA_CLEAN, DATA_LABEL, DATA_FEATURE,
                            OUT_TABLE, OUT_FIG, OUT_LOG)
        for d in (DATA_RAW, DATA_CLEAN, DATA_LABEL, DATA_FEATURE, OUT_TABLE, OUT_FIG, OUT_LOG):
            print(f"  [OK]   目录: {d}")
    except Exception as e:
        print(f"  [错误] 无法导入 config.py，请确认它和本脚本在同一文件夹: {e}")
        ok = False

    print("=" * 60)
    if ok:
        print("环境自检通过，可以进入下一步（阶段 3 数据采集）。")
    else:
        old = sys.version_info < (3, 9)
        req = "requirements-py37.txt" if old else "requirements.txt"
        print("有核心库缺失，请先执行：")
        print(f"  python -m pip install -r {req} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if old:
            print("")
            print("提示：Python 3.7 上这些库多半装不上（scikit-learn / shap 尤其容易失败）。")
            print("      更省事的做法是装 Anaconda 后用 Anaconda Prompt 重来。")
        else:
            print("")
            print("提示：如果报 GBK / UnicodeDecodeError，说明 requirements 文件被编辑器改成了非 UTF-8，")
            print("      请重新从原始交付里复制一份 requirements.txt。")

if __name__ == "__main__":
    main()
