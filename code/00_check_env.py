# -*- coding: utf-8 -*-
"""Environment self-check: once this script runs, the environment is fine."""
import sys
import platform

def check(name, optional=False):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK]   {name:<14} {ver}")
        return True
    except Exception as e:
        flag = "[optional]" if optional else "[missing]"
        print(f"  {flag} {name:<14} {type(e).__name__}: {e}")
        return optional

def main():
    print("=" * 60)
    print("Wuzhen 双栅 study - environment self-check")
    print("=" * 60)
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Operating system : {platform.system()} {platform.release()}")
    print(f"Python path      : {sys.executable}")
    if sys.version_info < (3, 9):
        print("!" * 62)
        print("!! SEVERE WARNING: your Python version is below 3.9, so this project's core libraries cannot be installed.")
        print("!! Python 3.7 reached end of life in June 2023. Pick one of these two options:")
        print("!!   [recommended] Install Anaconda (it ships Python 3.12 and every scientific library)")
        print("!!           https://www.anaconda.com/download")
        print("!!           After installing, open 'Start Menu → Anaconda Prompt' and cd into the project folder")
        print("!!   [alternative] If you only want a quick try: pip install -r requirements-py37.txt")
        print("!" * 62)
    print("-" * 60)

    core = ["pandas", "numpy", "sklearn", "jieba", "matplotlib"]
    opt = ["seaborn", "openpyxl", "requests", "tqdm", "shap",
           "statsmodels", "joblib", "snownlp"]

    ok = True
    print("Core libraries:")
    for m in core:
        ok &= check(m)
    print("Recommended/optional libraries:")
    for m in opt:
        check(m, optional=True)

    print("-" * 60)
    # Chinese font check (mandatory on Windows, otherwise every chart shows empty boxes)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]:
            if cand in names:
                print(f"  [OK]   Chinese font available: {cand}")
                break
        else:
            print("  [WARNING] no common Chinese font found, Chinese text in charts will show as boxes")
    except Exception as e:
        print(f"  [WARNING] font check failed: {e}")

    print("-" * 60)
    # Directory check
    try:
        from config import (DATA_RAW, DATA_CLEAN, DATA_LABEL, DATA_FEATURE,
                            OUT_TABLE, OUT_FIG, OUT_LOG)
        for d in (DATA_RAW, DATA_CLEAN, DATA_LABEL, DATA_FEATURE, OUT_TABLE, OUT_FIG, OUT_LOG):
            print(f"  [OK]   directory: {d}")
    except Exception as e:
        print(f"  [ERROR] cannot import config.py; make sure it is in the same folder as this script: {e}")
        ok = False

    print("=" * 60)
    if ok:
        print("Environment check passed; you can move on to the next step (stage 3, data collection).")
    else:
        old = sys.version_info < (3, 9)
        req = "requirements-py37.txt" if old else "requirements.txt"
        print("Some core libraries are missing, please run:")
        print(f"  python -m pip install -r {req} -i https://pypi.tuna.tsinghua.edu.cn/simple")
        if old:
            print("")
            print("Note: on Python 3.7 most of these libraries will fail to install (scikit-learn / shap especially).")
            print("      The easier route is to install Anaconda and start over from Anaconda Prompt.")
        else:
            print("")
            print("Note: if you see a GBK / UnicodeDecodeError, the requirements file was re-saved as non-UTF-8,")
            print("      so please copy requirements.txt again from the original delivery.")

if __name__ == "__main__":
    main()
