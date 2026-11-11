# -*- coding: utf-8 -*-
"""
Global configuration. All scripts read paths and parameters from here.
This is the only file you need to edit.
"""
from pathlib import Path

# ============================================================
# 1. Directories
# ============================================================
ROOT = Path(__file__).resolve().parent.parent          # Project root (auto-detected: renaming the folder or switching drives has no effect)
DATA_RAW     = ROOT / "data" / "raw"
DATA_CLEAN   = ROOT / "data" / "clean"
DATA_LABEL   = ROOT / "data" / "label"
DATA_FEATURE = ROOT / "data" / "feature"
OUT_TABLE    = ROOT / "outputs" / "tables"
OUT_FIG      = ROOT / "outputs" / "figures"
OUT_LOG      = ROOT / "outputs" / "logs"
DOCS         = ROOT / "docs"

for _d in (DATA_RAW, DATA_CLEAN, DATA_LABEL, DATA_FEATURE,
           OUT_TABLE, OUT_FIG, OUT_LOG, DOCS):
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Research scope
# ============================================================
SITE_EAST = "东栅"
SITE_WEST = "西栅"
SITE_BOTH = "双栅"      # combo-ticket reviews (mention both the East and West Scenic Areas)
SITE_UNK  = "未提及"    # not mentioned in the text, and not available from the page tags either

SITES = [SITE_EAST, SITE_WEST]

# Fallback keywords for when a review mentions "the whole Wuzhen" (without the East/West wording)
UZHEN_WORDS = ["乌镇", "Wuzhen", "wuzhen"]

# ============================================================
# 3. Cleaning parameters
# ============================================================
MIN_LEN = 8            # Minimum characters kept per review (East Scenic Area reviews are short; do not raise this)
MAX_LEN = 3000         # Truncate over-long reviews (usually copy-pasted travel guides)
DUP_SENT_MIN_COUNT = 3 # A sentence repeated >= this many times -> treated as a copy-pasted scenic-area intro and removed
DUP_SENT_MIN_WORDS = 6 # and only if it has at least this many words after stopword removal

# Sentence splitting: first split on strong punctuation; if a resulting sentence is still long, split on commas too
STRONG_PUNCT = "。！？!?；;\n\r"
WEAK_PUNCT   = "，,、 "
MAX_SENT_LEN = 60      # Only keep splitting a sentence on commas when it exceeds 60 characters

# Advertising / traffic-diverting words: the whole review is removed if any is hit
AD_WORDS = ["加微信", "微信号", "加我微信", "VX", "vx", "QQ群", "代购", "私信我",
            "扫码加", "免费领取", "优惠券群", "带货", "推广", "广告位", "请加",
            "☎", "电话咨询", "旅行社报名"]

# ============================================================
# 4. Dictionary of 21 old-town aspect categories (keyword-matching baseline classifier)
#     Rule: if a sentence hits any keyword of a category -> the sentence belongs to that category (multi-label allowed)
#     The dictionary can and should keep growing: look at the "unmatched examples" printed by script 05 and add new words
# ============================================================
CATEGORIES = {
    # ---------- A Landscape and appearance ----------
    "C01_水乡建筑风貌": ["白墙", "黛瓦", "黑瓦", "马头墙", "木雕", "砖雕", "石雕", "古建筑", "老房子",
                    "老宅", "明清", "雕花", "古色古香", "青石板", "石板路", "木结构", "骑楼",
                    "临水而建", "枕水", "建筑"],
    "C02_河道水景":     ["河道", "河水", "摇橹", "乌篷船", "划船", "坐船", "乘船", "游船", "水乡",
                    "水巷", "船夫", "泛舟", "水上", "船娘", "水阁", "雨天水景"],
    "C03_古桥":         ["古桥", "石桥", "逢源双桥", "桥头", "小桥", "拱桥", "定升桥", "桥边"],
    "C04_夜景灯光":     ["夜景", "灯光", "夜色", "亮灯", "灯火", "夜游", "灯影", "亮化", "晚上去",
                    "夜幕", "星空", "水幕", "灯光秀", "夜里"],
    "C05_自然生态":     ["绿化", "柳树", "花草", "荷花", "竹林", "植被", "空气清新", "鸟鸣",
                    "蓝天", "生态", "树荫"],

    # ---------- B Cultural experience ----------
    "C06_历史底蕴":     ["历史", "千年", "文化底蕴", "古镇", "底蕴", "古朴", "岁月", "沧桑",
                    "故事", "人文", "传统", "老底子", "原汁原味"],
    "C07_民俗演艺":     ["皮影", "花鼓戏", "评弹", "戏曲", "庙会", "香市", "戏剧节", "演出",
                    "表演", "民俗", "非遗表演", "拳船", "高竿", "水灯", "巡游", "情景剧",
                    "水上集市", "乌镇戏剧节", "沉浸式"],
    "C08_名人故居展馆": ["茅盾", "故居", "木心", "美术馆", "纪念馆", "博物馆", "展览", "三寸金莲",
                    "百床馆", "江南百床", "酒坊", "当铺", "余榴梁", "钱币馆", "修真观", "戏台"],
    "C09_商业化程度":   ["商业化", "商业气息", "商铺", "店铺", "小商品", "义乌", "商业街", "购物",
                    "同质化", "千篇一律", "原生态", "生活气息", "居民", "本地人", "市井",
                    "过度开发", "太商业"],
    "C10_非遗手工艺":   ["蓝印花布", "染布", "染坊", "三白酒", "酿酒", "手工", "手作", "织布",
                    "竹编", "花灯", "油纸伞", "杭白菊", "姑嫂饼", "工艺品"],

    # ---------- C Infrastructure ----------
    "C11_餐饮美食":     ["小吃", "餐厅", "美食", "味道", "好吃", "难吃", "定胜糕", "羊肉面",
                    "酱鸭", "臭豆腐", "萝卜丝饼", "馄饨", "面馆", "吃饭", "菜品", "早餐",
                    "咖啡馆", "茶室", "饭店", "青团"],
    "C12_住宿民宿":     ["民宿", "客栈", "酒店", "住宿", "房间", "住一晚", "临水房", "河景房",
                    "通安", "枕水度假", "宜园", "住店", "前台"],
    "C13_交通接驳":     ["停车", "停车场", "摆渡", "接驳", "大巴", "公交", "高铁", "打车", "交通",
                    "自驾", "导航", "景区车", "桐乡站", "接驳车", "步行距离"],
    "C14_指示导览":     ["指示牌", "路标", "导览", "地图", "讲解", "标识", "二维码", "导览图",
                    "找不到", "迷路", "导视", "公众号"],
    "C15_门票性价比":   ["门票", "联票", "票价", "价格", "性价比", "太贵", "便宜", "划算", "套票",
                    "免费", "收费", "消费高", "不值", "值不值"],

    # ---------- D Service and management ----------
    "C16_服务态度":     ["工作人员", "服务态度", "讲解员", "热情", "态度好", "态度差", "服务",
                    "员工", "保安", "志愿者", "咨询", "投诉", "贴心"],
    "C17_拥挤客流":     ["人多", "拥挤", "排队", "人流", "扎堆", "清静", "安静", "冷清",
                    "客流", "限流", "摩肩接踵", "太挤", "人山人海", "游客多"],
    "C18_卫生秩序":     ["卫生", "干净", "整洁", "脏", "垃圾", "厕所", "卫生间", "秩序",
                    "乱七八糟", "保洁", "异味"],
    "C19_安全治安":     ["安全", "治安", "放心", "危险", "护栏", "走失", "安保", "应急",
                    "急救", "防火"],

    # ---------- E Emotional experience ----------
    "C20_氛围放松":     ["氛围", "宁静", "惬意", "放松", "舒服", "悠闲", "恬静", "慢生活",
                    "沉浸", "心情", "治愈", "静谧", "安详", "慢下来", "远离喧嚣", "烟火气"],
    "C21_拍照打卡社交": ["拍照", "写真", "汉服", "打卡", "出片", "网红", "亲子", "带娃", "孩子",
                    "朋友", "情侣", "闺蜜", "团建", "摄影", "机位", "必打卡", "遛娃"],
}

# Layer 1: content unrelated to the study that must be removed (matching the original study's "remove place names/weather/time")
IRRELEVANT_WORDS = ["天气", "下雨", "晴天", "阴天", "气温", "几度", "季节", "几月", "周末", "假期",
                    "公司组织", "出差", "路过", "上班", "机场", "高铁站", "酒店预订", "行程安排",
                    "第一天", "第二天", "第三天", "攻略"]

# ============================================================
# 5. Aspect-level sentiment lexicon (local, burns no API quota)
#     Purpose: score each "sentence" so we can average aspect sentiment per category
#     For more rigor you can switch to the Dalian University of Technology sentiment lexicon / HowNet (see 6.4)
# ============================================================
POS_WORDS = {
    "美": 1, "漂亮": 1, "好看": 1, "精致": 1, "震撼": 1.5, "惊艳": 1.5, "值得": 1, "推荐": 1,
    "满意": 1, "喜欢": 1, "开心": 1, "愉快": 1, "舒服": 1, "惬意": 1, "放松": 1, "宁静": 1,
    "干净": 1, "整洁": 1, "方便": 1, "热情": 1, "贴心": 1, "划算": 1, "便宜": 0.8, "丰富": 1,
    "有趣": 1, "精彩": 1, "独特": 1, "古朴": 1, "原汁原味": 1, "有味道": 1, "出片": 1,
    "难忘": 1, "流连忘返": 1.5, "不虚此行": 1.5, "物超所值": 1.5, "惊喜": 1.5, "惊艳到": 1.5,
    "好吃": 1, "美味": 1, "赞": 1, "棒": 1, "很好": 1, "不错": 0.6, "还行": 0.3,
}
NEG_WORDS = {
    "差": 1, "很差": 1.5, "糟糕": 1.5, "难吃": 1.5, "失望": 1.5, "后悔": 1.5, "坑": 1.5,
    "贵": 1, "太贵": 1.5, "不值": 1.5, "宰客": 2, "商业化": 0.8, "同质化": 1, "千篇一律": 1,
    "拥挤": 1, "太挤": 1.5, "排队": 0.8, "脏": 1.5, "乱": 1, "差评": 1.5, "态度差": 1.5,
    "无聊": 1, "没意思": 1, "一般": 0.5, "普通": 0.5, "吵": 1, "闹": 0.8, "假": 1,
    "过度开发": 1.2, "强制消费": 2, "乱收费": 2, "找不到": 0.8, "迷路": 0.8, "累": 0.6,
}
# Degree adverbs: weight the strength of sentiment words
DEGREE_WORDS = {"非常": 1.8, "很": 1.5, "太": 1.7, "特别": 1.7, "超": 1.8, "极其": 2.0,
                "十分": 1.7, "格外": 1.7, "有点": 0.6, "稍微": 0.5, "略": 0.5, "不太": 0.5}
# Negation words: appearing within 3 characters before a sentiment word -> flip the sentiment
NEGATION_WORDS = ["不", "没", "无", "别", "非", "未", "毫无", "谈不上"]

# ============================================================
# 6. Sentiment analysis method
# ============================================================
# "baidu"   = call Baidu AI Cloud NLP (recommended, highest review-level accuracy)
# "snownlp" = local offline (free fallback, slightly lower accuracy)
SENTIMENT_METHOD = "baidu"

# Baidu AI Cloud NLP (apply at https://console.bce.baidu.com/ai/, see 6.1)
BAIDU_API_KEY    = ""     # <- put your API Key here
BAIDU_SECRET_KEY = ""     # <- put your Secret Key here
BAIDU_QPS_SLEEP  = 0.35   # seconds between requests (free tier QPS is about 2; do not set this too small)
BAIDU_MAX_BYTES  = 1000   # per-request text byte limit (API hard limit is 2048 bytes; keep a safety margin)

# ============================================================
# 7. Random forest and statistics
# ============================================================
RANDOM_STATE = 42
TEST_SIZE    = 0.20          # Same as the paper: 80% train / 20% validation
RF_N_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 5
CV_FOLDS = 5

# Sentence-level category decision threshold. The classifier sometimes "over-predicts" (a sentence gets too many categories),
# which shows up in script 09 as a "some category mention rate > 90%" diagnostic warning.
# In that case, raise the threshold to 0.6 or 0.7 and re-run script 09.
CLF_PROB_THRESHOLD = 0.5

# Use "East-only" and "West-only" reviews for the main analysis; "both" reviews are used separately for robustness checks
MAIN_SITES = [SITE_EAST, SITE_WEST]

# ============================================================
# 8. File names (managed in one place to avoid typos)
# ============================================================
F_COMMENTS_RAW   = DATA_RAW / "comments_raw.csv"        # input for 03 cleaning
F_COMMENTS_CLEAN = DATA_CLEAN / "comments_clean.csv"
F_SENTENCES      = DATA_CLEAN / "sentences.csv"
F_SENT_LABEL_RULE= DATA_CLEAN / "sentences_rule_label.csv"
F_CMT_LABEL_RULE = DATA_CLEAN / "comments_rule_label.csv"
F_LABEL_SHEET    = DATA_LABEL / "label_sheet.xlsx"
F_LABEL_DONE     = DATA_LABEL / "label_done.xlsx"
F_CLF_MODEL      = DATA_FEATURE / "clf_model.joblib"
F_SENT_CLF       = DATA_CLEAN / "sentences_clf_label.csv"
F_SENTIMENT      = DATA_CLEAN / "comments_sentiment.csv"
F_SENT_SENTIMENT = DATA_CLEAN / "sentences_sentiment.csv"
F_FEATURES       = DATA_FEATURE / "features_comment.csv"
F_FEATURES_SENT  = DATA_FEATURE / "features_sentence.csv"
