"""
合成人口 (persona population)。

设计原则：
- 属性从几个基础维度组合而来（年龄 × 职业 × 兴趣 × 性格 × 平台 × 地域），
  用固定随机种子采样，保证可复现。
- 每个 persona 派生一组 `audience_tags`（年龄、职业、兴趣、地域都算标签），
  用于在 R0 计算时做"受众匹配"（某个 forwarder 说"我会转给育儿妈妈"时，
  我们就拿所有 tag 里含"育儿"或职业=家庭主妇的 persona 算他们的 β）。
- 这个设计不是让 LLM 自己产 persona，而是手工分层采样。好处：
    1) 人口结构可控（不会所有人都是"爱刷抖音的年轻人"）
    2) 零 LLM 成本
    3) 跑不同内容时用同一批 persona，R0 数值可比较
"""

from dataclasses import asdict, dataclass, field
from typing import List
import json
import os
import random


AGE_GROUPS = [
    "青少年(13-17)", "大学生(18-22)", "青年(23-30)",
    "中青年(31-40)", "中年(41-55)", "中老年(56-65)", "老年(66+)",
]

OCCUPATIONS = [
    "学生", "互联网工程师", "教师", "医生/护士",
    "内容创作者/自由职业", "蓝领/工厂工人", "公务员/事业单位",
    "销售/市场", "金融从业者", "个体户/小老板",
    "家庭主妇/主夫", "退休人员", "服务业员工", "农民/农村居民",
]

INTERESTS = [
    "科技数码", "时事政治", "体育赛事", "娱乐八卦", "育儿亲子",
    "美食烹饪", "旅游摄影", "游戏电竞", "二次元", "健身养生",
    "理财投资", "星座命理", "宠物", "情感故事", "国际新闻",
    "军事历史", "汽车", "职场", "购物消费", "读书学习",
    "房产", "教育", "医疗健康", "传统文化",
]

PERSONALITIES = [
    "情绪化、易共情",
    "理性冷静、讲逻辑",
    "好奇心强、追新",
    "愤世嫉俗、爱吐槽",
    "保守谨慎、怕被骗",
    "热情、爱分享爱表达",
    "内向安静、潜水型",
    "追热点、易从众",
    "怀疑论者、爱辟谣",
]

PLATFORMS = [
    "微信朋友圈+公众号",
    "微博",
    "抖音/快手",
    "小红书",
    "知乎",
    "B站",
    "QQ群/微信群为主",
    "贴吧",
    "虎扑/懂球帝",
    "X/Twitter + 油管",
]

EDUCATIONS = ["初中及以下", "高中/中专", "大专", "本科", "硕士及以上"]

REGIONS = [
    "一线城市(北上广深)", "新一线", "二线城市",
    "三四线城市", "县城", "农村",
]


@dataclass
class Persona:
    id: str
    age_group: str
    occupation: str
    primary_interests: List[str]
    personality: str
    main_platform: str
    education: str
    region: str
    audience_tags: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """自然语言描述，喂给 LLM 用。"""
        return (
            f"- 年龄段：{self.age_group}\n"
            f"- 职业：{self.occupation}\n"
            f"- 所在地类型：{self.region}\n"
            f"- 学历：{self.education}\n"
            f"- 主要兴趣：{'、'.join(self.primary_interests)}\n"
            f"- 性格：{self.personality}\n"
            f"- 最常使用的平台：{self.main_platform}"
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def _derive_audience_tags(p: Persona) -> List[str]:
    """从 persona 属性派生受众标签。用于 forwarder 指明目标受众时的匹配。"""
    tags = []
    tags.append(p.age_group)
    tags.append(p.occupation)
    tags.extend(p.primary_interests)
    tags.append(p.region)
    tags.append(p.education)
    # 加一些"粗粒度"标签方便匹配
    if "青少年" in p.age_group or "大学生" in p.age_group:
        tags.append("年轻人")
    if "中年" in p.age_group or "老年" in p.age_group:
        tags.append("长辈")
    if "主妇" in p.occupation:
        tags.append("宝妈")
    if "育儿亲子" in p.primary_interests:
        tags.append("宝爸宝妈")
    if "退休" in p.occupation or "老年" in p.age_group:
        tags.append("银发族")
    return tags


def generate_population(n: int = 25, seed: int = 42) -> List[Persona]:
    """分层采样生成 n 个 persona。"""
    rng = random.Random(seed)
    personas = []
    for i in range(n):
        age = rng.choice(AGE_GROUPS)
        # 年龄-职业相容性做点简单约束
        if "青少年" in age:
            occ = "学生"
        elif "大学生" in age:
            occ = rng.choice(["学生", "学生", "内容创作者/自由职业"])
        elif "老年" in age:
            occ = rng.choice(["退休人员", "退休人员", "农民/农村居民"])
        else:
            occ = rng.choice([o for o in OCCUPATIONS
                              if o not in ("学生", "退休人员")])

        p = Persona(
            id=f"P{i:03d}",
            age_group=age,
            occupation=occ,
            primary_interests=rng.sample(INTERESTS, k=rng.randint(2, 4)),
            personality=rng.choice(PERSONALITIES),
            main_platform=rng.choice(PLATFORMS),
            education=rng.choice(EDUCATIONS),
            region=rng.choice(REGIONS),
        )
        p.audience_tags = _derive_audience_tags(p)
        personas.append(p)
    return personas


def save_personas(personas, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in personas], f,
                  ensure_ascii=False, indent=2)


def load_personas(path) -> List[Persona]:
    with open(path, "r", encoding="utf-8") as f:
        return [Persona.from_dict(d) for d in json.load(f)]
