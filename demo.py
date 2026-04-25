"""
演示脚本：用几条典型内容对比 R0，验证系统是否按预期工作。

预期结果（如果模型按设计工作）：
- A 条（纯技术干货）: 低 R0，高学历人群会转发
- B 条（情绪化社会话题）: 高 R0，跨人群转发
- C 条（中老年养生谣言）: 中等 R0，集中在中老年
- D 条（平淡日常记录）: 极低 R0

这不是严格的校准，但可以用来快速 sanity check。
"""

import json
import sys

from llm_backend import LlamaCppBackend
from personas import generate_population, save_personas, load_personas
from queries import QueryCache
from simulator import compute_R0
import config
import os


DEMO_CONTENTS = {
    "A_技术干货": (
        "深度讲解：为什么说 Rust 的所有权模型从根本上解决了 C++ 几十年悬而未决"
        "的内存安全问题？本文从编译器角度拆解借用检查器的实现原理，并给出三个"
        "典型 use-after-free 场景的对比代码。"
    ),
    "B_社会情绪": (
        "凌晨三点的急诊室，一个外卖小哥被车撞了，膝盖骨折。他躺在担架上第一句"
        "话不是喊疼，是拼命求医生先帮他把手机里的订单超时取消——他说这一单超"
        "时扣款比他半天的工资还多。转发，让更多人看看这个时代普通人的体面是"
        "怎么被一分一秒榨干的。"
    ),
    "C_养生谣言": (
        "【紧急提醒】医院不会告诉你的秘密：每天早上空腹喝一杯它，连喝七天，"
        "血管里的垃圾全部排出来！已经有上万人见证效果，家里有高血压、糖尿病"
        "老人的一定要看，转给爸妈！"
    ),
    "D_平淡日常": (
        "今天天气不错，中午吃了碗面，加了两个蛋，味道还可以。下午去超市买了"
        "点菜，晚上准备做个西红柿炒蛋。"
    ),
    "E_争议观点": (
        "说句得罪人的话：这一代年轻人不是不努力，是彻底看透了——上一代靠买房"
        "躺赢了三十年，现在反过来教他们'要奋斗不要躺平'，这不是忽悠下一代接"
        "盘吗？凭什么？"
    ),
}


def main():
    print("加载模型...")
    backend = LlamaCppBackend(
        model_path=config.MODEL_PATH,
        n_ctx=config.N_CTX,
        n_gpu_layers=config.N_GPU_LAYERS,
    )

    # 加载或生成人口
    if os.path.exists(config.PERSONA_FILE):
        personas = load_personas(config.PERSONA_FILE)
        print(f"加载 {len(personas)} 个 persona")
    else:
        personas = generate_population(n=config.DEFAULT_POPULATION_SIZE)
        save_personas(personas, config.PERSONA_FILE)
        print(f"生成 {len(personas)} 个 persona")

    cache = QueryCache()

    results = {}
    for name, content in DEMO_CONTENTS.items():
        print(f"\n{'='*60}")
        print(f" 评估 {name}")
        print('='*60)
        print(f"内容：{content}\n")
        metrics = compute_R0(backend, personas, content, cache=cache,
                             verbose=True)
        results[name] = {k: v for k, v in metrics.items() if k != "responses"}
        print(f"\n>>> {name}: R0 = {metrics['R0']}")

    print(f"\n\n{'='*60}")
    print(" DEMO 汇总（按 R0 降序）")
    print('='*60)
    ranked = sorted(results.items(),
                    key=lambda kv: kv[1]["R0"], reverse=True)
    for name, r in ranked:
        print(f"\n{name}")
        print(f"  R0                  = {r['R0']}")
        print(f"  转发率              = {r['forward_rate']}")
        print(f"  平均情绪强度        = {r['avg_emotional_intensity']}")
        print(f"  转发者中平均触达    = {r['avg_reach_among_forwarders']}")
        print(f"  情绪分布            = {r['emotion_distribution']}")

    with open("./data/demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n完整结果已存到 ./data/demo_results.json")


if __name__ == "__main__":
    main()
