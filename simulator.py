"""
R0 计算核心。

模型（解析式蒙特卡洛）：

    R0 = E_{p 是转发者} [ κ_p × β_p ]

其中：
    κ_p = persona p 若转发，预期触达人数（LLM 给出 expected_reach）
    β_p = p 触达的人群中，再次转发的平均概率
        ≈ mean over q in audience(p) of forward_probability(q)
    权重 = forward_probability(p)  （越可能是 forwarder 的人权重越高）

audience(p) 通过 persona 的 audience_tags 和 p 指定的
matched_audiences 做字符串子串匹配得到——粗但稳。匹配不到就退化到整个人口。

关键：这只是一次"解析一阶近似"，不是多代模拟。多代模拟需要 O(K^G) 次
LLM 调用，对 35B MoE 来说不现实；而且对"排序不同内容的传播力"而言，
一阶 R0 足够把相对优劣分出来。
"""

from collections import Counter, defaultdict
from typing import Dict, List

from personas import Persona
from queries import query_persona


def _audience_to_ids(matched_audiences: List[str], by_tag: Dict[str, list],
                     all_ids: List[str], self_id: str) -> List[str]:
    """把 forwarder 说的受众描述，匹配到 persona id 列表。"""
    target_ids = set()
    for aud in matched_audiences:
        aud_s = str(aud).strip()
        if not aud_s:
            continue
        for tag, ids in by_tag.items():
            # 双向子串匹配（宽松）
            if aud_s in tag or tag in aud_s:
                target_ids.update(ids)
    target_ids.discard(self_id)   # 自己不传给自己

    # 完全匹配不到 → 退化为整个人口（除自己）
    if not target_ids:
        target_ids = set(all_ids) - {self_id}

    return list(target_ids)


def compute_R0(backend, personas: List[Persona], content: str,
               cache=None, verbose: bool = True) -> Dict:
    """对一条 content 计算 R0 及一系列诊断指标。"""

    # ---- 1. 人口普查：每个 persona 查一次 ----
    responses = {}
    for idx, p in enumerate(personas):
        r = query_persona(backend, p, content, cache=cache)
        responses[p.id] = r
        if verbose:
            flag = "✓" if r["would_forward"] else " "
            print(
                f"  [{flag}] {idx+1:2d}/{len(personas)} {p.id} "
                f"({p.age_group} / {p.occupation}): "
                f"p={r['forward_probability']:.2f} "
                f"reach={r['expected_reach']:3d} "
                f"emo={r['emotional_intensity']}({r['emotion_type']})"
            )

    # ---- 2. 受众标签倒排索引 ----
    by_tag: Dict[str, list] = defaultdict(list)
    for p in personas:
        for tag in p.audience_tags:
            by_tag[tag].append(p.id)
    all_ids = [p.id for p in personas]

    # ---- 3. 对每个 "会转发" 的 persona 算期望次级转发数 ----
    secondary_counts = []
    weights = []
    target_coverage = Counter()      # 哪些 persona 被多次作为转发目标

    for p in personas:
        r = responses[p.id]
        if not r["would_forward"]:
            continue
        reach = r["expected_reach"]
        if reach <= 0:
            continue

        target_ids = _audience_to_ids(
            r["matched_audiences"], by_tag, all_ids, p.id
        )
        if not target_ids:
            continue

        for tid in target_ids:
            target_coverage[tid] += 1

        target_probs = [responses[tid]["forward_probability"] for tid in target_ids]
        avg_beta = sum(target_probs) / len(target_probs)

        secondary = reach * avg_beta
        secondary_counts.append(secondary)
        weights.append(r["forward_probability"])

    # 加权平均
    if secondary_counts:
        total_w = sum(weights)
        if total_w > 0:
            R0 = sum(s * w for s, w in zip(secondary_counts, weights)) / total_w
        else:
            R0 = sum(secondary_counts) / len(secondary_counts)
    else:
        R0 = 0.0

    # ---- 4. 诊断指标 ----
    n = len(personas)
    forwarders = [p for p in personas if responses[p.id]["would_forward"]]

    forward_rate = len(forwarders) / n if n else 0.0
    avg_emotion = sum(r["emotional_intensity"] for r in responses.values()) / n if n else 0.0
    avg_reach = (
        sum(responses[p.id]["expected_reach"] for p in forwarders) / len(forwarders)
        if forwarders else 0.0
    )

    emotion_dist = Counter(r["emotion_type"] for r in responses.values())

    forwarder_profile = {
        "age_groups": dict(Counter(p.age_group for p in forwarders)),
        "occupations": dict(Counter(p.occupation for p in forwarders)),
        "top_interests": Counter(
            i for p in forwarders for i in p.primary_interests
        ).most_common(5),
        "platforms": dict(Counter(p.main_platform for p in forwarders)),
    }

    # 提取几条代表性理由
    sample_reasons = []
    for p in forwarders[:5]:
        r = responses[p.id]
        sample_reasons.append({
            "persona": f"{p.age_group}/{p.occupation}",
            "reason": r["reason"],
        })
    for p in personas:
        if len(sample_reasons) >= 8:
            break
        if not responses[p.id]["would_forward"]:
            sample_reasons.append({
                "persona": f"{p.age_group}/{p.occupation}",
                "reason": "(不转发) " + responses[p.id]["reason"],
            })

    return {
        "R0": round(R0, 3),
        "forward_rate": round(forward_rate, 3),
        "avg_emotional_intensity": round(avg_emotion, 2),
        "avg_reach_among_forwarders": round(avg_reach, 1),
        "n_forwarders": len(forwarders),
        "n_total": n,
        "emotion_distribution": dict(emotion_dist),
        "forwarder_profile": forwarder_profile,
        "sample_reasons": sample_reasons,
        "responses": responses,   # 完整原始数据，便于后续分析
    }
