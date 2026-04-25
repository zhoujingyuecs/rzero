"""
进化式内容优化。

本文件提供两套进化循环：

1. evolve_content(seed_contents, ...)
   —— 给定一条或几条种子内容，自由变异，目标只看 R0。
      用于主题已经定好、想看看怎么改写更易传播的场景。

2. evolve_for_topic(brief, ...)            ★ 主推
   —— 给定一段"创作要求"（brief，例如 word.txt 中写的需求），
      程序自己生成初始草稿、迭代变异，每一代都对每条内容打两个分：
        (a) R0：传播力（来自 simulator.compute_R0）
        (b) adherence：切题分（0-10，由 LLM 当裁判判断是否符合 brief）
      最终 fitness = R0 × (adherence/10)，并以此排序。
      变异时把上一代真实人群的反馈喂回给 LLM，让它针对性改进。

注意：LLM 生成变体的随机性决定了搜索的多样性。温度给得较高 (0.95)。
"""

import hashlib
from typing import List, Dict

from llm_backend import extract_json_array, extract_json_object
from prompts import (
    CONTENT_VARIATION_PROMPT,
    TOPIC_INITIAL_GENERATION_PROMPT,
    TOPIC_AWARE_VARIATION_PROMPT,
    TOPIC_ADHERENCE_PROMPT,
)
from simulator import compute_R0
import config


# =====================================================================
#  原有：自由进化（无 brief 约束）
# =====================================================================

def generate_variants(backend, seed_content: str, n: int = 3,
                      max_retries: int = 3) -> List[Dict]:
    """让 LLM 基于种子内容生成 n 条变体。"""
    prompt = CONTENT_VARIATION_PROMPT.format(
        seed_content=seed_content, n=n
    )
    for _ in range(max_retries):
        try:
            raw = backend.chat(
                prompt,
                temperature=config.VARIATION_TEMPERATURE,
                max_tokens=4096,
            )
        except Exception:
            continue
        parsed = extract_json_array(raw)
        if not parsed:
            continue
        out = []
        for item in parsed:
            if isinstance(item, dict):
                content = item.get("content", "").strip()
                strategy = item.get("strategy", "").strip()
                if content:
                    out.append({"content": content, "strategy": strategy})
        if out:
            return out
    return []


def evolve_content(backend, personas, seed_contents: List[str],
                   generations: int = 3,
                   variants_per_parent: int = 3,
                   top_k: int = 3,
                   cache=None,
                   verbose: bool = True) -> List[Dict]:
    """返回历代所有评估过的内容，按 R0 降序。"""
    pool = [{"content": c, "strategy": "seed", "parent": None}
            for c in seed_contents]
    history: List[Dict] = []
    seen_contents = set()

    for gen in range(generations):
        if verbose:
            print(f"\n{'='*60}")
            print(f" 第 {gen} 代  |  池大小 {len(pool)}")
            print('='*60)

        evaluated = []
        for i, item in enumerate(pool):
            content = item["content"]
            if content in seen_contents:
                prev = next((h for h in history if h["content"] == content), None)
                if prev:
                    evaluated.append((item, prev["metrics_full"]))
                continue
            seen_contents.add(content)

            if verbose:
                print(f"\n-- 评估 {i+1}/{len(pool)} --")
                print(f"策略：{item['strategy']}")
                print(f"内容：{content[:120]}"
                      f"{'...' if len(content) > 120 else ''}")

            metrics = compute_R0(backend, personas, content,
                                 cache=cache, verbose=verbose)
            metrics_full = metrics
            metrics_compact = {k: v for k, v in metrics.items() if k != "responses"}

            history_entry = {
                "generation": gen,
                "content": content,
                "strategy": item["strategy"],
                "parent": item["parent"],
                "metrics": metrics_compact,
                "metrics_full": metrics_full,
            }
            history.append(history_entry)
            evaluated.append((item, metrics_full))

            if verbose:
                print(f"→ R0 = {metrics['R0']} | "
                      f"转发率 = {metrics['forward_rate']} | "
                      f"情绪 = {metrics['avg_emotional_intensity']}")

        evaluated.sort(key=lambda x: x[1]["R0"], reverse=True)
        top = evaluated[:top_k]

        if verbose:
            print(f"\n本代 top {len(top)}：")
            for it, m in top:
                print(f"  R0={m['R0']:.3f}  [{it['strategy']}]  "
                      f"{it['content'][:70]}...")

        if gen == generations - 1:
            break

        next_pool = []
        for it, _ in top:
            next_pool.append(it)
        for it, _ in top:
            if verbose:
                print(f"\n生成变体 (基于 R0={_['R0']:.3f} 的内容)")
            variants = generate_variants(backend, it["content"],
                                         n=variants_per_parent)
            parent_prefix = it["content"][:40]
            for v in variants:
                if v["content"] in seen_contents:
                    continue
                next_pool.append({
                    "content": v["content"],
                    "strategy": v.get("strategy", ""),
                    "parent": parent_prefix,
                })

        pool = next_pool
        if not pool:
            if verbose:
                print("池已空，终止。")
            break

    history.sort(key=lambda h: h["metrics"]["R0"], reverse=True)
    for h in history:
        h.pop("metrics_full", None)
    return history


# =====================================================================
#  新增：基于 brief 的主题感知进化
# =====================================================================

def generate_initial_drafts(backend, brief: str, n: int = 5,
                            max_retries: int = 3) -> List[Dict]:
    """根据 brief 生成 n 条初始草稿（gen 0 的池）。"""
    prompt = TOPIC_INITIAL_GENERATION_PROMPT.format(brief=brief, n=n)
    for _ in range(max_retries):
        try:
            raw = backend.chat(prompt,
                               temperature=config.VARIATION_TEMPERATURE,
                               max_tokens=4096)
        except Exception:
            continue
        parsed = extract_json_array(raw)
        if not parsed:
            continue
        out = []
        for item in parsed:
            if isinstance(item, dict):
                content = item.get("content", "").strip()
                strategy = item.get("strategy", "").strip()
                if content:
                    out.append({"content": content, "strategy": strategy})
        if out:
            return out
    return []


def score_topic_adherence(backend, brief: str, content: str,
                          cache=None, max_retries: int = 3) -> Dict:
    """让 LLM 当裁判：内容是否切合 brief。返回
    {'adherence_score': 0..10, 'reason': str}。"""
    # 缓存 key：用 brief 的 hash + content
    cache_pseudo_id = "_adh_" + hashlib.md5(
        brief.encode("utf-8")).hexdigest()[:12]

    if cache is not None:
        cached = cache.get(cache_pseudo_id, content)
        if cached is not None:
            return cached

    prompt = TOPIC_ADHERENCE_PROMPT.format(brief=brief, content=content)
    for _ in range(max_retries):
        try:
            raw = backend.chat(prompt,
                               temperature=config.ADHERENCE_TEMPERATURE,
                               max_tokens=256)
        except Exception:
            continue
        parsed = extract_json_object(raw)
        if not parsed or "adherence_score" not in parsed:
            continue
        try:
            score = max(0, min(10, int(parsed["adherence_score"])))
        except (TypeError, ValueError):
            continue
        result = {
            "adherence_score": score,
            "reason": str(parsed.get("reason", "")).strip(),
        }
        if cache is not None:
            cache.set(cache_pseudo_id, content, result)
        return result

    # 解析失败：取中位 5 分（不奖也不惩）
    return {"adherence_score": 5,
            "reason": "[adherence judge parse failed, defaulting to 5]"}


def _format_feedback_block(metrics: Dict) -> str:
    """把上一代的 R0 评估结果格式化成给下一代变异 prompt 用的反馈段。

    包含两类信息：
      - 数值指标（R0、转发率、情绪强度、情绪分布）
      - 转发者 / 不转发者的代表性理由（最有信号的部分）
    """
    lines = []
    lines.append(f"- R0 = {metrics['R0']:.2f}")
    n_fwd = metrics.get("n_forwarders", 0)
    n_total = metrics.get("n_total", 0)
    fwd_rate = metrics.get("forward_rate", 0.0)
    lines.append(f"- 转发率 = {fwd_rate*100:.0f}% ({n_fwd}/{n_total})")
    lines.append(f"- 平均情绪强度 = {metrics.get('avg_emotional_intensity', 0):.1f}/10")

    emo = metrics.get("emotion_distribution", {})
    if emo:
        emo_str = ", ".join(
            f"{k}({v})"
            for k, v in sorted(emo.items(), key=lambda x: -x[1])
        )
        lines.append(f"- 情绪分布: {emo_str}")

    reasons = metrics.get("sample_reasons", [])
    fwd_reasons = [r for r in reasons
                   if not r["reason"].startswith("(不转发)")]
    nfwd_reasons = [r for r in reasons
                    if r["reason"].startswith("(不转发)")]

    if fwd_reasons:
        lines.append("\n转发者的真实理由（这条为什么被转）:")
        for r in fwd_reasons[:4]:
            lines.append(f"  - {r['persona']}: {r['reason']}")

    if nfwd_reasons:
        lines.append("\n不转发者的真实理由（这条为什么没被转）:")
        for r in nfwd_reasons[:4]:
            text = r["reason"].replace("(不转发) ", "")
            lines.append(f"  - {r['persona']}: {text}")

    return "\n".join(lines)


def generate_topic_aware_variants(backend, brief: str, parent_content: str,
                                  parent_metrics: Dict, n: int = 3,
                                  max_retries: int = 3) -> List[Dict]:
    """在 brief 约束下，基于父代内容 + 真实反馈生成 n 条变体。"""
    feedback = _format_feedback_block(parent_metrics)
    prompt = TOPIC_AWARE_VARIATION_PROMPT.format(
        brief=brief,
        seed_content=parent_content,
        feedback_block=feedback,
        n=n,
    )
    for _ in range(max_retries):
        try:
            raw = backend.chat(prompt,
                               temperature=config.VARIATION_TEMPERATURE,
                               max_tokens=4096)
        except Exception:
            continue
        parsed = extract_json_array(raw)
        if not parsed:
            continue
        out = []
        for item in parsed:
            if isinstance(item, dict):
                content = item.get("content", "").strip()
                strategy = item.get("strategy", "").strip()
                if content:
                    out.append({"content": content, "strategy": strategy})
        if out:
            return out
    return []


def evolve_for_topic(backend, personas, brief: str,
                    n_initial: int = 5,
                    generations: int = 3,
                    variants_per_parent: int = 3,
                    top_k: int = 3,
                    min_adherence: int = 5,
                    cache=None,
                    verbose: bool = True) -> List[Dict]:
    """主题感知的进化优化。

    流程：
      gen 0:
        用 LLM 根据 brief 生成 n_initial 条初始草稿
      每一代：
        对池中每条新内容：
          1. 算切题分（adherence, 0-10）
          2. 切题分 < min_adherence 直接淘汰，fitness=0，不算 R0
          3. 否则算 R0
          4. fitness = R0 × (adherence/10)
        按 fitness 取 top_k 精英
        每个精英用 LLM + 真实反馈生成 variants_per_parent 条变体
        下一代池 = 精英 + 变体

    返回历代所有评估过的内容，按 fitness 降序。
    """
    if verbose:
        print(f"\n[gen 0] 根据 brief 生成 {n_initial} 条初始草稿...")
    initial = generate_initial_drafts(backend, brief, n=n_initial)
    if not initial:
        raise RuntimeError(
            "无法生成初始草稿。请检查模型是否正常加载，以及 brief 是否完整。")

    pool = [
        {"content": d["content"], "strategy": d["strategy"], "parent": None}
        for d in initial
    ]

    history: List[Dict] = []
    seen: Dict[str, Dict] = {}   # content -> history entry

    for gen in range(generations):
        if verbose:
            print(f"\n{'='*60}")
            print(f" 第 {gen} 代  |  待评估 {len(pool)} 条")
            print('='*60)

        evaluated = []   # list of (item, fitness, metrics, adherence)

        for i, item in enumerate(pool):
            content = item["content"]
            if content in seen:
                prev = seen[content]
                evaluated.append((item, prev["fitness"],
                                  prev["metrics_full"], prev["adherence"]))
                continue

            if verbose:
                print(f"\n-- {i+1}/{len(pool)} --")
                print(f"策略: {item['strategy']}")
                snippet = content[:120] + ("..." if len(content) > 120 else "")
                print(f"内容: {snippet}")

            # 1) 切题分
            adh = score_topic_adherence(backend, brief, content, cache=cache)
            adh_score = adh["adherence_score"]
            if verbose:
                print(f"切题分: {adh_score}/10  ({adh['reason']})")

            # 2) 不切题就跳过 R0 评估，省钱
            if adh_score < min_adherence:
                if verbose:
                    print(f"  ✗ 切题分 < {min_adherence}，跳过 R0 评估")
                metrics = {
                    "R0": 0.0, "forward_rate": 0.0,
                    "avg_emotional_intensity": 0.0,
                    "avg_reach_among_forwarders": 0.0,
                    "n_forwarders": 0, "n_total": len(personas),
                    "emotion_distribution": {}, "forwarder_profile": {},
                    "sample_reasons": [],
                }
                fitness = 0.0
            else:
                metrics = compute_R0(backend, personas, content,
                                     cache=cache, verbose=verbose)
                fitness = metrics["R0"] * (adh_score / 10.0)

            metrics_full = metrics
            metrics_compact = {k: v for k, v in metrics.items() if k != "responses"}

            entry = {
                "generation": gen,
                "content": content,
                "strategy": item["strategy"],
                "parent": item["parent"],
                "adherence": adh_score,
                "adherence_reason": adh["reason"],
                "fitness": round(fitness, 3),
                "metrics": metrics_compact,
                "metrics_full": metrics_full,
            }
            history.append(entry)
            seen[content] = entry
            evaluated.append((item, fitness, metrics_full, adh_score))

            if verbose:
                print(f"→ R0={metrics['R0']:.3f}  切题={adh_score}/10  "
                      f"fitness={fitness:.3f}")

        # 取 top_k
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top = evaluated[:top_k]

        if verbose:
            print(f"\n本代 top {len(top)}:")
            for it, fit, m, adh in top:
                print(f"  fitness={fit:.3f}  R0={m['R0']:.3f}  "
                      f"切题={adh}/10  [{it['strategy']}]")
                snippet = it['content'][:80] + ("..." if len(it['content']) > 80 else "")
                print(f"    {snippet}")

        if gen == generations - 1:
            break

        # 扩种
        next_pool = []
        for it, _, _, _ in top:
            next_pool.append(it)   # 精英保留

        for it, fit, m, adh in top:
            if fit <= 0:
                continue   # 切题不过关的不衍生
            if verbose:
                print(f"\n衍生变体 (基于 fitness={fit:.3f} 的内容，"
                      f"用真实反馈作为提示)")
            variants = generate_topic_aware_variants(
                backend, brief, it["content"], m,
                n=variants_per_parent,
            )
            parent_prefix = it["content"][:40]
            for v in variants:
                if v["content"] in seen:
                    continue
                next_pool.append({
                    "content": v["content"],
                    "strategy": v.get("strategy", ""),
                    "parent": parent_prefix,
                })

        if not next_pool:
            if verbose:
                print("池已空，终止。")
            break
        pool = next_pool

    # 最终排序
    history.sort(key=lambda h: h["fitness"], reverse=True)
    for h in history:
        h.pop("metrics_full", None)
    return history
