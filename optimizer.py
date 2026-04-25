"""
进化式内容优化。

基本循环：
    gen 0:  pool = [seed_contents...]
    repeat G 轮:
        评估 pool 中每条的 R0
        取 top_k 精英
        每个精英用 LLM 扩展 variants_per_parent 条变体
        pool = 精英 + 变体

最终按历代 R0 排序返回。

注意：LLM 生成变体的随机性决定了搜索的多样性。温度给的较高 (0.95)。
"""

from typing import List, Dict

from llm_backend import extract_json_array
from prompts import CONTENT_VARIATION_PROMPT
from simulator import compute_R0
import config


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
    """返回历代所有评估过的内容，按 R0 降序。

    每个元素:
      {
        "generation": int,
        "content": str,
        "strategy": str,   # 如果是变体，记录策略
        "parent": str,     # 如果是变体，记录父代内容的前缀
        "metrics": {...}   # compute_R0 的完整结果，已去掉 responses
      }
    """
    pool = [{"content": c, "strategy": "seed", "parent": None}
            for c in seed_contents]
    history: List[Dict] = []
    seen_contents = set()   # 去重：同一字符串不重复评估

    for gen in range(generations):
        if verbose:
            print(f"\n{'='*60}")
            print(f" 第 {gen} 代  |  池大小 {len(pool)}")
            print('='*60)

        # ---- 评估 ----
        evaluated = []
        for i, item in enumerate(pool):
            content = item["content"]
            if content in seen_contents:
                # 已经评估过，复用
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
            # 保留一份带 responses 的，和一份精简的
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

        # ---- 选 top_k ----
        evaluated.sort(key=lambda x: x[1]["R0"], reverse=True)
        top = evaluated[:top_k]

        if verbose:
            print(f"\n本代 top {len(top)}：")
            for it, m in top:
                print(f"  R0={m['R0']:.3f}  [{it['strategy']}]  "
                      f"{it['content'][:70]}...")

        # ---- 如果是最后一代，停止扩种 ----
        if gen == generations - 1:
            break

        # ---- 扩种 ----
        next_pool = []
        # 精英保留
        for it, _ in top:
            next_pool.append(it)
        # 变体
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

    # 最终排序
    history.sort(key=lambda h: h["metrics"]["R0"], reverse=True)
    # 去掉 metrics_full 减小返回体积
    for h in history:
        h.pop("metrics_full", None)
    return history
