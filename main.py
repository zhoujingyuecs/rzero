"""
命令行入口。用法示例：

    # 1) 初始化人口（只需一次）
    python main.py init-personas

    # 2) 评估一条内容的 R0
    python main.py evaluate --content "某条文字内容..."
    python main.py evaluate --content-file my_post.txt --out result.json

    # 3) 进化优化一条种子内容
    python main.py evolve --seed "某条种子文字..." \
         --generations 3 --variants 3 --top-k 3 --out history.json

    # 若改用 socket 后端（复用你跑着的 llama_gateway.py，记得清空 word.txt）：
    python main.py --backend socket evaluate --content "..."
"""

import argparse
import json
import os
import sys

import config
from llm_backend import LlamaCppBackend, SocketBackend
from personas import generate_population, load_personas, save_personas
from queries import QueryCache
from simulator import compute_R0
from optimizer import evolve_content


def build_backend(args):
    if args.backend == "socket":
        print(f"[backend] socket @ {args.host}:{args.port}")
        print("  注意：llama_gateway.py 会把 word.txt 拼到 prompt 后面，"
              "请确保 word.txt 为空，否则会污染查询！")
        return SocketBackend(host=args.host, port=args.port)
    else:
        print(f"[backend] loading llama_cpp: {args.model_path}")
        return LlamaCppBackend(
            model_path=args.model_path,
            n_ctx=config.N_CTX,
            n_gpu_layers=config.N_GPU_LAYERS,
        )


def get_personas(args):
    if os.path.exists(args.personas):
        pop = load_personas(args.personas)
        print(f"[personas] loaded {len(pop)} from {args.personas}")
        return pop
    pop = generate_population(n=args.n_personas)
    save_personas(pop, args.personas)
    print(f"[personas] generated {len(pop)} → {args.personas}")
    return pop


def cmd_init_personas(args):
    pop = generate_population(n=args.n_personas)
    save_personas(pop, args.personas)
    print(f"\n生成了 {len(pop)} 个 persona，保存到 {args.personas}\n")
    print("-" * 50)
    for p in pop[:5]:
        print(f"\n{p.id}:")
        print(p.describe())
    if len(pop) > 5:
        print(f"\n... 还有 {len(pop) - 5} 个 ...")


def cmd_evaluate(args):
    backend = build_backend(args)
    personas = get_personas(args)
    cache = QueryCache() if args.use_cache else None

    if args.content:
        content = args.content
    elif args.content_file:
        with open(args.content_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        print("从 stdin 读取内容（Ctrl-D 结束）...", file=sys.stderr)
        content = sys.stdin.read()

    if not content.strip():
        print("内容为空，退出。", file=sys.stderr)
        sys.exit(1)

    print(f"\n评估内容（{len(content)} 字）...\n")
    result = compute_R0(backend, personas, content, cache=cache,
                        verbose=not args.quiet)

    print(f"\n{'='*60}")
    print("结果")
    print('='*60)
    summary = {k: v for k, v in result.items() if k != "responses"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n==> R0 = {result['R0']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n完整结果（含每个 persona 原始回答）→ {args.out}")


def cmd_evolve(args):
    backend = build_backend(args)
    personas = get_personas(args)
    cache = QueryCache() if args.use_cache else None

    if args.seed_file:
        with open(args.seed_file, "r", encoding="utf-8") as f:
            # 空行分隔多个种子
            text = f.read()
            seeds = [s.strip() for s in text.split("\n\n") if s.strip()]
    elif args.seed:
        seeds = [args.seed]
    else:
        print("从 stdin 读取种子内容（Ctrl-D 结束）...", file=sys.stderr)
        seeds = [sys.stdin.read()]

    if not seeds or not seeds[0].strip():
        print("种子为空，退出。", file=sys.stderr)
        sys.exit(1)

    print(f"\n开始进化：{len(seeds)} 条种子，"
          f"{args.generations} 代，每代 top-{args.top_k}，"
          f"每条父代 {args.variants} 变体\n")

    history = evolve_content(
        backend, personas, seeds,
        generations=args.generations,
        variants_per_parent=args.variants,
        top_k=args.top_k,
        cache=cache,
        verbose=not args.quiet,
    )

    print(f"\n{'='*60}")
    print(" 历代最佳")
    print('='*60)
    for i, h in enumerate(history[:10]):
        m = h["metrics"]
        print(f"\n#{i+1}  R0={m['R0']:.3f}  (gen {h['generation']})  "
              f"[{h['strategy']}]")
        print(f"    转发率={m['forward_rate']:.2f} | "
              f"情绪强度={m['avg_emotional_intensity']:.2f} | "
              f"触达={m['avg_reach_among_forwarders']:.1f}")
        print(f"    内容：{h['content']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        print(f"\n完整历史 → {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description="内容基本再生数 (R0) 分析与优化系统",
    )
    # 共享参数
    parser.add_argument("--backend", choices=["llama_cpp", "socket"],
                        default="llama_cpp")
    parser.add_argument("--model-path", default=config.MODEL_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--personas", default=config.PERSONA_FILE)
    parser.add_argument("--n-personas", type=int,
                        default=config.DEFAULT_POPULATION_SIZE)
    parser.add_argument("--no-cache", dest="use_cache",
                        action="store_false", default=True)
    parser.add_argument("--quiet", action="store_true")

    subs = parser.add_subparsers(dest="cmd", required=True)

    p_init = subs.add_parser("init-personas", help="生成 / 重置 persona 人口")
    p_init.set_defaults(func=cmd_init_personas)

    p_eval = subs.add_parser("evaluate", help="评估一条内容的 R0")
    src = p_eval.add_mutually_exclusive_group()
    src.add_argument("--content", help="直接给出内容")
    src.add_argument("--content-file", help="从文件读取内容")
    p_eval.add_argument("--out", help="保存完整结果到 JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    p_evo = subs.add_parser("evolve", help="进化优化高 R0 内容")
    src2 = p_evo.add_mutually_exclusive_group()
    src2.add_argument("--seed", help="单条种子")
    src2.add_argument("--seed-file", help="从文件读取种子（空行分隔）")
    p_evo.add_argument("--generations", type=int, default=3)
    p_evo.add_argument("--variants", type=int, default=3)
    p_evo.add_argument("--top-k", type=int, default=3)
    p_evo.add_argument("--out", help="保存历史到 JSON")
    p_evo.set_defaults(func=cmd_evolve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
