# 配置

# 本地模型路径（这台机器独占给本程序，启动后基本占满显存）
MODEL_PATH = "/home/zhou/shared/model/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# 默认人口规模。每条内容评估需要 N 次 LLM 调用，N 越大越稳但越慢。
# 20-30 是实验起点；认真跑建议 50+。
DEFAULT_POPULATION_SIZE = 25

# persona 存储位置（生成一次就复用，保证跨内容的公平性）
PERSONA_FILE = "./data/personas.json"

# 缓存目录：同一 (persona, content) 对只调用 LLM 一次
CACHE_DIR = "./data/cache"

# 创作要求文件（generate 命令的默认输入）
# 在文件中写入你想生成什么内容的描述（"brief"），程序会基于它迭代出
# 切题 + 高传播力的版本。
WORD_FILE = "./word.txt"

# LLM 参数
N_CTX = 32768                   # 上下文长度（persona prompt 很短，不用 65536）
N_GPU_LAYERS = 99
PERSONA_TEMPERATURE = 0.85      # persona 判断：希望个体间有差异
VARIATION_TEMPERATURE = 0.95    # 内容变体生成：高温度以获得多样性
ADHERENCE_TEMPERATURE = 0.3     # 切题分判断：要稳定，温度低

# generate 命令的默认参数
GENERATE_N_INITIAL = 5          # 初始草稿数（gen 0 的池大小）
GENERATE_GENERATIONS = 3        # 进化代数
GENERATE_VARIANTS = 3           # 每个精英衍生变体数
GENERATE_TOP_K = 3              # 每代保留精英数
GENERATE_MIN_ADHERENCE = 5      # 切题分阈值（0-10），低于此值的内容直接淘汰，
                                # 不浪费 LLM 调用去算它的 R0
