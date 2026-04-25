# 配置

# 本地模型路径（和 llama_gateway.py 一致）
MODEL_PATH = "/home/zhou/shared/model/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"

# 默认人口规模。每条内容评估需要 N 次 LLM 调用，N 越大越稳但越慢。
# 20-30 是实验起点；认真跑建议 50+。
DEFAULT_POPULATION_SIZE = 25

# persona 存储位置（生成一次就复用，保证跨内容的公平性）
PERSONA_FILE = "./data/personas.json"

# 缓存目录：同一 (persona, content) 对只调用 LLM 一次
CACHE_DIR = "./data/cache"

# LLM 参数
N_CTX = 32768           # 上下文长度（persona prompt 很短，不用 65536）
N_GPU_LAYERS = 99
PERSONA_TEMPERATURE = 0.85   # persona 判断：希望个体间有差异
VARIATION_TEMPERATURE = 0.95 # 内容变体生成：高温度以获得多样性
