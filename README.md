# 内容基本再生数 (R0) 分析与优化系统

把流行病学的 SIR 思想搬到网络内容上：每条内容都有一个**基本再生数 R0**——
平均一个看到内容的人，会让多少新的人也看到。R0 > 1 内容会扩散，R0 < 1 内容会消失。

数据来源不是真实社交平台抓取，而是用本地大模型（Qwen3.5-35B）扮演一群结构化的
合成人物 (persona)，逐个询问"你看到这条会不会转发"，再用这些响应做蒙特卡洛估计。

---

## 一、理论模型

对一条内容 c，基本再生数定义为：

```
R0(c) = E_{p 是转发者} [ κ_p(c) × β_p(c) ]
```

其中：

- **κ_p(c)**：persona p 若转发，预计触达多少人（LLM 直接报告 `expected_reach`）
- **β_p(c)**：被 p 触达的人群中，再次转发的平均概率
  - 通过 p 提供的 `matched_audiences` 标签匹配人口子集得到
  - β_p ≈ mean({forward_probability(q) : q ∈ audience(p)})
- **权重**：每个 p 的贡献按 `forward_probability(p)` 加权，越像 forwarder 的人权重越高

直觉上：一条内容传播力强 = "它能打动一些人转发 (κ × β 里的 β)"
+ "这些转发者每人能触达不少人 (κ)"
+ "他们触达的下一批人也会继续转 (audience 里的 β)"。

### 为什么不做多代模拟？

严格的 SIR/分支过程模拟需要 G 代 × 每代分支 K，LLM 调用数是 O(K^G)。
对 35B 模型来说，G=5、K=20 就是 320 万次调用——纯属灾难。

我们做的是**一次普查 + 解析估计**：每条内容 N 次 LLM 调用（N = 人口规模）。
牺牲的是 R0 数值的绝对准确度，保留的是**对不同内容的相对排序能力**——
而对优化（找高 R0 内容）来说，相对排序才是关键。

### 还报告了什么诊断指标

- 转发率（多少 persona 选择转发）
- 平均情绪强度（0-10）
- 情绪类型分布（愤怒/共鸣/好奇/无感 ...）
- 转发者画像（年龄段、职业、兴趣、平台分布）—— 即"易感人群"
- 每人的疲劳天数 `fatigue_days` —— 对应你提到的"看过一阵不想看"

---

## 二、文件结构

```
content_r0/
  config.py        # 模型路径、人口规模、温度等
  llm_backend.py   # LlamaCppBackend / SocketBackend + JSON 提取
  personas.py      # 分层采样合成人口
  prompts.py       # 所有 LLM prompt 模板
  queries.py       # 单 persona 单内容查询 + 磁盘缓存
  simulator.py     # R0 计算核心
  optimizer.py     # 进化式优化
  main.py          # CLI 入口
  demo.py          # 演示脚本（5 条典型内容对比）
  data/            # 自动生成：personas.json + cache/
```

---

## 三、用法

### 0. 安装依赖

```bash
pip install llama-cpp-python   # 需要 CUDA 编译，参考官方文档
```

如果不想再装一份 llama_cpp，可以让本系统通过 socket 复用你已有的
`llama_gateway.py`（**注意要先清空 `word.txt`**，否则 gateway 会把额外文本拼到 prompt 后面污染查询）。

### 1. 初始化人口（仅首次）

```bash
cd content_r0
python main.py init-personas --n-personas 25
```

人口随机种子固定 (=42)，保证不同内容评估时面对同一群人，R0 数值可比。

### 2. 评估单条内容

```bash
python main.py evaluate --content "你的内容文本..." --out result.json
```

输出会列出每个 persona 的判断、最终 R0、转发者画像。

### 3. 进化优化

```bash
python main.py evolve \
    --seed "你的种子内容" \
    --generations 3 \
    --variants 3 \
    --top-k 3 \
    --out history.json
```

每代会评估池中所有内容，选 top-k 让 LLM 生成变体，迭代 G 代。

### 4. Demo

```bash
python demo.py
```

跑 5 条手工设计的内容（技术干货 / 社会情绪 / 养生谣言 / 平淡日常 / 争议观点），
快速验证系统是否给出符合直觉的相对排序。

### 5. 用 socket 后端

```bash
# 先确保 word.txt 为空，然后启动 llama_gateway.py
python main.py --backend socket evaluate --content "..."
```

---

## 四、性能与时间预算

在 RTX 3090 + Qwen3.5-35B-A3B Q4_K_M 上，单次 persona 查询大概 1-3 秒
（取决于上下文长度和输出 token 数）。

- 一次内容评估（25 persona）：约 0.5-1.5 分钟
- 3 代 × (3 精英 + 9 变体) ≈ 30 次评估：约 30-60 分钟
- demo.py 5 条内容 ≈ 3-7 分钟

加了磁盘缓存：同一 (persona, content) 对永远只调用一次 LLM。
反复调试同一条内容、或者父代精英在下一代继续被评估时，都会命中缓存。

---

## 五、参数怎么调

**人口规模 `n_personas`**（最关键）
- 起步 20-25，做相对排序够用
- 50+ 才能让 R0 数值本身相对稳定（蒙特卡洛误差 ~ 1/√N）
- 100+ 会很慢，但群体细分研究需要

**`PERSONA_TEMPERATURE`**
- 默认 0.85。太低 → 所有 persona 给出趋同的"中庸"判断，区分度变差
- 太高 → 同一个 persona 对同一内容多次回答会很不稳定

**`VARIATION_TEMPERATURE`**
- 默认 0.95，故意高，是为了让进化搜索有探索能力

**`top_k` / `variants`**
- 经典权衡：top_k 大、variants 小 → 探索宽
- top_k 小、variants 大 → 精英深挖

---

## 六、这套东西的局限（必须坦白）

**1. LLM 模拟人 ≠ 真人。** Qwen 扮演的"中老年阿姨"和真实的中老年阿姨在
转发某条内容时的判断可能差很远。**绝对 R0 没有校准价值，相对排序才有。**
要校准，需要拿一批真实平台的转发数据回归一下。

**2. 受众匹配是字符串子串。** 简单粗暴，可能错配。生产用应该改成 embedding 匹配
（比如用 BGE-small-zh 给 audience_tags 和 matched_audiences 算 cosine）。

**3. 一阶近似。** 多代衰减、平台算法分发偏好、舆论反扑，都没建模。

**4. LLM 自洽性偏差。** 当 LLM 既扮演 persona、又生成内容变体时，进化容易
陷入 LLM 自己的偏好（"我觉得好的我也觉得别人觉得好"）。**要破这个，需要至少
两个不同的模型分别担任 persona 和创作者**——但你只有一台 3090，先这样。

**5. 价值观风险。** 系统会学到"愤怒/对立/恐惧/猎奇"内容 R0 更高——这正是
真实平台算法的内核。如果你拿生成结果去发布，你就在亲手生产带毒内容。
**强烈建议这套东西用于研究和防御**（识别哪类内容容易病毒式传播 →
平台/读者怎么应对），而不是用来制造爆款。

**6. 安全护栏。** 我没在 prompt 里加内容过滤——你用的是 Uncensored 版模型。
如果你打算把这套东西扩展成可发布的内容生成器，至少加一层毒性/事实性过滤。

---

## 七、可以怎么扩展

- **真实数据校准**：抓一批微博/小红书的转发计数，对比模型 R0，做线性回归校准。
- **多代精确模拟**：用更小的本地模型（7B 量级）做下游 persona，省调用成本，
  跑真正的分支过程到 R0 收敛或断流。
- **图结构人口**：现在 persona 之间通过 audience_tags 字符串匹配，可以改成
  显式社交图，用图传播算法（如 IC/LT 模型）跑扩散。
- **疲劳 / 群体免疫**：`fatigue_days` 已经收集了，可以加一个时间维度的衰减项，
  研究"同主题内容连续推送 N 天后 R0 怎么衰减"。
- **多模态**：把 content 扩展为带图片/视频描述的复合输入，prompt 里让 persona
  也对视觉元素表态。

## 八、代码库地址

https://gitee.com/shadoubaoo/rzero  
https://github.com/zhoujingyuecs/rzero  

