"""
LLM 后端抽象。两个选项：

- LlamaCppBackend:  直接用 llama_cpp 加载模型（推荐，独占 GPU 最干净）
- SocketBackend:    复用你已经跑着的 llama_gateway.py (端口 10000)
                    注意：gateway 会把 word.txt 拼到你的 prompt 后面，
                    用前把 word.txt 清空。

两者都暴露 .chat(user_prompt, **kwargs) -> str。
"""

import json
import re
import socket
import time


class LlamaCppBackend:
    """直接用 llama_cpp 加载模型。参数与你的 llama_gateway.py 保持一致。"""

    def __init__(self, model_path, n_ctx=32768, n_gpu_layers=99, seed=-1):
        # lazy import，避免仅用 socket 后端时也被迫装 llama_cpp
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            flash_attn=True,
            verbose=False,
            seed=seed,
        )

    def chat(self, user_prompt, max_tokens=1024, temperature=0.85,
             top_p=0.95, top_k=20):
        # Qwen ChatML 模板
        prompt = (
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        out = self.llm(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=0,
            presence_penalty=0,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        return out["choices"][0]["text"].strip()


class SocketBackend:
    """复用 llama_gateway.py。注意 gateway 会 append word.txt，用前请清空。"""

    def __init__(self, host="127.0.0.1", port=10000, timeout=300):
        self.host = host
        self.port = port
        self.timeout = timeout

    def chat(self, user_prompt, **kwargs):
        # 忽略 temperature 等（gateway 是固定参数的）
        s = socket.socket()
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.sendall(user_prompt.encode("utf-8"))
        s.shutdown(socket.SHUT_WR)   # 通知 server 发送结束
        data = b""
        s.settimeout(1.0)
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
            except socket.timeout:
                break
        s.close()
        return data.decode("utf-8", errors="replace").strip()


# ---------- JSON 提取工具（LLM 输出经常带 markdown / 前后缀）----------

def extract_json_object(text):
    """提取文本里第一个合法的 JSON 对象 {...}。失败返回 None。"""
    # 优先处理 markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        # 找最外层花括号
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return None
        candidate = text[start:end + 1]

    # 试一次，失败就清理一下再试
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # 去掉尾逗号
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        # 单引号转双引号（粗暴但有时有效）
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def extract_json_array(text):
    """提取文本里第一个合法的 JSON 数组 [...]。失败返回 None。"""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        candidate = m.group(1)
    else:
        start = text.find("[")
        if start < 0:
            return None
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end < 0:
            return None
        candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None
