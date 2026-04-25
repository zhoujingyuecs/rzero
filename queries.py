"""
向某个 persona 询问对某条内容的反应，返回结构化 dict。
带磁盘缓存：同一 (persona_id, content) 对只调用一次 LLM。
"""

import hashlib
import json
import os

from llm_backend import extract_json_object
from prompts import PERSONA_QUERY_PROMPT
import config


class QueryCache:
    """简单磁盘缓存。key = hash(persona_id + content)。"""

    def __init__(self, cache_dir=config.CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, persona_id, content):
        h = hashlib.md5(
            (persona_id + "\x1f" + content).encode("utf-8")
        ).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, persona_id, content):
        p = self._path(persona_id, content)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, persona_id, content, response):
        p = self._path(persona_id, content)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)


_FALLBACK = {
    "would_forward": False,
    "forward_probability": 0.0,
    "emotional_intensity": 0,
    "emotion_type": "无感",
    "matched_audiences": [],
    "expected_reach": 0,
    "fatigue_days": 0,
    "reason": "[parse failed]",
}


def _sanitize(parsed):
    """补全缺失字段 + 合法化数值范围。"""
    r = dict(_FALLBACK)
    r.update({k: v for k, v in parsed.items() if k in _FALLBACK})

    # 类型/范围兜底
    r["would_forward"] = bool(r["would_forward"])
    try:
        r["forward_probability"] = max(0.0, min(1.0, float(r["forward_probability"])))
    except (TypeError, ValueError):
        r["forward_probability"] = 1.0 if r["would_forward"] else 0.0
    try:
        r["emotional_intensity"] = max(0, min(10, int(r["emotional_intensity"])))
    except (TypeError, ValueError):
        r["emotional_intensity"] = 0
    try:
        r["expected_reach"] = max(0, min(500, int(r["expected_reach"])))  # 上限 500 防爆
    except (TypeError, ValueError):
        r["expected_reach"] = 10 if r["would_forward"] else 0
    try:
        r["fatigue_days"] = max(0, min(90, int(r["fatigue_days"])))
    except (TypeError, ValueError):
        r["fatigue_days"] = 3

    if not isinstance(r["matched_audiences"], list):
        r["matched_audiences"] = []
    r["matched_audiences"] = [str(x) for x in r["matched_audiences"]][:5]

    # 逻辑一致性：如果说不转发，概率就不该很高
    if not r["would_forward"]:
        r["forward_probability"] = min(r["forward_probability"], 0.2)

    return r


def query_persona(backend, persona, content, cache=None, max_retries=3,
                  temperature=None):
    """让 persona 判断是否会转发 content。返回 dict。"""
    if cache is not None:
        cached = cache.get(persona.id, content)
        if cached is not None:
            return cached

    prompt = PERSONA_QUERY_PROMPT.format(
        persona_desc=persona.describe(),
        content=content,
    )

    temp = temperature if temperature is not None else config.PERSONA_TEMPERATURE

    for attempt in range(max_retries):
        try:
            raw = backend.chat(prompt, temperature=temp, max_tokens=512)
        except Exception as e:
            # 网络/模型层面的临时错误，重试
            if attempt == max_retries - 1:
                break
            continue

        parsed = extract_json_object(raw)
        if parsed and "would_forward" in parsed:
            result = _sanitize(parsed)
            if cache is not None:
                cache.set(persona.id, content, result)
            return result
        # 否则重试（LLM 也许下次会老实一点）

    # 失败降级
    if cache is not None:
        cache.set(persona.id, content, _FALLBACK)
    return dict(_FALLBACK)
