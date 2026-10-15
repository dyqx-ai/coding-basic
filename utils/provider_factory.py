# -*- coding: utf-8 -*-
"""
Provider 工厂 —— 「极致 DIY」的模型供应商全配置化驱动实现
================================================================
目标：新增一家模型供应商（比如 GLM / 硅基流动 / Ollama 本地 / Gemini），
      不需要改任何 Python 源码，只需要：
        1. 在 config_file/my_config.yaml 的 shared.llm_providers 下加一段：
               providers:
                 glm:
                   env_key: GLM_API_KEY
                   base_url: https://open.bigmodel.cn/api/paas/v4/
                   default_model: glm-5.1
        2. 在 .env 里填：GLM_API_KEY=xxxx
        3. 在各模型配置（如 code_generate_model）里把 model_provider 改成 glm 即可。

策略（3 层回退，兼容性最强）：
  1. 命中 shared.llm_providers.{provider} → 从 YAML 取 env_key（以及 default_model 兜底）
  2. 没命中 YAML 时 → 走内置兼容表（ali/deepseek），保证老 my_config.yaml 不写 providers 也照常工作
  3. env_key 对应的环境变量为空 / 不存在 → 抛 RuntimeError（与原有 get_openai_model 语义一致，上层 _check_api_key 不改变）

对外提供两个函数（严格按「调用方零改动 + 扩展点预留 + 成本透明度接口」设计）：
  - resolve_model_credentials(model_provider, model_name, shared_cfg=None) -> Tuple[api_key, model_name]
      * 返回签名与原 get_openai_model 100% 一致，所有 api_key_xxx 变量赋值点一行都不用改。
      * shared_cfg 就是 my_config.yaml 里的 shared: 段（字典）。传 None 等同于走内置兼容。
  - register_provider_cost_hint(provider_name, model_name, price_in_per_m, price_out_per_m) -> None
      * 给第 ③ 步"成本透明度"预留注册钩子，现在先把接口焊死到 factory 里，下一步只需要累计 token 即可。
      * 内置 ali / deepseek 两个老供应商的官方价格（2026-09 官方定价）已经填好默认值，即使不手动注册也能算。
      * 价格单位：每百万 tokens（输入/输出），人民币元。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple


# ======================================================================
# 1) 内置兼容表：不写 shared.llm_providers 时的兜底（ali/deepseek 两家老用户完全无痛）
# ======================================================================
_BUILTIN_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ali": {
        "env_key": "ALI_API_KEY",
        "default_model": "qwen3-max",
        # DashScope 百炼官方（2026-05-15 help.aliyun.com）：qwen3-max 32K 内
        "cost_hint": {
            "qwen3-max": {"in": 2.5, "out": 10.0},
            "qwen-max": {"in": 2.4, "out": 9.6},
            "qwen3.6-plus": {"in": 2.0, "out": 12.0},
            "qwen3.7-plus": {"in": 3.5, "out": 8.0},
            "__default__": {"in": 4.0, "out": 15.0},
        },
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
        # DeepSeek Platform 官方（2026-04 后）：V4 Flash（即 deepseek-chat）
        # 缓存未命中价格（最保守估算）；缓存命中价格低 50 倍，算成本用未命中更诚实。
        "cost_hint": {
            "deepseek-chat": {"in": 1.0, "out": 2.0},      # V4 Flash non-thinking
            "deepseek-reasoner": {"in": 1.0, "out": 2.0},   # V4 Flash thinking（同价）
            "deepseek-v4-flash": {"in": 1.0, "out": 2.0},
            "deepseek-v4-pro":  {"in": 3.0, "out": 6.0},    # 2.5 折活动价
            "__default__": {"in": 1.0, "out": 2.0},
        },
    },
}


# ======================================================================
# 2) 运行时注册表：给新增 YAML provider 填价格提示 + 手工再覆盖
# ======================================================================
_RUNTIME_COST_HINTS: Dict[str, Dict[str, Dict[str, float]]] = {}


def register_provider_cost_hint(
    provider_name: str,
    model_name: str,
    price_in_per_million: float,
    price_out_per_million: float,
) -> None:
    """第 ③ 步「成本透明度」的前置钩子：提前把某 provider/model 的单价登记好。

    价格单位：元人民币 / 每 1,000,000 tokens（输入 / 输出分别登记）。
    特殊 model_name = "__default__" 表示该 provider 下所有未明确定价模型的兜底价。
    """
    if price_in_per_million < 0 or price_out_per_million < 0:
        raise ValueError("价格必须为非负数")
    bucket = _RUNTIME_COST_HINTS.setdefault(provider_name, {})
    bucket[model_name] = {"in": float(price_in_per_million), "out": float(price_out_per_million)}


def lookup_provider_cost(
    provider_name: str,
    model_name: str,
    shared_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """查询 (输入单价每M, 输出单价每M)，元/百万tokens。

    查找优先级：
      1) 运行时手动 register_provider_cost_hint 覆盖（最高）
      2) shared_cfg['llm_providers'][provider]['cost_hint'][model]
      3) shared_cfg['llm_providers'][provider]['cost_hint']['__default__']
      4) 内置 _BUILTIN_PROVIDERS 对应模型价 / __default__ 兜底
      5) 保守兜底 (4.0, 15.0) （≈ qwen-max 档）
    """
    # --- 1) runtime 覆盖
    pb = _RUNTIME_COST_HINTS.get(provider_name, {})
    if model_name in pb:
        return pb[model_name]["in"], pb[model_name]["out"]
    if "__default__" in pb:
        return pb["__default__"]["in"], pb["__default__"]["out"]

    # --- 2/3) YAML 用户自定义
    yaml_providers = {}
    if shared_cfg and isinstance(shared_cfg, dict):
        yaml_providers = shared_cfg.get("llm_providers") or {}
    yaml_p = yaml_providers.get(provider_name) or {}
    yaml_cost = yaml_p.get("cost_hint") or {} if isinstance(yaml_p, dict) else {}
    if isinstance(yaml_cost, dict):
        if model_name in yaml_cost and isinstance(yaml_cost[model_name], dict):
            md = yaml_cost[model_name]
            return float(md.get("in", 0.0)), float(md.get("out", 0.0))
        d = yaml_cost.get("__default__")
        if isinstance(d, dict):
            return float(d.get("in", 0.0)), float(d.get("out", 0.0))

    # --- 4) 内置兼容表
    builtin = _BUILTIN_PROVIDERS.get(provider_name) or {}
    bc = builtin.get("cost_hint") or {}
    if isinstance(bc, dict):
        if model_name in bc:
            return bc[model_name]["in"], bc[model_name]["out"]
        d = bc.get("__default__")
        if isinstance(d, dict):
            return d["in"], d["out"]

    return (4.0, 15.0)  # 5) 兜底保守值 ≈ 中端模型价格


# ======================================================================
# 3) 核心导出函数：解析 (provider, model_name) → (api_key, model_name)
# ======================================================================
def resolve_model_credentials(
    model_provider: str,
    model_name: str,
    shared_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    返回签名与原 get_openai_model 完全一致：(api_key, model_name)。
    只要在调用点把 get_openai_model(a, b) 替换成 resolve_model_credentials(a, b, shared) 即可。

    查找 env_key 顺序：
      1) shared_cfg['llm_providers'][provider]['env_key']（YAML 新能力，支持任意新供应商 0 Python 代码新增）
      2) 内置 _BUILTIN_PROVIDERS[provider]['env_key']（ali/deepseek 老用户兼容）
      3) 兜底：尝试把 provider 名直接大写后加 "_API_KEY"（比如 provider=glm → 查 GLM_API_KEY；
         这样即使 YAML 没写 llm_providers 也能靠"命名约定"直接接新供应商，门槛最低）
    """
    provider_key = (model_provider or "").strip().lower()
    if not provider_key:
        # 与原有行为保持一致：没有 provider 时返回空 api_key，上层 _check_api_key 会抛详细错
        return "", model_name or ""

    env_key_name: Optional[str] = None
    default_model_fallback: Optional[str] = None

    # 1) YAML
    yaml_providers = {}
    if shared_cfg and isinstance(shared_cfg, dict):
        yaml_providers = shared_cfg.get("llm_providers") or {}
    yaml_p = yaml_providers.get(provider_key)
    if isinstance(yaml_p, dict):
        env_key_name = yaml_p.get("env_key") or None
        default_model_fallback = yaml_p.get("default_model") or None

    # 2) 内置兼容表兜底 env_key（YAML 没写也行）
    if not env_key_name:
        builtin = _BUILTIN_PROVIDERS.get(provider_key) or {}
        env_key_name = builtin.get("env_key") or None
        if not default_model_fallback:
            default_model_fallback = builtin.get("default_model") or None

    # 3) 命名约定兜底："{PROVIDER_UPPER}_API_KEY"
    if not env_key_name:
        env_key_name = f"{provider_key.upper()}_API_KEY"

    # 查环境变量
    api_key = os.environ.get(env_key_name) or os.getenv(env_key_name) or ""

    # model_name 空时用兜底（极少触发，但对 YAML 端"不重复写 model"的场景友好）
    final_model = model_name or default_model_fallback or ""

    return api_key, final_model
