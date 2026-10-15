import os
import json
import re
import base64
import subprocess
import time
import sys
import hashlib
from typing import Annotated, TypedDict, List, Dict, Optional, Callable, Type
from typing import Literal
from datetime import datetime

import docker
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI

# ========== Pydantic 校验导入 ==========
from pydantic import BaseModel, Field, ValidationError

# ★★★ 缓存管理器导入 ★★★
from my_vibecoding.utils.cache_manager import get_cached_summary, set_cached_summary

# ★★★ Prompt 安全渲染（替代 str.format，规避模板 JSON 花括号 KeyError） ★★★
from my_vibecoding.utils.prompt_utils import safe_format

# ★★★ 长任务进度实时入库（web 端轮询可见，失败静默不影响主流程） ★★★
try:
    from my_vibecoding.utils.progress_notifier import notify_progress
except ImportError:
    try:
        from progress_notifier import notify_progress  # type: ignore
    except Exception:
        def notify_progress(conv_id, text, min_interval=8.0):  # type: ignore
            pass

# ★★★ Provider 工厂（供应商全配置化驱动 / 新增供应商 0 Python 代码） ★★★
try:
    from my_vibecoding.utils.provider_factory import resolve_model_credentials
except ImportError:
    # 极端容错：utils 目录无法作为包 import 时退化为 sys.path 直引（保持可运行）
    try:
        _prov_mod_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
        if _prov_mod_dir not in sys.path:
            sys.path.insert(0, _prov_mod_dir)
        from provider_factory import resolve_model_credentials  # type: ignore
    except Exception:
        # 最后兜底：等价于原有 if/elif（保证不会因工厂导入失败导致整个模块崩）
        def resolve_model_credentials(model_provider, model_name, shared_cfg=None):  # type: ignore
            if model_provider == "ali":
                return os.getenv("ALI_API_KEY", ""), model_name
            if model_provider == "deepseek":
                return os.getenv("DEEPSEEK_API_KEY", ""), model_name
            return "", model_name

# ★★★ 从 project_generate_code 导入 AST 校验函数 ★★★
try:
    from my_vibecoding.project_generate_code.project_generate_code import (
        validate_code_against_schema,
        extract_exported_apis_from_code
    )
except ImportError:
    def validate_code_against_schema(source_code, expected_apis):
        return []
    def extract_exported_apis_from_code(source_code):
        return {}


load_dotenv()
# ========== 修复：基于 __file__ 的绝对路径，避免依赖 cwd 导致配置文件找不到 ==========
_config_script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(_config_script_dir, "../config_file/my_config.yaml"))

################################################################################################################
# 过滤系统模块
try:
    STDLIB_MODULES = set(sys.stdlib_module_names)
except AttributeError:
    STDLIB_MODULES = {
        "os", "sys", "re", "json", "time", "datetime", "math", "random",
        "subprocess", "threading", "multiprocessing", "socket", "ssl",
        "http", "urllib", "email", "xml", "html", "csv", "io", "struct",
        "hashlib", "hmac", "base64", "binascii", "zlib", "gzip", "zipfile",
        "tarfile", "shutil", "tempfile", "pathlib", "glob", "fnmatch",
        "argparse", "logging", "warnings", "contextlib", "itertools",
        "functools", "operator", "collections", "abc", "typing", "enum",
        "dataclasses", "asyncio", "concurrent", "signal", "atexit",
        "platform", "ctypes", "getpass", "getopt", "optparse",
        "pprint", "reprlib", "string", "textwrap", "unicodedata",
        "codecs", "pickle", "shelve", "copy", "weakref", "gc",
        "inspect", "dis", "traceback", "linecache", "tokenize",
        "keyword", "ast", "symtable", "parser",
        "compileall", "py_compile", "importlib", "pkgutil",
    }

################################################################################################################
# ====================== 统一日志系统 ======================
class LogColor:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    END = '\033[0m'


_indent_level = 0


def log_indent():
    global _indent_level
    _indent_level += 1


def log_dedent():
    global _indent_level
    if _indent_level > 0:
        _indent_level -= 1


def log_info(module: str, node: str, msg: str):
    indent = "  " * _indent_level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LogColor.CYAN}[{timestamp}] [{module}] [{node}] {indent}{msg}{LogColor.END}")


def log_success(module: str, node: str, msg: str):
    indent = "  " * _indent_level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LogColor.GREEN}[{timestamp}] [{module}] [{node}] {indent}✅ {msg}{LogColor.END}")


def log_warn(module: str, node: str, msg: str):
    indent = "  " * _indent_level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LogColor.YELLOW}[{timestamp}] [{module}] [{node}] {indent}⚠️ {msg}{LogColor.END}")


def log_error(module: str, node: str, msg: str):
    indent = "  " * _indent_level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LogColor.RED}[{timestamp}] [{module}] [{node}] {indent}❌ {msg}{LogColor.END}")


def log_separator(module: str, title: str = ""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if title:
        print(f"{LogColor.BOLD}{LogColor.MAGENTA}[{timestamp}] [===== {module} - {title} =====]{LogColor.END}")
    else:
        print(f"{LogColor.MAGENTA}[{timestamp}] [===== {module} =====]{LogColor.END}")


def log_debug(module: str, node: str, msg: str):
    indent = "  " * _indent_level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{LogColor.GRAY}[{timestamp}] [{module}] [{node}] {indent}{msg}{LogColor.END}")


################################################################################################################
# ====================== 配置读取 ======================
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败，请检查路径: {config_path}, 错误: {e}") from e

# ========== 从 shared 读取沙箱配置 ==========
shared = config.get("shared", {})
SANDBOX_IMAGE = shared.get("sandbox_image", "sandbox-web:v1")
SANDBOX_TIMEOUT = shared.get("single_sandbox_timeout", 240)
MAIN_SANDBOX_TIMEOUT = shared.get("main_sandbox_timeout", 300)
VENDOR_DIR = shared.get("vendor_mount_path", "/tmp/vendor")
# Docker 不可用时是否用本地 Python 兜底跑顶层主测（忘开 Docker 也不跳过）
LOCAL_RUN_FALLBACK = bool(shared.get("local_run_fallback", True))
# 本地兜底用的 Python 解释器（留空用当前运行环境 sys.executable）
LOCAL_PYTHON = str(shared.get("local_python", "") or "").strip()
# Docker 连接失败后自动重连的冷却秒数（中途打开 Docker 后最迟这么久恢复，无需重启服务）
DOCKER_RETRY_COOLDOWN = float(shared.get("docker_retry_cooldown", 20))
PIP_INDEX_URL = shared.get("pip_index_url", "https://mirrors.aliyun.com/pypi/simple/")
PIP_TRUSTED_HOST = shared.get("pip_trusted_host", "mirrors.aliyun.com")
PIP_MAX_RETRIES = shared.get("pip_max_retries", 3)
PIP_RETRY_DELAY = shared.get("pip_retry_delay", 5)

# ========== 镜像预装依赖列表 ==========
preinstalled_pip = shared.get("preinstalled_pip", [])
preinstalled_names = {item.split("==")[0] for item in preinstalled_pip}

# ========== 模块名 → pip 包名映射 ==========
module_to_pip = shared.get("module_to_pip", {})

# ========== 子图2专属配置 ==========
project_path = config["change_project_path"]
absolute_project_path = os.path.abspath(project_path)


def _auto_detect_project_root(base_path: str) -> str:
    """自动检测项目根目录"""
    if not os.path.exists(base_path):
        log_warn("子图2", "路径检测", f"配置的项目路径不存在，将按原样使用: {base_path}")
        return base_path
    try:
        items = os.listdir(base_path)
    except OSError as e:
        log_warn("子图2", "路径检测", f"无法列出目录 {base_path}: {e}，按原样使用")
        return base_path
    subdirs = [item for item in items if os.path.isdir(os.path.join(base_path, item))]

    # ★★★ 排除 config/ 目录（它只是配置文件夹，不是项目根目录）★★★
    exclude_dirs = ["config", "configs", ".ai_file_summary", ".modify_backup", "__pycache__", ".git", ".idea",
                    ".vscode"]
    subdirs = [d for d in subdirs if d not in exclude_dirs]

    # 如果只有一个子目录，直接进入
    if len(subdirs) == 1:
        subdir_path = os.path.join(base_path, subdirs[0])
        log_info("子图2", "路径检测", f"唯一子目录，自动进入: {subdirs[0]}")
        return subdir_path

    # ★★★ 优先查找包含 src/ 的目录（这才是真正的项目根目录）★★★
    for sub in subdirs:
        subdir_path = os.path.join(base_path, sub)
        if os.path.exists(os.path.join(subdir_path, "src")):
            log_info("子图2", "路径检测", f"发现 src 目录，自动进入: {sub}")
            return subdir_path

    # 检查包含 Python 文件的目录（但如果有 src/ 则跳过）
    for sub in subdirs:
        subdir_path = os.path.join(base_path, sub)
        # ★★★ 如果这个目录下有 src/ 子目录，说明它不是真正的根目录 ★★★
        if os.path.exists(os.path.join(subdir_path, "src")):
            continue
        try:
            dir_items = os.listdir(subdir_path) if os.path.exists(subdir_path) else []
        except OSError:
            dir_items = []
        py_files = [f for f in dir_items if f.endswith('.py')]
        if len(py_files) > 0:
            log_info("子图2", "路径检测", f"检测到 Python 文件，自动进入: {sub}")
            return subdir_path

    return base_path


try:
    absolute_project_path = _auto_detect_project_root(absolute_project_path)
except Exception as e:
    log_warn("子图2", "路径检测", f"自动检测项目根目录失败 {e}，将使用配置路径: {absolute_project_path}")
log_info("子图2", "路径检测", f"最终项目路径: {absolute_project_path}")

# ★★★ 提取顶层目录名（用于路径标准化）★★★
PROJECT_TOP_DIR = os.path.basename(absolute_project_path) or ""
log_info("子图2", "路径检测", f"项目顶层目录名: {PROJECT_TOP_DIR}")

ai_summary_dir = os.path.join(absolute_project_path, ".ai_file_summary")
summary_md_path = os.path.join(ai_summary_dir, "summary.md")


# ============================================================================
# ========== Schema 存在性检测（决定是否执行 AST 契约校验） ==========
# 背景:
#   1) 本系统 complete 模式生成的项目 → 有 final_plan.md / .schema.json，
#      export_apis 来自 Schema，具有权威性，应该做 AST 校验。
#   2) 普通已有项目（如图书管理系统）→ 没有 Schema，export_apis 是
#      LLM 生成摘要时"猜"的，用猜测结果去校验修改会形成死循环，应跳过。
# ============================================================================
def _detect_schema(project_root: str, summary_file: str) -> bool:
    """
    检测当前项目是否有本系统 complete 模式产出的 Schema。

    判定顺序 (任意一条命中即视为有 Schema):
      1. 项目根下有 final_plan.md   (complete 模式固定产物)
      2. 项目根下有 schema.json 或 .schema.json
      3. .ai_file_summary/summary.md 中任意一个 export_apis 元素是 dict
         (complete 模式存储的是完整接口定义: {name,type,params,...},
          而 LLM 摘要时仅存储 name 字符串列表)
    """
    # 1. final_plan.md 检测（直接在项目根以及父目录一起查，兼容 src/ 双层结构）
    plan_candidates = [
        os.path.join(project_root, "final_plan.md"),
        os.path.join(os.path.dirname(project_root), "final_plan.md"),
    ]
    for p in plan_candidates:
        try:
            if os.path.isfile(p):
                log_info("子图2", "Schema检测", f"命中 final_plan.md → HAS_SCHEMA=True  ({p})")
                return True
        except Exception:
            pass

    # 2. schema.json 检测
    schema_candidates = [
        os.path.join(project_root, "schema.json"),
        os.path.join(project_root, ".schema.json"),
        os.path.join(os.path.dirname(project_root), "schema.json"),
        os.path.join(os.path.dirname(project_root), ".schema.json"),
    ]
    for s in schema_candidates:
        try:
            if os.path.isfile(s):
                log_info("子图2", "Schema检测", f"命中 schema.json → HAS_SCHEMA=True  ({s})")
                return True
        except Exception:
            pass

    # 3. summary.md 中 export_apis 类型检测
    try:
        if os.path.isfile(summary_file):
            import re as _re
            content = open(summary_file, "r", encoding="utf-8", errors="ignore").read()
            # 找 export_apis 字段附近的内容，取 2KB 窗口做 json 抽样
            idx = content.lower().find("export_apis")
            if idx >= 0:
                window = content[idx: idx + 2048]
                # 若直接能看到 [{ 开头形式 → 说明是 dict 列表（Schema产物）
                dict_sign = _re.search(r"export_apis\s*[:=]\s*\[", window, _re.IGNORECASE)
                if dict_sign:
                    after_bracket = window[dict_sign.end(): dict_sign.end() + 50]
                    if "{" in after_bracket:
                        log_info("子图2", "Schema检测",
                                 "summary.md 中 export_apis 元素为 dict 格式 → HAS_SCHEMA=True")
                        return True
    except Exception:
        pass

    log_info("子图2", "Schema检测", "未发现 Schema 背书 → HAS_SCHEMA=False（跳过 AST 契约校验）")
    return False


try:
    HAS_SCHEMA: bool = _detect_schema(absolute_project_path, summary_md_path)
except Exception as _e:
    log_warn("子图2", "Schema检测", f"检测异常，按无 Schema 处理: {_e}")
    HAS_SCHEMA = False
# ============================================================================

test_config = config.get("change_test_runtime", {})
UNIT_MAX_RETRY = test_config.get("unit_max_retry", 3)
MAIN_MAX_RETRY = test_config.get("main_max_retry", 3)
COMMON_ENTRY_FILES = test_config.get("common_entry_files", [
    "main.py", "app.py", "run.py", "cli.py", "src/main.py", "src/app.py"
])

# ========== Docker 客户端（惰性初始化） ==========
_docker_client = None
_docker_client_failed_at = 0.0


def get_docker_client():
    """获取 Docker 客户端。

    关键修复：连接失败【不再永久缓存】。旧逻辑一旦在服务启动时 Docker 没开，
    就把 _docker_client 钉死成 False，之后哪怕中途打开 Docker，不重启服务也永远返回 None
    （= 主测每次都跳过）。现在改为：
      - 成功连接后缓存，并在每次取用前 ping 探活（Docker 中途关了会自动重连）；
      - 连接失败只记录时间戳，冷却 DOCKER_RETRY_COOLDOWN 秒后允许重试，
        这样用户中途打开 Docker，最迟冷却期过后下一次主测/单测即可自动恢复，无需重启服务。
    """
    global _docker_client, _docker_client_failed_at
    import time as _time

    # 已缓存的可用连接：先探活，掉了就落到下面重连
    if _docker_client is not None and _docker_client is not False:
        try:
            _docker_client.ping()
            return _docker_client
        except Exception:
            _docker_client = None

    # 之前失败过且仍在冷却期内 → 直接返回 None，避免每次调用都慢等连接超时
    if _docker_client is False:
        if _time.time() - _docker_client_failed_at < DOCKER_RETRY_COOLDOWN:
            return None
        # 冷却已过，允许重新尝试
        _docker_client = None

    if _docker_client is None:
        try:
            os.environ["DOCKER_CONTEXT"] = "default"
            c = docker.from_env()
            c.ping()
            _docker_client = c
            _docker_client_failed_at = 0.0
            log_info("子图2", "Docker", "连接成功")
            return _docker_client
        except Exception as e:
            log_warn("子图2", "Docker",
                     f"连接失败，沙箱功能暂不可用: {e}（{DOCKER_RETRY_COOLDOWN:.0f}s 后自动重试，无需重启服务）")
            _docker_client = False
            _docker_client_failed_at = _time.time()
            return None
    return None


################################################################################################################
# ====================== Pydantic 校验模型定义 ======================
class DetectSummaryOutput(BaseModel):
    exist_summary_file: bool
    summary_file_rel_path: str = Field(default="")


class FileSummaryOutput(BaseModel):
    rel_path: str
    file_desc: str = Field(default="")
    export_apis: List[str] = Field(default_factory=list)
    depend_on: List[str] = Field(default_factory=list)


class ChangeCodeOutput(BaseModel):
    is_enough_info: bool
    insufficient_reason: str = Field(default="")
    need_additional_file_list: List[str] = Field(default_factory=list)
    changed_files: Dict[str, str] = Field(default_factory=dict)
    processed_file_list: List[str] = Field(default_factory=list)
    inferred_can_unit_test: List[str] = Field(default_factory=list)
    inferred_top_entry: str = Field(default="")
    top_entry_confidence: Literal["high", "medium", "low"] = Field(default="low")


class SandboxJudgeOutput(BaseModel):
    success: bool
    is_environment_error: bool = False
    brief_reason: str = Field(default="")
    error_detail: str = Field(default="")


class MainFixOutput(BaseModel):
    is_environment_error: bool
    error_reason: str = Field(default="")
    modified_files: List[Dict[str, str]] = Field(default_factory=list)


################################################################################################################
# ====================== 业务逻辑校验函数 ======================
def validate_change_code_logic(validated: ChangeCodeOutput) -> List[str]:
    errors = []
    if validated.is_enough_info and not validated.changed_files:
        errors.append("is_enough_info 为 true，但 changed_files 为空")
    if not validated.is_enough_info and not validated.insufficient_reason.strip():
        errors.append("is_enough_info 为 false，但 insufficient_reason 为空")
    for fp in validated.changed_files.keys():
        if not fp or fp.strip() == "":
            errors.append("changed_files 中存在空的文件路径")
    for content in validated.changed_files.values():
        if content is None:
            errors.append("changed_files 中存在 None 值")
    return errors


def validate_file_summary_logic(validated: FileSummaryOutput) -> List[str]:
    errors = []
    if not validated.rel_path or validated.rel_path.strip() == "":
        errors.append("rel_path 不能为空")
    return errors


def validate_main_fix_logic(validated: MainFixOutput) -> List[str]:
    errors = []
    if not validated.is_environment_error and not validated.modified_files:
        errors.append("is_environment_error 为 false，但 modified_files 为空")
    if validated.is_environment_error and not validated.error_reason.strip():
        errors.append("is_environment_error 为 true，但 error_reason 为空")
    return errors


################################################################################################################
# ====================== 通用校验函数 ======================
def get_retry_count_for_node(
        node_key: str,
        group_retry: int,
        model_output_retry_cfg: dict
) -> int:
    default = model_output_retry_cfg.get("default", 2)
    overrides = model_output_retry_cfg.get("overrides", {})
    if node_key in overrides:
        return overrides[node_key]
    return group_retry


def _build_retry_prompt(original_prompt: str, error_history: List[str], new_error: str) -> str:
    recent_errors = error_history[-3:] if len(error_history) > 3 else error_history
    history_text = "\n".join([f"  - {e}" for e in recent_errors])
    return (
            original_prompt
            + f"\n\n【⚠️ 上一轮输出校验失败，请修正后重新输出】\n"
            + f"错误历史:\n{history_text}\n\n"
            + f"最新错误: {new_error}\n\n"
            + "请仔细检查上述所有错误，逐一修正后重新输出完整的 JSON。\n"
            + "只输出纯净的 JSON，不要添加任何解释、思考、markdown 标记。"
    )


def invoke_with_validation(
        model: ChatOpenAI,
        prompt: str,
        pydantic_model: Type[BaseModel],
        max_retries: int = 2,
        node_name: str = "unknown",
        logic_validator: Optional[Callable[[BaseModel], List[str]]] = None,
        validator_context: str = ""
) -> BaseModel:
    current_prompt = prompt
    error_history = []
    last_raw_output = ""
    last_format_error = ""
    last_logic_error = ""

    for attempt in range(max_retries + 1):
        try:
            resp = model.invoke(current_prompt)
            last_raw_output = resp.content
            cleaned = clean_json_data(resp.content)

            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_format_error = f"JSON解析失败: {e}"
                error_history.append(f"[格式错误] {last_format_error}")
                if attempt < max_retries:
                    current_prompt = _build_retry_prompt(
                        prompt, error_history,
                        f"JSON 格式解析失败，请检查括号、引号是否匹配。错误: {e}"
                    )
                    log_warn("子图2", node_name, f"JSON解析失败，第 {attempt + 1} 次重试...")
                    continue
                else:
                    raise RuntimeError(
                        f"模型输出格式校验失败（重试{max_retries}次后仍失败）: {last_format_error}"
                    )

            try:
                validated = pydantic_model(**data)
            except ValidationError as e:
                last_format_error = f"字段校验失败: {e}"
                error_history.append(f"[字段校验] {last_format_error}")
                if attempt < max_retries:
                    current_prompt = _build_retry_prompt(
                        prompt, error_history,
                        f"JSON 字段不符合预期格式。错误: {e}"
                    )
                    log_warn("子图2", node_name, f"字段校验失败，第 {attempt + 1} 次重试...")
                    continue
                else:
                    raise RuntimeError(
                        f"模型输出字段校验失败（重试{max_retries}次后仍失败）: {last_format_error}"
                    )

            if logic_validator:
                logic_errors = logic_validator(validated)
                if logic_errors:
                    last_logic_error = "\n".join(logic_errors)
                    error_history.append(f"[业务逻辑] {last_logic_error}")
                    if attempt < max_retries:
                        current_prompt = _build_retry_prompt(
                            prompt, error_history,
                            f"业务逻辑校验失败:\n{last_logic_error}"
                        )
                        log_warn("子图2", node_name, f"业务逻辑校验失败，第 {attempt + 1} 次重试...")
                        continue
                    else:
                        raise RuntimeError(
                            f"业务逻辑校验失败（重试{max_retries}次后仍失败）: {last_logic_error}"
                        )

            return validated

        except RuntimeError:
            raise

        except Exception as e:
            error_history.append(f"[未知错误] {str(e)}")
            if attempt < max_retries:
                current_prompt = _build_retry_prompt(
                    prompt, error_history,
                    f"发生未知错误: {str(e)}"
                )
                log_warn("子图2", node_name, f"未知错误，第 {attempt + 1} 次重试...")
                continue
            else:
                raise RuntimeError(
                    f"模型调用失败（重试{max_retries}次后仍失败）: {str(e)}"
                )

    raise RuntimeError(f"模型调用失败（重试{max_retries}次后仍失败）")


def get_openai_model(model_provider: str, model_name: str):
    """Provider 工厂兼容封装（保留原函数名与签名，调用方零改动即可升级）。
    新增供应商支持：在 shared.llm_providers 加 YAML 段 + .env 加 key 即可，不改 Python。"""
    return resolve_model_credentials(model_provider, model_name, shared)


def _check_api_key(provider_name, key):
    if not key:
        raise RuntimeError(f"[{provider_name}] API密钥未配置，请检查.env文件")


# ========== 5个模型 ==========
api_key_analyze_change, model_name_analyze_change = get_openai_model(
    config["analyze_change_model"]["model_provider"],
    config["analyze_change_model"]["model_name"])

api_key_summary, model_name_summary = get_openai_model(
    config["file_summary_model"]["model_provider"],
    config["file_summary_model"]["model_name"])

api_key_code_generate, model_name_code_generate = get_openai_model(
    config["code_change_model"]["model_provider"],
    config["code_change_model"]["model_name"])

change_test_fix_cfg = config.get("change_test_fix_model", {})
api_key_change_test_fix, model_name_change_test_fix = get_openai_model(
    change_test_fix_cfg.get("model_provider", ""),
    change_test_fix_cfg.get("model_name", ""))

change_test_judge_cfg = config.get("change_test_judge_model", {})
api_key_change_test_judge, model_name_change_test_judge = get_openai_model(
    change_test_judge_cfg.get("model_provider", ""),
    change_test_judge_cfg.get("model_name", ""))

_check_api_key(config["analyze_change_model"]["model_provider"], api_key_analyze_change)
_check_api_key(config["file_summary_model"]["model_provider"], api_key_summary)
_check_api_key(config["code_change_model"]["model_provider"], api_key_code_generate)
_check_api_key("change_test_fix_model", api_key_change_test_fix)
_check_api_key("change_test_judge_model", api_key_change_test_judge)


model_analyze_change = ChatOpenAI(
    model=model_name_analyze_change,
    api_key=api_key_analyze_change,
    base_url=config["analyze_change_model"]["base_url"],
    temperature=0.1,
    timeout=60
)

model_file_summary = ChatOpenAI(
    model=model_name_summary,
    api_key=api_key_summary,
    base_url=config["file_summary_model"]["base_url"],
    temperature=0.1,
    timeout=60
)

model_code_change = ChatOpenAI(
    model=model_name_code_generate,
    api_key=api_key_code_generate,
    base_url=config["code_change_model"]["base_url"],
    temperature=0.2,
    timeout=60
)

model_change_test_fix = ChatOpenAI(
    model=model_name_change_test_fix,
    api_key=api_key_change_test_fix,
    base_url=change_test_fix_cfg.get("base_url", ""),
    temperature=change_test_fix_cfg.get("temperature", 0.1),
    timeout=change_test_fix_cfg.get("timeout", 60)
)

model_change_test_judge = ChatOpenAI(
    model=model_name_change_test_judge,
    api_key=api_key_change_test_judge,
    base_url=change_test_judge_cfg.get("base_url", ""),
    temperature=change_test_judge_cfg.get("temperature", 0.0),
    timeout=change_test_judge_cfg.get("timeout", 30)
)


################################################################################################################
# json数据清洗
def clean_json_data(json_str: str) -> str:
    content = json_str.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json")
        content = content.removesuffix("```")
    elif content.startswith("```"):
        content = content.removeprefix("```")
        content = content.removesuffix("```")
    content = content.strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if match:
        content = match.group(0)
    return content.strip()


def parse_local_summary_md(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, Exception) as e:
        log_error("子图2", "解析摘要", f"{e}")
        return []


################################################################################################################
# ====================== 核心沙箱函数 ======================
def _extract_imports(source_code: str) -> list[str]:
    imports = []
    lines = source_code.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("import "):
            parts = line[7:].split(",")
            for p in parts:
                p = p.strip()
                if p.startswith("."):
                    continue
                p = p.split()[0]
                if p and p not in STDLIB_MODULES:
                    try:
                        import importlib.util
                        spec = importlib.util.find_spec(p)
                        if spec is None:
                            imports.append(p)
                    except Exception:
                        imports.append(p)

        elif line.startswith("from "):
            module_part = line[5:].split(" import ")[0].strip()
            if module_part.startswith("."):
                continue
            p = module_part.split()[0] if module_part else ""
            if p and p not in STDLIB_MODULES:
                try:
                    import importlib.util
                    spec = importlib.util.find_spec(p)
                    if spec is None:
                        imports.append(p)
                except Exception:
                    imports.append(p)

    return imports


def run_single_file_sandbox(source_code: str, deps: Optional[List[str]] = None) -> dict:
    client = get_docker_client()
    if client is None:
        return {
            "success": False,
            "return_code": -3,
            "stdout": "",
            "stderr": "Docker 守护程序未运行或连接失败，沙箱功能不可用。请启动 Docker 服务。"
        }

    import tempfile
    temp_file_path = None

    try:
        if deps is not None:
            pip_deps = [pkg for pkg in deps if pkg not in preinstalled_names]
        else:
            required_modules = _extract_imports(source_code)
            need_pkg_set = set()
            for mod_name in required_modules:
                pkg = module_to_pip.get(mod_name, mod_name)
                need_pkg_set.add(pkg)
            pip_deps = [pkg for pkg in need_pkg_set if pkg not in preinstalled_names]

        pip_mirrors = shared.get("pip_mirrors")
        if pip_mirrors and isinstance(pip_mirrors, list):
            mirrors = pip_mirrors
        else:
            fallback_url = shared.get("pip_index_url", "https://mirrors.aliyun.com/pypi/simple/")
            fallback_host = shared.get("pip_trusted_host", "mirrors.aliyun.com")
            mirrors = [{"index_url": fallback_url, "trusted_host": fallback_host}]

        payload = {
            "source_code": source_code,
            "pip_deps": pip_deps,
            "pip_mirrors": mirrors,
            "pip_max_retries": PIP_MAX_RETRIES,
            "pip_retry_delay": PIP_RETRY_DELAY,
            "vendor_dir": VENDOR_DIR,
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False)
            temp_file_path = f.name

        inner_script = f'''import sys
import json
import subprocess
import os
import time

def main():
    PAYLOAD_FILE = "/tmp/payload.json"
    try:
        with open(PAYLOAD_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(json.dumps({{"success": False, "stderr": f"读取payload失败: {{e}}"}}))
        return

    source_code = payload["source_code"]
    pip_deps = payload["pip_deps"]
    pip_mirrors = payload.get("pip_mirrors", [])
    pip_max_retries = payload.get("pip_max_retries", 3)
    pip_retry_delay = payload.get("pip_retry_delay", 5)
    vendor_dir = payload.get("vendor_dir", "/tmp/vendor")

    result = {{
        "success": False,
        "stdout": "",
        "stderr": "",
        "return_code": 1
    }}

    if pip_deps:
        os.makedirs(vendor_dir, exist_ok=True)
        installed = False
        last_error = ""

        for mirror in pip_mirrors:
            index_url = mirror.get("index_url")
            trusted_host = mirror.get("trusted_host")
            if not index_url:
                continue
            cmd = [
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "-i", index_url, "--trusted-host", trusted_host,
                "--target", vendor_dir,
                *pip_deps
            ]
            for attempt in range(pip_max_retries):
                try:
                    subprocess.run(cmd, capture_output=True, check=True, timeout=180)
                    installed = True
                    break
                except subprocess.CalledProcessError as e:
                    last_error = e.stderr.decode("utf-8", errors="ignore") if e.stderr else ""
                    if attempt < pip_max_retries - 1:
                        time.sleep(pip_retry_delay)
                except subprocess.TimeoutExpired as e:
                    last_error = f"pip install timeout (attempt {{attempt + 1}})"
                    if attempt < pip_max_retries - 1:
                        time.sleep(pip_retry_delay)
            if installed:
                break
            else:
                last_error = f"镜像 {{index_url}} 安装失败: {{last_error}}"

        if not installed:
            result["stderr"] = f"所有镜像源安装失败: {{last_error}}"
            result["return_code"] = 1
            print(json.dumps(result, ensure_ascii=False))
            return

        sys.path.insert(0, vendor_dir)

    SRC_FILE = "/tmp/demo_src.py"
    try:
        with open(SRC_FILE, "w", encoding="utf-8") as f:
            f.write(source_code)
    except Exception as e:
        result["stderr"] = str(e)
        result["return_code"] = 1
        print(json.dumps(result, ensure_ascii=False))
        return

    sys.path.insert(0, "/tmp")
    if "demo_src" in sys.modules:
        del sys.modules["demo_src"]

    try:
        __import__("demo_src")
        result["success"] = True
        result["stdout"] = "语法校验通过，模块导入成功"
        result["return_code"] = 0
    except SyntaxError as e:
        result["stderr"] = f"语法错误：{{str(e)}}\\n行号：{{e.lineno}}"
        result["return_code"] = 1
    except ImportError as e:
        result["stderr"] = f"导入错误：{{str(e)}}"
        result["return_code"] = 1
    except Exception as e:
        import traceback
        result["stderr"] = traceback.format_exc()
        result["return_code"] = 1

    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
'''

        output = client.containers.run(
            image=SANDBOX_IMAGE,
            command=["python", "-c", inner_script],
            detach=False,
            remove=True,
            volumes={
                temp_file_path: {"bind": "/tmp/payload.json", "mode": "ro"}
            },
        )
        log_text = output.decode("utf-8", errors="ignore").strip()

        if not log_text:
            return {"success": False, "return_code": -1, "stdout": "", "stderr": "容器无输出"}

        json_match = re.search(r'\{[\s\S]*\}', log_text)
        if json_match:
            report = json.loads(json_match.group(0))
            return {
                "success": report.get("success", False),
                "return_code": report.get("return_code", 1),
                "stdout": report.get("stdout", ""),
                "stderr": report.get("stderr", "")
            }
        else:
            return {"success": False, "return_code": -1, "stdout": "", "stderr": f"输出格式异常: {log_text[:200]}"}

    except docker.errors.ImageNotFound:
        return {"success": False, "return_code": -2, "stdout": "", "stderr": f"镜像 {SANDBOX_IMAGE} 不存在，请先构建"}
    except Exception as e:
        return {"success": False, "return_code": -1, "stdout": "", "stderr": f"沙箱执行异常：{str(e)}"}
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


################################################################################################################
# ====================== 提取文件末尾 @generated 注释 ======================
def extract_summary_from_file(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        pattern = r'#\s*@generated:\s*(\{.*?\})'
        matches = re.findall(pattern, content, re.DOTALL)
        if matches:
            return json.loads(matches[-1])
    except Exception:
        return None
    return None


################################################################################################################
# ====================== 所有提示词（示例均使用相对路径，不加顶层目录） ======================

sandbox_judge_prompt = """
你是代码测试判定专家。根据文件信息、沙箱执行输出，判断测试是否通过。

【文件信息】
文件路径：{file_path}
文件职责：{file_purpose}

【沙箱执行结果】
返回码：{return_code}
标准输出：
{stdout}
错误输出：
{stderr}

## 输出格式
严格输出 JSON 对象，格式如下：

### 正确示例（测试通过）
{{"success": true, "is_environment_error": false, "brief_reason": "语法校验通过，模块导入成功", "error_detail": ""}}

### 正确示例（测试失败，代码错误）
{{"success": false, "is_environment_error": false, "brief_reason": "语法错误：函数名拼写错误", "error_detail": "SyntaxError: invalid syntax at line 42"}}

### 正确示例（环境错误）
{{"success": false, "is_environment_error": true, "brief_reason": "缺少依赖包 openpyxl", "error_detail": "ModuleNotFoundError: No module named 'openpyxl'"}}

## JSON字段说明
1. success: bool - 是否通过测试
2. is_environment_error: bool - 是否环境错误
3. brief_reason: str - 简短判定原因
4. error_detail: str - 完整错误信息（成功则为空字符串）

硬性约束：只输出纯净JSON，不要添加任何解释、标记或额外文字。
"""

main_code_fix_prompt = """
你是Python高级修复工程师。根据顶层入口运行错误日志，定位并修复所有关联文件的问题。
⚠️ 可以修改顶层文件及其依赖的所有关联文件，严格保留所有对外接口契约。

【项目信息】
项目根目录：{project_root}
顶层入口文件：{entry_file}
文件职责：{file_purpose}

【错误日志】
{error_log}

## 输出格式
严格输出 JSON 对象，格式如下：

### 正确示例（代码可修复，已修改文件）
{{"is_environment_error": false, "error_reason": "", "modified_files": [{{"file_path": "src/services/book_service.py", "fixed_code": "修复后的完整源码...", "change_desc": "修复了函数名拼写错误"}}, {{"file_path": "src/cli/main.py", "fixed_code": "修复后的完整源码...", "change_desc": "更新了import语句"}}]}}

### 正确示例（环境错误，无法修复）
{{"is_environment_error": true, "error_reason": "缺少依赖包 openpyxl，需要手动安装", "modified_files": []}}

## JSON字段说明
1. is_environment_error: bool - 是否环境错误（无法通过修改代码修复）
2. error_reason: str - 环境错误说明（is_environment_error=true时必填）
3. modified_files: list[dict] - 修改的文件列表
   - file_path: str - 文件相对路径
   - fixed_code: str - 修复后的完整源码
   - change_desc: str - 修改说明

## 修复规则
1. 全局扫描错误根源，**允许修改顶层入口及所有关联依赖文件**
2. 严格保留所有对外导出接口签名
3. 精准修改出错部分，不改动无关逻辑
4. 修改类名、函数名、变量名时，**所有引用文件必须同步更新**
5. 修复后的代码必须保留 `if __name__ == "__main__":` 测试块

硬性约束：只输出纯净JSON，不要添加任何解释、标记或额外文字。
"""

detect_summary_md_prompt = """
你收到一份项目目录树文本，请扫描全部目录与文件，自主识别目录树内是否存在**项目代码摘要/架构摘要类markdown文档**。

## 输出强制要求
严格输出 JSON 对象，格式如下：

### 正确示例（存在摘要文件）
{{"exist_summary_file": true, "summary_file_rel_path": ".ai_file_summary/summary.md"}}

### 正确示例（不存在摘要文件）
{{"exist_summary_file": false, "summary_file_rel_path": ""}}

### JSON字段定义
1. exist_summary_file: bool - 是否存在摘要文件
2. summary_file_rel_path: str - 摘要文件的相对路径（如果存在）

【项目目录树】
{PROJECT_TREE}

⚠️ summary_file_rel_path 必须是相对于项目根目录的路径，不要包含额外的顶层目录。

硬性规则：只输出纯净 JSON，不要添加任何解释、标记或额外文字。
"""

analyze_change_task_prompt = """
你是项目修改分析师，结合用户修改需求、项目目录树、已经收集完成的全部文件摘要，以及上一轮反馈的信息不足原因、建议补充读取的文件，做出判断。

【用户修改需求】
{USER_CHANGE_REQ}

【项目完整目录树】
{PROJECT_TREE}

【已经收集的文件摘要信息】
{SUMMARY_CONTEXT}

【上一轮信息不足原因】
{INSUFFICIENT_REASON}

【上一轮处理过的文件黑名单】
{PROCESSED_FILE_LIST}

【上一轮建议补充读取的文件列表】
{SUGGEST_ADDITIONAL_FILES}

## 输出格式
严格输出 JSON 对象，格式如下：

### 正确示例（需要读取文件）
{{"need_call_file_list": ["src/cli/main.py", "src/models/book.py", "src/services/book_service.py"]}}

### 正确示例（不需要读取文件）
{{"need_call_file_list": []}}

### 字段说明
- need_call_file_list: list[str] - 需要读取源码的文件路径列表
- ⚠️ 路径必须相对于项目根目录，不要包含额外的顶层目录！

硬性规则：只输出纯净 JSON，不要添加任何解释、标记或额外文字。
"""

file_summary_prompt = """
你是资深Python代码分析专家，仅分析传入的这一个源码文件，输出单个JSON字典对象。

## 正确示例
{{"rel_path": "src/models/book.py", "file_desc": "图书数据模型，定义Book和BookCreate两个Pydantic模型", "export_apis": ["Book", "BookCreate"], "depend_on": []}}

## 字段说明
1. rel_path: str - 文件相对路径
2. file_desc: str - 文件职责描述（一句话概括）
3. export_apis: list[str] - 对外导出的类/函数名称列表
4. depend_on: list[str] - 依赖的其他文件路径列表（无依赖写空数组）

【文件相对路径】
{REL_PATH}

【文件完整源码】
{SOURCE_CODE}

硬性约束：只输出纯净单行JSON，不要添加任何解释、标记或额外文字。
"""

execute_change_code_prompt = """
你是资深Python开发工程师。
输入：用户修改需求、项目完整目录树、已收集的文件摘要、当前已经读取的文件源码、旧版已处理文件黑名单。

【用户修改需求】
{USER_CHANGE_REQ}

【项目完整目录树】
{PROJECT_TREE}

【已收集文件摘要】
{SUMMARY_LIST_JSON}

【旧版已处理文件黑名单】
{PROCESSED_FILE_LIST}

【当前读取到的文件源码】
{FILE_SOURCE_DATA}

## 输出格式
严格输出 JSON 对象，格式如下：

### 正确示例（信息充足，有修改）
{{"is_enough_info": true, "insufficient_reason": "", "need_additional_file_list": [], "changed_files": {{"src/services/book_service.py": "修改后的完整源码内容..."}}, "processed_file_list": ["src/models/book.py"], "inferred_can_unit_test": ["src/models/book.py"], "inferred_top_entry": "src/cli/main.py", "top_entry_confidence": "high"}}

### 正确示例（信息不足）
{{"is_enough_info": false, "insufficient_reason": "项目目录树、文件摘要和源码均为空，无法了解项目结构", "need_additional_file_list": ["src/cli/main.py"], "changed_files": {{}}, "processed_file_list": [], "inferred_can_unit_test": [], "inferred_top_entry": "", "top_entry_confidence": "low"}}

### 正确示例（信息充足但无需修改文件）
{{"is_enough_info": true, "insufficient_reason": "", "need_additional_file_list": [], "changed_files": {{}}, "processed_file_list": ["src/models/book.py"], "inferred_can_unit_test": ["src/models/book.py"], "inferred_top_entry": "src/cli/main.py", "top_entry_confidence": "medium"}}

## JSON字段说明
1. is_enough_info: bool - 信息是否充足
2. insufficient_reason: str - 信息不足的原因（is_enough_info=false时必填）
3. need_additional_file_list: list[str] - 需要额外读取的文件路径列表
4. changed_files: object - 修改的文件，键为文件路径，值为修改后的完整源码
5. processed_file_list: list[str] - 本次已处理过的文件路径列表
6. inferred_can_unit_test: list[str] - 推断可以独立单测的文件路径列表
7. inferred_top_entry: str - 推断的顶层入口文件路径
8. top_entry_confidence: str - 置信度，可选值：high / medium / low

⚠️ 所有文件路径必须相对于项目根目录，不要包含额外的顶层目录！

硬性约束：只输出纯净单行JSON，不要添加任何解释、标记或额外文字。
"""

re_summary_modified_prompt = """
你是Python代码分析专家，针对已经修改完成的单个文件，重新生成该文件最新摘要字典。

## 正确示例
{{"rel_path": "src/services/book_service.py", "file_desc": "图书服务层，实现图书的增删改查和导出功能", "export_apis": ["create_book", "get_book", "list_books", "export_to_excel"], "depend_on": ["src/models/book.py"]}}

## 字段说明
1. rel_path: str - 文件相对路径
2. file_desc: str - 文件职责描述（一句话概括）
3. export_apis: list[str] - 对外导出的类/函数名称列表
4. depend_on: list[str] - 依赖的其他文件路径列表（无依赖写空数组）

【文件相对路径】
{REL_PATH}

【修改完成后的完整源码】
{CHANGED_FILE_SOURCE}

硬性约束：只输出纯净单行JSON，不要添加任何解释、标记或额外文字。
"""


################################################################################################################
# ====================== 图状态定义 ======================
class CheckAndChangeCode(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_change_requirement: str | None
    change_is_ok: bool
    file_name_list: list[str]
    change_content_list: list[str]
    project_file_tree: str
    file_summary_list: list[dict]
    next_file_name_list: list[str]
    user_question: list[str] | None
    loop_count: int
    max_loop: int
    has_local_summary_md: bool
    is_enough_info: bool
    insufficient_reason: str | None
    need_additional_file_list: list[str] | None
    target_file_source_map: dict[str, str] | None
    changed_file_map: dict[str, str] | None
    processed_file_list: list[str]

    inferred_can_unit_test: list[str] | None
    inferred_top_entry: str | None
    top_entry_confidence: str | None
    unit_test_files: list[dict] | None
    top_entry_file: dict | None
    current_test_file: dict | None
    pending_file_code_update: str | None
    raw_sandbox_result: dict | None
    last_test_error: str | None
    test_retry_count: int
    max_test_retry: int
    pending_main_code_update: dict | None
    raw_main_sandbox_result: dict | None
    last_main_error: str | None
    main_retry_count: int
    max_main_retry: int
    is_environment_error: bool | None
    modify_summary: list[dict] | None
    test_phase: str | None
    need_re_evaluate: bool | None
    candidate_top_entries: list[str] | None

    impacted_files: list[str] | None
    ast_validation_errors: list[str] | None

    # 主图作为编译子图节点传入（运行时父图 state dict 含该字段），用于进度实时入库
    conv_id: str | None

    # 内部：连续"读取 0 个文件"轮次（熔断 5 次死循环）
    _zero_read_streak: int | None


################################################################################################################
# ====================== 工具函数 ======================
def scan_project_tree_os(root_dir: str):
    skip_dirs = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "build", "dist", ".ai_file_summary",
                 ".modify_backup"}
    lines = []
    root = os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        depth = dirpath.replace(root, "").count(os.sep)
        indent = "  " * depth
        dir_name = os.path.basename(dirpath)
        lines.append(f"{indent}- {dir_name}/")
        sub_indent = "  " * (depth + 1)
        for fname in filenames:
            lines.append(f"{sub_indent}- {fname}")
    return "\n".join(lines)


def _strip_file_marker(src: str) -> str:
    src = re.sub(r"^===== FILE_PATH:.+? =====\n", "", src, count=1)
    src = re.sub(r"\n===== END_FILE_PATH:.+? =====$", "", src, count=1)
    return src


def _normalize_path(path: str) -> str:
    """标准化路径：将 Windows 反斜杠转为正斜杠"""
    return path.replace("\\", "/")


def _remove_top_dir_if_present(path: str, top_dir: str) -> str:
    """如果路径以顶层目录名开头，去掉它（精确按路径边界匹配）"""
    if not path or not top_dir:
        return path
    # 精确 3 类边界：目录分隔符 / 或 \；或 top_dir 后面是扩展分隔符且前还有目录分隔符（绝不允许 "crawler.py" 被当成 top_dir='crawler' + '.py'）
    patterns = [
        f"{top_dir}/",
        f"{top_dir}\\",
    ]
    for p in patterns:
        if path.startswith(p):
            return path[len(p):]
    # 处理 "crawler.crawler.py" 这类"目录+点"的 LLM 错误写法：仅当路径里"top_dir."前面没有任何目录分隔符（即 top_dir 真的是路径第一段）时才去
    if path.startswith(f"{top_dir}.") and os.sep not in path and "/" not in path:
        # 兜底：如果去掉 top_dir + "." 后是一个"看起来正常的文件名"（含扩展名或含别的.）才切
        _rest = path[len(top_dir) + 1:]
        if "." in _rest and len(_rest) > 3:
            return _rest
    return path


def _normalize_path_list(paths: List[str], top_dir: str) -> List[str]:
    """标准化路径列表：去掉重复的顶层目录，转换分隔符"""
    result = []
    for p in paths:
        if not p:
            continue
        p = _normalize_path(p)
        p = _remove_top_dir_if_present(p, top_dir)
        result.append(p)
    return result


def _fix_path_for_project(path: str) -> str:
    """根据项目实际结构修正路径"""
    if not path:
        return path
    path = _normalize_path(path)
    path = _remove_top_dir_if_present(path, PROJECT_TOP_DIR)

    # ★★★ 先按"实际存在的文件"做模糊匹配：拼到 absolute_project_path 下，如果存在就不做硬映射（更通用） ★★★
    # 背景：原来的 main.py → src/cli/main.py 是图书管理系统的硬编码；crawler 项目里 crawler.py 就在根目录，不需要映射
    _abs_candidate = os.path.abspath(os.path.join(absolute_project_path, path))
    if os.path.isfile(_abs_candidate):
        return path
    # 兜底尝试：去掉一层 "crawler/" 前缀（crawler.py 就存在 crawler/crawler.py → 但 absolute 是 crawler，所以这是去重层）
    for _p in (path, os.path.join(os.path.basename(absolute_project_path), path)):
        _abs = os.path.abspath(os.path.join(absolute_project_path, _p))
        if os.path.isfile(_abs):
            return _p

    # ★★★ 路径修正映射（无法找到对应文件时，仅针对配置里 change_project_path 如果包含图书管理系统结构才生效） ★★★
    # （crawler/web_scraper 这类项目不会命中，保持原样）
    _HAS_SRC_CLI = os.path.isdir(os.path.join(absolute_project_path, "src", "cli"))
    if _HAS_SRC_CLI:
        if path == "src/main.py":
            log_info("子图2", "路径修正", f"src/main.py → src/cli/main.py（图书项目结构）")
            return "src/cli/main.py"
        if path == "main.py":
            log_info("子图2", "路径修正", f"main.py → src/cli/main.py（图书项目结构）")
            return "src/cli/main.py"

    return path


def _fix_path_list_for_project(paths: List[str]) -> List[str]:
    """对路径列表应用路径修正"""
    # ★★★ 已知裸后缀集合（LLM 偶尔会把 `.py` / `py` 当独立条目返回，或非文件对象/数字 → 全部过滤掉防 "文件不存在: py" 这类误导报错） ★★★
    _KNOWN_SUFFIXES = {
        "py", "pyi", "pyx", "sh", "bash", "zsh", "bat", "cmd", "ps1", "json", "yaml", "yml", "toml",
        "ini", "cfg", "conf", "txt", "md", "sql", "db", "sqlite", "sqlite3", "csv", "tsv",
        "html", "htm", "css", "scss", "less", "js", "mjs", "cjs", "jsx", "ts", "tsx", "vue", "svelte",
        "java", "kt", "go", "rs", "c", "cpp", "h", "hpp", "cs", "rb", "php", "swift", "dart", "lua",
        "r", "m", "mm", "pl", "pm", "scala", "clj", "ex", "exs", "erl", "hs", "ml",
        "png", "jpg", "jpeg", "gif", "svg", "ico", "pdf", "xlsx", "xls", "docx", "doc", "ppt", "pptx",
        "zip", "tar", "gz", "bz2", "7z", "rar",
        "lock", "example", "dockerfile", "makefile", "gitignore", "env",
        "python",  # 偶尔被当成 ".Python" 或 "python" 单字
    }

    # 顺序：先做 fix_path_for_project（有 hard-coded 映射如 main.py → src/cli/main.py），再做"裸后缀判断+去重"。
    # 这样映射后变成路径的条目不被误杀为裸后缀。
    rough = []
    for p in paths:
        if not p:
            continue
        if not isinstance(p, str):
            continue
        p_text = str(p).strip()
        if not p_text:
            continue
        fixed = _fix_path_for_project(p_text)
        if fixed:
            rough.append(fixed)

    result = []
    seen = set()
    for p in rough:
        if p in seen:
            continue
        # 过滤"看起来像文件后缀本身"的条目：既不含路径分隔符，去掉前导 . 后正好命中 _KNOWN_SUFFIXES
        low_no_dot = p.lower().lstrip(".")
        if ("/" not in p and "\\" not in p
                and low_no_dot in _KNOWN_SUFFIXES
                and len(low_no_dot) <= 12):
            continue
        seen.add(p)
        result.append(p)
    return result


################################################################################################################
# ====================== 节点1：read_local_project_tree_node ======================
def read_local_project_tree_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "扫描目录", "开始扫描项目结构...")
    the_project_tree = scan_project_tree_os(str(absolute_project_path))
    log_success("子图2", "扫描目录", f"扫描完成，项目路径: {absolute_project_path}")
    return {"project_file_tree": the_project_tree}


################################################################################################################
# ====================== 节点2：detect_exist_summary_md_node ======================
def detect_exist_summary_md_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "检测摘要", "检查是否存在摘要缓存...")
    tree_text = state["project_file_tree"]
    prompt = safe_format(detect_summary_md_prompt, PROJECT_TREE=tree_text)
    new_has_md = False
    merge_summary = state.get("file_summary_list", []).copy()

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    light_retry = model_output_retry_cfg.get("light_retry", default_retry)
    detect_summary_retry = get_retry_count_for_node("detect_summary", light_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_analyze_change,
            prompt=prompt,
            pydantic_model=DetectSummaryOutput,
            max_retries=detect_summary_retry,
            node_name="摘要检测",
            validator_context="输出必须包含 exist_summary_file (bool)、summary_file_rel_path (str) 两个字段。"
        )
        exist = validated.exist_summary_file
        rel_path = validated.summary_file_rel_path
        # ★★★ 标准化路径 ★★★
        rel_path = _normalize_path(rel_path)
        rel_path = _remove_top_dir_if_present(rel_path, PROJECT_TOP_DIR)
        log_info("子图2", "检测摘要", f"exist={exist}, rel_path={rel_path}")
    except RuntimeError as e:
        log_warn("子图2", "检测摘要", f"失败: {e}，认为不存在摘要文件")
        return {
            "has_local_summary_md": False,
            "file_summary_list": merge_summary
        }

    if exist and not rel_path.endswith(".md"):
        exist = False

    if exist:
        real_md_path = os.path.join(absolute_project_path, rel_path)
        real_md_path = os.path.abspath(real_md_path)
        if os.path.exists(real_md_path):
            new_has_md = True
            loaded_summary_arr = parse_local_summary_md(real_md_path)
            exist_map = {item["rel_path"]: item for item in merge_summary}
            for s in loaded_summary_arr:
                # ★★★ 标准化摘要中的路径 ★★★
                s_rel = _normalize_path(s.get("rel_path", ""))
                s_rel = _remove_top_dir_if_present(s_rel, PROJECT_TOP_DIR)
                s["rel_path"] = s_rel
                exist_map[s_rel] = s
            merge_summary = list(exist_map.values())
            log_success("子图2", "检测摘要", f"加载缓存成功，共 {len(loaded_summary_arr)} 条摘要")
        else:
            log_warn("子图2", "检测摘要", f"摘要文件不存在: {real_md_path}")

    return {
        "has_local_summary_md": new_has_md,
        "file_summary_list": merge_summary
    }


################################################################################################################
# ====================== 节点3：analyze_task_node ======================
def analyze_task_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "任务分析", "分析需要读取的文件...")
    conv_id = state.get("conv_id")
    user_req = state.get("user_change_requirement", "")
    tree_text = state.get("project_file_tree", "")
    old_summary_list = state.get("file_summary_list", [])
    insufficient_reason = state.get("insufficient_reason")
    suggest_add_files = state.get("need_additional_file_list") or []
    processed_file_list = state.get("processed_file_list") or []
    impacted_files = state.get("impacted_files") or []
    ast_validation_errors = state.get("ast_validation_errors") or []
    loop_count = state.get("loop_count", 0)

    if loop_count == 0:
        notify_progress(conv_id, "🔄 [修改模式] 分析需求中...（第 1 轮，约 1~3 分钟）")
    else:
        notify_progress(conv_id, f"🔄 [修改模式] 补充分析中（第 {loop_count + 1} 轮）...")

    summary_context = json.dumps(old_summary_list, ensure_ascii=False)
    ins_reason_text = insufficient_reason if insufficient_reason is not None else ""
    suggest_file_text = json.dumps(suggest_add_files, ensure_ascii=False)

    prompt = safe_format(
        analyze_change_task_prompt,
        USER_CHANGE_REQ=user_req,
        PROJECT_TREE=tree_text,
        SUMMARY_CONTEXT=summary_context,
        INSUFFICIENT_REASON=ins_reason_text,
        SUGGEST_ADDITIONAL_FILES=suggest_file_text,
        PROCESSED_FILE_LIST=json.dumps(processed_file_list, ensure_ascii=False)
    )
    try:
        resp = model_analyze_change.invoke(prompt)
        raw_json = clean_json_data(resp.content)
        result = json.loads(raw_json)
    except Exception as e:
        log_error("子图2", "任务分析", f"模型调用异常: {e}")
        result = {"need_call_file_list": []}

    next_list = result.get("need_call_file_list", [])

    # ★★★ 核心修复1：标准化路径 ★★★
    next_list = _normalize_path_list(next_list, PROJECT_TOP_DIR)

    # ★★★ 核心修复2：根据项目实际结构修正路径 ★★★
    next_list = _fix_path_list_for_project(next_list)

    # ★★★ 标准化 existed_rel / processed_set / suggest_add_files 路径，避免 Windows 分隔符导致的命中失效 ★★★
    existed_rel = {
        _fix_path_for_project(_remove_top_dir_if_present(_normalize_path(item.get("rel_path", "")), PROJECT_TOP_DIR))
        for item in old_summary_list
    }
    processed_set = {
        _fix_path_for_project(_remove_top_dir_if_present(_normalize_path(p), PROJECT_TOP_DIR))
        for p in processed_file_list
    }
    suggest_norm = _normalize_path_list(suggest_add_files, PROJECT_TOP_DIR)
    suggest_norm = _fix_path_list_for_project(suggest_norm)
    impacted_files_norm = _normalize_path_list(impacted_files, PROJECT_TOP_DIR)
    impacted_files_norm = _fix_path_list_for_project(impacted_files_norm)
    impacted_set = set(impacted_files_norm)

    # ★★★ BUG修复A：将 suggest_add_files（上一轮明确建议补充的文件）强制加入 next_list，
    # 即使它们在 processed 或 existed_rel 中，因为上一轮的修改可能失败了需要重审
    force_read_set = set(suggest_norm)
    # 去重、保持建议顺序
    merged_next = list(dict.fromkeys(suggest_norm + next_list))
    next_list = merged_next

    # ★★★ 默认过滤：去掉已存在摘要 + 已处理黑名单的文件
    # 但 force_read_set 中的文件不过滤（重试场景必须重新读取）
    def _should_keep(p: str) -> bool:
        if p in force_read_set:
            return True
        if p in existed_rel:
            return False
        if p in processed_set:
            return False
        return True

    next_list = [p for p in next_list if _should_keep(p)]

    # ★★★ impacted_files 同样按 force_read 语义豁免过滤
    if impacted_set:
        impacted_remaining = []
        for f in impacted_files_norm:
            if f in force_read_set:
                impacted_remaining.append(f)
            elif f not in processed_set and f not in existed_rel:
                impacted_remaining.append(f)
        impacted_remaining = list(dict.fromkeys(impacted_remaining))
        if impacted_remaining:
            log_info("子图2", "任务分析", f"检测到 {len(impacted_remaining)} 个受影响的文件需要重新分析")
            next_list = list(dict.fromkeys(impacted_remaining + next_list))

    # ★★★ BUG修复B：0文件兜底——彻底打破死循环 ★★★
    if not next_list:
        fallback_files = set()

        # 兜底1：从 ast_validation_errors 中提取涉事文件
        for err_msg in ast_validation_errors:
            if ":" in err_msg:
                candidate = err_msg.split(":", 1)[0].strip()
                if candidate:
                    candidate = _normalize_path(candidate)
                    candidate = _remove_top_dir_if_present(candidate, PROJECT_TOP_DIR)
                    candidate = _fix_path_for_project(candidate)
                    if candidate:
                        fallback_files.add(candidate)

        # 兜底2：如果用户有明确需求，扫描目录树中最核心的文件（models/services/cli/main.py）
        if not fallback_files and insufficient_reason and loop_count > 0:
            # 根据目录树猜测核心文件
            keywords_priority = [
                "main.py", "app.py", "cli.py",
                "book_service.py", "service.py", "services.py",
                "book.py", "model.py", "models.py",
                "router.py", "route.py",
            ]
            tree_lines = tree_text.splitlines()
            for line in tree_lines:
                line_clean = line.strip().lstrip("- ").strip()
                for kw in keywords_priority:
                    if line_clean.endswith(kw):
                        rel = line_clean.replace("\\", "/")
                        rel = _remove_top_dir_if_present(rel, PROJECT_TOP_DIR)
                        rel = _fix_path_for_project(rel)
                        # 排除目录只保留文件
                        if rel and not rel.endswith("/") and os.path.isfile(
                            os.path.join(absolute_project_path, rel)
                        ):
                            fallback_files.add(rel)
                            if len(fallback_files) >= 3:
                                break
                if len(fallback_files) >= 3:
                    break

        # 兜底3：如果 loop_count >= 1 且仍没有文件，强制放宽 processed_file_list 拦截，允许读取 processed 中所有文件（摘要已作废）
        if not fallback_files and loop_count >= 1:
            log_warn("子图2", "任务分析",
                     f"0文件兜底：放宽 processed_file_list 拦截（{len(processed_set)} 个候选），重新允许读取")
            for p in processed_set:
                abs_p = os.path.join(absolute_project_path, p)
                if os.path.isfile(abs_p):
                    fallback_files.add(p)

        fallback_list = sorted(fallback_files)
        if fallback_list:
            log_warn("子图2", "任务分析",
                     f"0文件兜底生效：强制加载 {len(fallback_list)} 个文件 {fallback_list}")
            next_list = fallback_list

    # ★★★ 安全去重保持顺序 ★★★
    next_list = list(dict.fromkeys(next_list))
    # 路径逃逸过滤
    next_list = [p for p in next_list if p]

    log_info("子图2", "任务分析", f"需要读取 {len(next_list)} 个文件")
    if next_list:
        notify_progress(conv_id, f"📂 [修改模式] 计划读取 {len(next_list)} 个文件源码 + 生成摘要...")
    return {
        "next_file_name_list": next_list,
        "insufficient_reason": None,
        "need_additional_file_list": None,
        "target_file_source_map": None,
        "impacted_files": [],
        "ast_validation_errors": []  # ★ 消费掉 ast_errors，避免下一轮兜底继续触发
    }


################################################################################################################
# ====================== 节点4：read_target_file_node ======================
def read_target_file_node(state: CheckAndChangeCode) -> dict:
    rel_path_list = state["next_file_name_list"]
    source_map = {}
    base_abs = os.path.abspath(absolute_project_path)
    log_info("子图2", "读取文件", f"读取 {len(rel_path_list)} 个文件...")

    for rel in rel_path_list:
        if not rel:
            continue

        # ★★★ 标准化路径 ★★★
        rel = _normalize_path(rel)
        rel = _remove_top_dir_if_present(rel, PROJECT_TOP_DIR)
        rel = _fix_path_for_project(rel)

        full_p = os.path.join(absolute_project_path, rel)
        full_p = os.path.abspath(full_p)

        if not full_p.startswith(base_abs):
            log_warn("子图2", "读取文件", f"路径逃逸拦截: {rel}")
            continue

        # ★★★ 如果文件不存在，尝试去掉顶层目录再找一次（兜底） ★★★
        if not os.path.isfile(full_p):
            rel_clean = _remove_top_dir_if_present(rel, PROJECT_TOP_DIR)
            if rel_clean != rel:
                full_p_retry = os.path.join(absolute_project_path, rel_clean)
                full_p_retry = os.path.abspath(full_p_retry)
                if os.path.isfile(full_p_retry):
                    log_info("子图2", "读取文件", f"路径重试成功: {rel} → {rel_clean}")
                    full_p = full_p_retry
                    rel = rel_clean

        if os.path.isfile(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    src = f.read()
                wrapped_src = f"===== FILE_PATH:{rel} =====\n{src}\n===== END_FILE_PATH:{rel} ====="
                source_map[rel] = wrapped_src
                log_info("子图2", "读取文件", f"✅ 读取成功: {rel}")
            except Exception as e:
                log_error("子图2", "读取文件", f"读取失败 {rel}: {e}")
        else:
            log_warn("子图2", "读取文件", f"文件不存在: {rel}")

    log_success("子图2", "读取文件", f"成功读取 {len(source_map)} 个文件")

    # ★★★ 熔断：读取 0 个文件时计数 + 提前跳出循环（对齐用户贴的"5次死循环烧钱"）★★★
    zero_streak = 0 if len(source_map) > 0 else int(state.get("_zero_read_streak") or 0) + 1
    loop_count = state.get("loop_count", 0)
    max_loop = state.get("max_loop", 5)
    if len(source_map) == 0 and zero_streak >= 2 and loop_count + 1 >= max_loop - 1:
        notify_progress(state.get("conv_id"),
                        f"⚠️ [修改模式] 连续 {zero_streak} 轮没读到任何新文件，正在整理结果（为了避免死循环提前结束）...")
    elif len(source_map) == 0:
        notify_progress(state.get("conv_id"),
                        f"⚠️ [修改模式] 本轮没有读到新文件（可能模型请求的文件不存在），正在尝试重新分析...")
    else:
        notify_progress(state.get("conv_id"),
                        f"🧠 [修改模式] 读取完成，生成 {len(source_map)} 个文件的 AI 摘要 + 执行修改中...")

    return {
        "target_file_source_map": source_map,
        "_zero_read_streak": zero_streak,
    }


################################################################################################################
# ====================== 节点5：generate_file_summary_node ======================
def generate_file_summary_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "生成摘要", "开始生成文件摘要...")
    old_summaries = state.get("file_summary_list", [])
    source_map = state.get("target_file_source_map") or {}
    exist_map = {item["rel_path"]: item for item in old_summaries}

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    normal_retry = model_output_retry_cfg.get("normal_retry", default_retry)
    file_summary_retry = get_retry_count_for_node("file_summary", normal_retry, model_output_retry_cfg)

    success_count = 0
    cache_hit_count = 0

    for rel_path, src in source_map.items():
        # ★★★ 标准化路径 ★★★
        rel_path = _normalize_path(rel_path)
        rel_path = _remove_top_dir_if_present(rel_path, PROJECT_TOP_DIR)
        rel_path = _fix_path_for_project(rel_path)

        cached = get_cached_summary(rel_path, src)
        if cached:
            cached["rel_path"] = _normalize_path(cached.get("rel_path", ""))
            cached["rel_path"] = _remove_top_dir_if_present(cached["rel_path"], PROJECT_TOP_DIR)
            cached["rel_path"] = _fix_path_for_project(cached["rel_path"])
            exist_map[rel_path] = cached
            cache_hit_count += 1
            log_success("子图2", "缓存命中", f"✅ {rel_path} 摘要复用（省去 1 次 LLM 调用）")
            continue

        pmt = safe_format(file_summary_prompt, REL_PATH=rel_path, SOURCE_CODE=src)

        try:
            validated = invoke_with_validation(
                model=model_file_summary,
                prompt=pmt,
                pydantic_model=FileSummaryOutput,
                max_retries=file_summary_retry,
                node_name=f"摘要({rel_path})",
                logic_validator=validate_file_summary_logic,
                validator_context="输出必须包含 rel_path (str)、file_desc (str)、export_apis (list[str])、depend_on (list[str]) 四个字段。"
            )
            validated_dict = validated.dict()
            # ★★★ 标准化路径 ★★★
            validated_dict["rel_path"] = _normalize_path(validated_dict.get("rel_path", ""))
            validated_dict["rel_path"] = _remove_top_dir_if_present(validated_dict["rel_path"], PROJECT_TOP_DIR)
            validated_dict["rel_path"] = _fix_path_for_project(validated_dict["rel_path"])
            exist_map[rel_path] = validated_dict
            success_count += 1

            set_cached_summary(rel_path, src, validated_dict)
            log_success("子图2", "缓存写入", f"💾 {rel_path} 摘要已缓存")

        except RuntimeError as e:
            log_warn("子图2", "生成摘要", f"跳过 {rel_path}: {e}")
            continue

    new_summary_list = list(exist_map.values())
    log_info("子图2", "生成摘要",
             f"完成：成功 {success_count} 个（新生成），缓存命中 {cache_hit_count} 个，总计 {len(source_map)} 个文件")
    return {"file_summary_list": new_summary_list}


################################################################################################################
# ====================== 节点6：execute_change_node ======================
def execute_change_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "执行修改", "开始执行代码修改...")
    user_req = state.get("user_change_requirement", "")
    tree_text = state.get("project_file_tree", "")
    summary_list = state.get("file_summary_list", [])
    source_map = state.get("target_file_source_map") or {}
    processed_file_list = state.get("processed_file_list") or []

    prompt = safe_format(
        execute_change_code_prompt,
        USER_CHANGE_REQ=user_req,
        PROJECT_TREE=tree_text,
        SUMMARY_LIST_JSON=json.dumps(summary_list, ensure_ascii=False),
        FILE_SOURCE_DATA=json.dumps(source_map, ensure_ascii=False),
        PROCESSED_FILE_LIST=json.dumps(processed_file_list, ensure_ascii=False)
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    normal_retry = model_output_retry_cfg.get("normal_retry", default_retry)
    change_code_retry = get_retry_count_for_node("change_code", normal_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_code_change,
            prompt=prompt,
            pydantic_model=ChangeCodeOutput,
            max_retries=change_code_retry,
            node_name="代码修改",
            logic_validator=validate_change_code_logic,
            validator_context="输出必须包含 is_enough_info (bool)、insufficient_reason (str)、need_additional_file_list (list[str])、changed_files (dict)、processed_file_list (list[str])、inferred_can_unit_test (list[str])、inferred_top_entry (str)、top_entry_confidence (str) 等字段。"
        )
        change_result = validated.dict()
        log_success("子图2", "执行修改", "输出校验通过")
    except RuntimeError as e:
        log_warn("子图2", "执行修改", f"输出校验失败，降级为信息不足: {e}")
        change_result = {
            "is_enough_info": False,
            "insufficient_reason": f"模型输出校验失败: {str(e)}",
            "need_additional_file_list": [],
            "changed_files": {},
            "processed_file_list": processed_file_list,
            "inferred_can_unit_test": [],
            "inferred_top_entry": "",
            "top_entry_confidence": "low"
        }

    is_enough = change_result.get("is_enough_info", False)
    ins_reason = change_result.get("insufficient_reason", "")
    add_file_list = change_result.get("need_additional_file_list", [])
    changed_files = change_result.get("changed_files", {})
    new_processed = change_result.get("processed_file_list", processed_file_list)
    inferred_unit = change_result.get("inferred_can_unit_test", [])
    inferred_top = change_result.get("inferred_top_entry", "")
    top_conf = change_result.get("top_entry_confidence", "low")

    # ★★★ 标准化所有路径 ★★★
    normalized_changed = {}
    for k, v in changed_files.items():
        k = _normalize_path(k)
        k = _remove_top_dir_if_present(k, PROJECT_TOP_DIR)
        k = _fix_path_for_project(k)
        normalized_changed[k] = v
    changed_files = normalized_changed

    add_file_list = _normalize_path_list(add_file_list, PROJECT_TOP_DIR)
    add_file_list = _fix_path_list_for_project(add_file_list)
    new_processed = _normalize_path_list(new_processed, PROJECT_TOP_DIR)
    new_processed = _fix_path_list_for_project(new_processed)
    inferred_unit = _normalize_path_list(inferred_unit, PROJECT_TOP_DIR)
    inferred_unit = _fix_path_list_for_project(inferred_unit)
    inferred_top = _normalize_path(inferred_top)
    inferred_top = _remove_top_dir_if_present(inferred_top, PROJECT_TOP_DIR)
    inferred_top = _fix_path_for_project(inferred_top)

    # ========== AST 契约校验 ==========
    # ★★★ HAS_SCHEMA 策略开关 ★★★
    #  - 本系统 complete 模式生成的项目（有 Schema）→ 严格执行 AST 契约校验，失败即回退重试
    #  - 普通已有项目（无 Schema）→ export_apis 是 LLM 猜测结果，跳过校验避免死循环
    ast_validation_errors = []
    if HAS_SCHEMA and is_enough and changed_files:
        summary_map = {s["rel_path"]: s for s in summary_list}
        for file_path, code in changed_files.items():
            original_item = summary_map.get(file_path)
            if original_item:
                expected_apis = original_item.get("export_apis", [])

                # ★★★ 核心修复：统一将 export_apis 转换为字典列表格式 ★★★
                if expected_apis:
                    if isinstance(expected_apis, list) and len(expected_apis) > 0 and isinstance(expected_apis[0], str):
                        converted_apis = []
                        for api_name in expected_apis:
                            converted_apis.append({
                                "name": api_name,
                                "type": "函数",
                                "params": [],
                                "return_desc": "",
                                "exception": "无"
                            })
                        expected_apis = converted_apis
                        log_info("子图2", "AST校验", f"export_apis 格式转换: {len(expected_apis)} 个接口")

                    ast_errors = validate_code_against_schema(code, expected_apis)
                    if ast_errors:
                        for err in ast_errors:
                            ast_validation_errors.append(f"{file_path}: {err}")
                        log_warn("子图2", "AST校验", f"❌ {file_path} 修改后接口契约不匹配")
                else:
                    actual_exports = list(extract_exported_apis_from_code(code).keys())
                    if actual_exports:
                        ast_validation_errors.append(
                            f"{file_path}: 原文件无对外导出接口，但修改后新增了接口: {actual_exports}"
                        )
                        log_warn("子图2", "AST校验", f"❌ {file_path} 新增了未声明的接口")

        if ast_validation_errors:
            log_warn("子图2", "执行修改", f"AST 校验发现 {len(ast_validation_errors)} 个问题")
            # ★★★ BUG修复：AST校验失败 → 本次修改作废，需从 processed 黑名单 + 摘要缓存中移除涉事文件
            # 从错误信息中解析出涉及的文件路径（冒号前为文件路径）
            ast_error_files = set()
            for err_msg in ast_validation_errors:
                if ":" in err_msg:
                    candidate = err_msg.split(":", 1)[0].strip()
                    if candidate:
                        candidate = _normalize_path(candidate)
                        candidate = _remove_top_dir_if_present(candidate, PROJECT_TOP_DIR)
                        candidate = _fix_path_for_project(candidate)
                        ast_error_files.add(candidate)
            # 同时以 changed_files.keys() 作为兜底来源
            changed_keys_norm = {
                _fix_path_for_project(_remove_top_dir_if_present(_normalize_path(k), PROJECT_TOP_DIR))
                for k in changed_files.keys()
            }
            ast_related_files = ast_error_files | changed_keys_norm

            # 1. 从 processed_file_list 中清除，允许下次 analyze_task_node 重新选择这些文件
            cleaned_processed = [
                _fix_path_for_project(_remove_top_dir_if_present(_normalize_path(p), PROJECT_TOP_DIR))
                for p in new_processed
            ]
            cleaned_processed = [p for p in cleaned_processed if p not in ast_related_files]

            # 2. 从 file_summary_list 中移除对应摘要（原摘要基于修改前代码生成，且本次修改未写入）
            old_summaries = state.get("file_summary_list", [])
            cleaned_summaries = [
                s for s in old_summaries
                if _fix_path_for_project(_remove_top_dir_if_present(
                    _normalize_path(s.get("rel_path", "")), PROJECT_TOP_DIR
                )) not in ast_related_files
            ]
            if len(old_summaries) != len(cleaned_summaries):
                log_info("子图2", "执行修改",
                         f"AST失败：从摘要缓存中移除 {len(old_summaries) - len(cleaned_summaries)} 个涉事文件摘要")

            # 3. 将涉事文件作为 need_additional_file_list，提示 analyze_task_node 强制读取
            suggest_reload = list(ast_related_files)

            return {
                "is_enough_info": False,
                "insufficient_reason": f"AST 契约校验失败: {'; '.join(ast_validation_errors[:3])}",
                "need_additional_file_list": suggest_reload,  # ★ 强制建议重新读取
                "changed_file_map": {},
                "file_name_list": [],
                "change_content_list": [],
                "processed_file_list": cleaned_processed,       # ★ 黑名单清洗
                "file_summary_list": cleaned_summaries,         # ★ 摘要清洗
                "inferred_can_unit_test": inferred_unit,
                "inferred_top_entry": inferred_top,
                "top_entry_confidence": top_conf,
                "ast_validation_errors": ast_validation_errors,
                "impacted_files": suggest_reload                # ★ 触发 impacted_files 强制加载逻辑
            }
    elif (not HAS_SCHEMA) and is_enough and changed_files:
        log_info("子图2", "AST校验", "HAS_SCHEMA=False → 跳过 AST 契约校验（无 Schema 背书的普通项目）")

    if is_enough:
        log_success("子图2", "执行修改", f"修改了 {len(changed_files)} 个文件")
    else:
        log_info("子图2", "执行修改", f"信息不足: {ins_reason[:50]}...")

    return {
        "is_enough_info": is_enough,
        "insufficient_reason": ins_reason,
        "need_additional_file_list": add_file_list,
        "changed_file_map": changed_files,
        "file_name_list": list(changed_files.keys()),
        "change_content_list": list(changed_files.values()),
        "processed_file_list": new_processed,
        "inferred_can_unit_test": inferred_unit,
        "inferred_top_entry": inferred_top,
        "top_entry_confidence": top_conf,
        "ast_validation_errors": ast_validation_errors
    }


################################################################################################################
# ====================== 节点7：analyze_impact_node ======================
def analyze_impact_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "影响分析", "开始分析修改影响...")
    changed_files = state.get("changed_file_map", {}).keys()
    summary_list = state.get("file_summary_list", [])
    processed_file_list = state.get("processed_file_list", [])

    impacted_files = set()

    depend_on_index = {}
    for item in summary_list:
        rel_path = item.get("rel_path", "")
        depend_on = item.get("depend_on", [])
        for dep in depend_on:
            dep = _normalize_path(dep)
            dep = _remove_top_dir_if_present(dep, PROJECT_TOP_DIR)
            dep = _fix_path_for_project(dep)
            if dep not in depend_on_index:
                depend_on_index[dep] = []
            if rel_path not in depend_on_index[dep]:
                depend_on_index[dep].append(rel_path)

    for changed_file in changed_files:
        changed_file = _normalize_path(changed_file)
        changed_file = _remove_top_dir_if_present(changed_file, PROJECT_TOP_DIR)
        changed_file = _fix_path_for_project(changed_file)
        if changed_file in depend_on_index:
            for dependent in depend_on_index[changed_file]:
                if dependent not in changed_files and dependent not in processed_file_list:
                    impacted_files.add(dependent)

    if impacted_files:
        log_info("子图2", "影响分析", f"发现 {len(impacted_files)} 个受影响的文件需要重新检查: {list(impacted_files)}")
    else:
        log_info("子图2", "影响分析", "没有发现受影响的文件")

    impacted_files = _normalize_path_list(list(impacted_files), PROJECT_TOP_DIR)
    impacted_files = _fix_path_list_for_project(list(impacted_files))

    return {
        "impacted_files": list(impacted_files),
        "next_file_name_list": list(impacted_files)
    }


################################################################################################################
# ====================== 节点8：inc_loop_count_node ======================
def inc_loop_count_node(state: CheckAndChangeCode) -> dict:
    new_count = state.get("loop_count", 0) + 1
    log_info("子图2", "循环计数", f"第 {new_count} 次循环")
    return {"loop_count": new_count}


################################################################################################################
# ====================== 节点9：re_summary_modified_node ======================
def re_summary_modified_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "刷新摘要", "开始刷新修改文件的摘要...")
    change_result_dict = state.get("changed_file_map") or {}
    summary_list = state.get("file_summary_list", []).copy()
    summary_map = {s["rel_path"]: s for s in summary_list}

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    normal_retry = model_output_retry_cfg.get("normal_retry", default_retry)
    file_summary_retry = get_retry_count_for_node("file_summary", normal_retry, model_output_retry_cfg)

    success_count = 0
    for rel_path, new_src in change_result_dict.items():
        rel_path = _normalize_path(rel_path)
        rel_path = _remove_top_dir_if_present(rel_path, PROJECT_TOP_DIR)
        rel_path = _fix_path_for_project(rel_path)
        pmt = safe_format(re_summary_modified_prompt, REL_PATH=rel_path, CHANGED_FILE_SOURCE=new_src)

        try:
            validated = invoke_with_validation(
                model=model_file_summary,
                prompt=pmt,
                pydantic_model=FileSummaryOutput,
                max_retries=file_summary_retry,
                node_name=f"刷新摘要({rel_path})",
                logic_validator=validate_file_summary_logic,
                validator_context="输出必须包含 rel_path (str)、file_desc (str)、export_apis (list[str])、depend_on (list[str]) 四个字段。"
            )
            validated_dict = validated.dict()
            validated_dict["rel_path"] = _normalize_path(validated_dict.get("rel_path", ""))
            validated_dict["rel_path"] = _remove_top_dir_if_present(validated_dict["rel_path"], PROJECT_TOP_DIR)
            validated_dict["rel_path"] = _fix_path_for_project(validated_dict["rel_path"])
            summary_map[rel_path] = validated_dict
            success_count += 1
            log_success("子图2", "刷新摘要", f"{rel_path} 成功")
        except RuntimeError as e:
            log_warn("子图2", "刷新摘要", f"跳过 {rel_path}，保留旧摘要: {e}")
            continue

    log_info("子图2", "刷新摘要", f"成功 {success_count}/{len(change_result_dict)} 个文件")
    return {"file_summary_list": list(summary_map.values())}


################################################################################################################
# ====================== 节点10：write_file_node ======================
def write_file_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "写入文件", "开始写入修改后的文件...")
    change_result_dict = state.get("changed_file_map") or {}
    modify_backup_dir = os.path.join(absolute_project_path, ".modify_backup")
    os.makedirs(modify_backup_dir, exist_ok=True)
    base_abs = os.path.abspath(absolute_project_path)
    ok_flag = True
    summary_list = state.get("file_summary_list", [])

    for rel_path, new_code in change_result_dict.items():
        rel_path = _normalize_path(rel_path)
        rel_path = _remove_top_dir_if_present(rel_path, PROJECT_TOP_DIR)
        rel_path = _fix_path_for_project(rel_path)
        clean_code = _strip_file_marker(new_code)
        full_path = os.path.join(absolute_project_path, rel_path)
        full_path = os.path.abspath(full_path)
        if not full_path.startswith(base_abs):
            log_warn("子图2", "写入文件", f"路径逃逸拦截: {rel_path}")
            ok_flag = False
            continue

        safe_filename = base64.urlsafe_b64encode(rel_path.encode("utf-8")).decode("utf-8")
        backup_path = os.path.join(modify_backup_dir, safe_filename)
        try:
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    old_code = f.read()
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(old_code)

            parent = os.path.dirname(full_path)
            os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(clean_code)
            log_success("子图2", "写入文件", f"{rel_path} 写入成功")

            if rel_path.endswith('.py'):
                try:
                    code_hash = hashlib.md5(clean_code.encode('utf-8')).hexdigest()
                    file_summary = next((s for s in summary_list if s.get("rel_path") == rel_path), {})
                    summary_text = file_summary.get("file_desc", "修改后的文件")[:80]
                    generated_block = f'\n# @generated: {{"file": "{rel_path}", "summary": "{summary_text}", "hash": "{code_hash}"}}\n'
                    with open(full_path, "a", encoding="utf-8") as f:
                        f.write(generated_block)
                    summary_data = {
                        "rel_path": rel_path,
                        "file_desc": file_summary.get("file_desc", "修改后的文件"),
                        "export_apis": file_summary.get("export_apis", []),
                        "depend_on": file_summary.get("depend_on", []),
                        "llm_summary": summary_text,
                        "hash": code_hash
                    }
                    set_cached_summary(rel_path, clean_code, summary_data)
                    log_info("子图2", "缓存更新", f"💾 {rel_path} 缓存已更新")
                except Exception as e:
                    log_warn("子图2", "缓存更新", f"{rel_path} 缓存更新失败: {e}")

        except Exception as e:
            log_error("子图2", "写入文件", f"{rel_path} 失败: {e}")
            ok_flag = False

    return {"change_is_ok": ok_flag}


################################################################################################################
# ====================== 节点11：collect_test_resources_node ======================
def collect_test_resources_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "收集资源", "开始收集测试资源...")

    inferred_unit_list = state.get("inferred_can_unit_test") or []
    inferred_top = state.get("inferred_top_entry") or ""
    confidence = state.get("top_entry_confidence") or "low"

    # ★★★ 标准化路径 ★★★
    inferred_unit_list = _normalize_path_list(inferred_unit_list, PROJECT_TOP_DIR)
    inferred_unit_list = _fix_path_list_for_project(inferred_unit_list)
    inferred_top = _normalize_path(inferred_top)
    inferred_top = _remove_top_dir_if_present(inferred_top, PROJECT_TOP_DIR)
    inferred_top = _fix_path_for_project(inferred_top)

    valid_unit_list = []
    for rel_path in inferred_unit_list:
        full_path = os.path.join(absolute_project_path, rel_path)
        if os.path.isfile(full_path) and rel_path.endswith(".py"):
            valid_unit_list.append({"file_path": rel_path, "can_unit_test": True, "file_purpose": "推断可单测文件"})
        else:
            log_warn("子图2", "收集资源", f"推断的可单测文件不存在，已忽略: {rel_path}")

    top_entry_obj = None
    if inferred_top and os.path.exists(os.path.join(absolute_project_path, inferred_top)):
        top_entry_obj = {
            "file_path": inferred_top,
            "is_top_entry": True,
            "confidence": confidence,
            "file_purpose": "推断顶层入口"
        }
        log_info("子图2", "收集资源", f"使用大模型推断的顶层入口: {inferred_top} (置信度: {confidence})")
    else:
        found_entries = []
        for entry in COMMON_ENTRY_FILES:
            full_path = os.path.join(absolute_project_path, entry)
            if os.path.isfile(full_path):
                found_entries.append(entry)

        if len(found_entries) == 1:
            top_entry_obj = {
                "file_path": found_entries[0],
                "is_top_entry": True,
                "confidence": "low",
                "auto_discovered": True,
                "file_purpose": "降级策略自动发现顶层入口"
            }
            log_info("子图2", "收集资源", f"降级策略：自动发现顶层入口: {found_entries[0]}")
        elif len(found_entries) > 1:
            log_warn("子图2", "收集资源", f"发现多个候选顶层入口: {found_entries}")
            log_info("子图2", "收集资源", "将回到 analyze_task_node 让大模型重新选择")
            return {
                "test_phase": "collecting",
                "need_re_evaluate": True,
                "candidate_top_entries": found_entries,
                "unit_test_files": valid_unit_list,
                "top_entry_file": None,
                "current_test_file": None,
                "last_test_error": None,
                "pending_file_code_update": None,
                "test_retry_count": 0,
                "raw_sandbox_result": None,
                "is_environment_error": False,
                "modify_summary": state.get("modify_summary") or []
            }
        else:
            log_warn("子图2", "收集资源", "未找到任何顶层入口文件，将跳过主测")
            top_entry_obj = None

    current_file = valid_unit_list[0] if valid_unit_list else None

    log_info("子图2", "收集资源", f"可单测文件: {len(valid_unit_list)} 个")
    if top_entry_obj:
        log_info("子图2", "收集资源", f"顶层入口: {top_entry_obj['file_path']}")

    return {
        "test_phase": "unit_testing",
        "unit_test_files": valid_unit_list,
        "top_entry_file": top_entry_obj,
        "current_test_file": current_file,
        "last_test_error": None,
        "pending_file_code_update": None,
        "test_retry_count": 0,
        "raw_sandbox_result": None,
        "last_main_error": None,
        "main_retry_count": 0,
        "is_environment_error": False,
        "need_re_evaluate": False,
        "candidate_top_entries": None,
        "max_test_retry": UNIT_MAX_RETRY,
        "max_main_retry": MAIN_MAX_RETRY,
        "modify_summary": state.get("modify_summary") or []
    }


################################################################################################################
# ====================== 节点12：unit_test_fix_and_run_node ======================
def unit_test_fix_and_run_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "单测执行", "开始...")
    current_file = state.get("current_test_file")
    last_error = state.get("last_test_error")
    retry_count = state.get("test_retry_count") or 0
    max_retry = state.get("max_test_retry") or UNIT_MAX_RETRY

    if not current_file:
        return {"raw_sandbox_result": {"success": False, "return_code": -1, "stderr": "无测试文件"}}

    f_path = current_file["file_path"]
    f_path = _normalize_path(f_path)
    f_path = _remove_top_dir_if_present(f_path, PROJECT_TOP_DIR)
    f_path = _fix_path_for_project(f_path)
    abs_file = os.path.join(absolute_project_path, f_path)

    if last_error and retry_count < max_retry:
        log_info("子图2", "单测执行", f"文件 {f_path} 第 {retry_count + 1}/{max_retry} 次修复")
        try:
            with open(abs_file, "r", encoding="utf-8") as f:
                original_code = f.read()

            fix_prompt = f"""
你是Python修复专家。以下文件在导入测试中失败：

文件路径：{f_path}
错误信息：{last_error}

--- 原始代码开始 ---
{original_code}
--- 原始代码结束 ---

请修复代码中的问题，确保文件可以被正常导入。
修复要求：
1. 只修复导致报错的部分，不改动无关逻辑
2. 严格保留对外接口签名
3. 只输出修复后的完整代码，不要解释

如果是缺少依赖包、API Key、环境变量等环境问题，直接输出原始代码。
"""
            resp = model_change_test_fix.invoke(fix_prompt)
            fixed_code = resp.content.strip()
            if fixed_code.startswith("```python"):
                fixed_code = fixed_code.removeprefix("```python").removesuffix("```")
            elif fixed_code.startswith("```"):
                fixed_code = fixed_code.removeprefix("```").removesuffix("```")
            fixed_code = fixed_code.strip()

            if fixed_code == original_code:
                log_warn("子图2", "单测执行", f"检测到环境错误，跳过 {f_path}")
                return {
                    "raw_sandbox_result": {"success": False, "return_code": 0, "stdout": "", "stderr": "ENV_ERROR_SKIP"},
                    "pending_file_code_update": None,
                    "is_environment_error": True
                }

            with open(abs_file, "w", encoding="utf-8") as f:
                f.write(fixed_code)
            log_info("子图2", "单测执行", f"修复代码已写入：{f_path}")
            return {
                "pending_file_code_update": fixed_code,
                "test_retry_count": retry_count + 1,
                "last_test_error": None
            }
        except Exception as e:
            log_error("子图2", "单测执行", f"修复失败: {e}")

    try:
        with open(abs_file, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        return {"raw_sandbox_result": {"success": False, "return_code": -1, "stderr": f"读取文件失败：{e}"}}

    sandbox_result = run_single_file_sandbox(source_code)
    log_info("子图2", "单测执行", f"沙箱完成，success={sandbox_result.get('success')}")

    return {
        "raw_sandbox_result": sandbox_result,
        "pending_file_code_update": None
    }


################################################################################################################
# ====================== 节点13：unit_test_judge_node ======================
def unit_test_judge_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "单测判定", "开始...")
    current_file = state.get("current_test_file")
    raw_result = state.get("raw_sandbox_result")
    unit_list = state.get("unit_test_files") or []
    retry_count = state.get("test_retry_count") or 0
    max_retry = state.get("max_test_retry") or UNIT_MAX_RETRY
    modify_summary = state.get("modify_summary") or []

    if not current_file:
        return {}

    f_path = current_file["file_path"]
    f_path = _normalize_path(f_path)
    f_path = _remove_top_dir_if_present(f_path, PROJECT_TOP_DIR)
    f_path = _fix_path_for_project(f_path)

    if raw_result and raw_result.get("stderr") == "ENV_ERROR_SKIP":
        log_warn("子图2", "单测判定", f"环境错误跳过：{f_path}")
        new_list = unit_list[1:] if unit_list else []
        next_file = new_list[0] if new_list else None
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": "环境依赖问题"})
        return {
            "unit_test_files": new_list,
            "current_test_file": next_file,
            "last_test_error": None,
            "test_retry_count": 0,
            "raw_sandbox_result": None,
            "is_environment_error": False,
            "modify_summary": modify_summary
        }

    judge_prompt = safe_format(
        sandbox_judge_prompt,
        file_path=f_path,
        file_purpose=current_file.get("file_purpose", ""),
        return_code=raw_result.get("return_code", -1) if raw_result else -1,
        stdout=raw_result.get("stdout", "") if raw_result else "",
        stderr=raw_result.get("stderr", "") if raw_result else ""
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    judge_retry = model_output_retry_cfg.get("judge_retry", default_retry)
    sandbox_judge_retry = get_retry_count_for_node("sandbox_judge", judge_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_change_test_judge,
            prompt=judge_prompt,
            pydantic_model=SandboxJudgeOutput,
            max_retries=sandbox_judge_retry,
            node_name="单测判定",
            validator_context="输出必须包含 success (bool)、is_environment_error (bool)、brief_reason (str)、error_detail (str) 四个字段。"
        )
        judge_dict = validated.dict()
    except RuntimeError as e:
        log_warn("子图2", "单测判定", f"判定模型失败，保守降级: {e}")
        judge_dict = {
            "success": False,
            "is_environment_error": False,
            "brief_reason": "判定模型校验失败，保守判定为失败",
            "error_detail": str(e)
        }

    success = judge_dict.get("success", False)
    is_env_error = judge_dict.get("is_environment_error", False)
    error_detail = judge_dict.get("error_detail", "测试失败")

    if success:
        log_success("子图2", "单测判定", f"通过：{f_path}")
        new_list = unit_list[1:] if unit_list else []
        next_file = new_list[0] if new_list else None
        modify_summary.append({"file": f_path, "status": "pass", "remark": "单测通过"})
        return {
            "unit_test_files": new_list,
            "current_test_file": next_file,
            "last_test_error": None,
            "test_retry_count": 0,
            "raw_sandbox_result": None,
            "is_environment_error": False,
            "modify_summary": modify_summary
        }
    elif is_env_error:
        log_warn("子图2", "单测判定", f"环境错误跳过：{f_path}")
        new_list = unit_list[1:] if unit_list else []
        next_file = new_list[0] if new_list else None
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": "环境依赖问题"})
        return {
            "unit_test_files": new_list,
            "current_test_file": next_file,
            "last_test_error": None,
            "test_retry_count": 0,
            "raw_sandbox_result": None,
            "is_environment_error": False,
            "modify_summary": modify_summary
        }
    else:
        if retry_count >= max_retry:
            log_warn("子图2", "单测判定", f"达到最大重试次数({max_retry})，跳过：{f_path}")
            new_list = unit_list[1:] if unit_list else []
            next_file = new_list[0] if new_list else None
            modify_summary.append({"file": f_path, "status": "skip_max_retry", "remark": f"重试{max_retry}次仍失败"})
            return {
                "unit_test_files": new_list,
                "current_test_file": next_file,
                "last_test_error": None,
                "test_retry_count": 0,
                "raw_sandbox_result": None,
                "is_environment_error": False,
                "modify_summary": modify_summary
            }

        log_warn("子图2", "单测判定", f"失败，准备重试：{f_path}")
        return {
            "last_test_error": error_detail,
            "raw_sandbox_result": None,
            "is_environment_error": False
        }


################################################################################################################
# ====================== 节点14：main_test_fix_and_run_node ======================
def _build_entry_bootstrap(work_root: str, entry_rel: str) -> str:
    """构造"内联启动入口文件"的 python -c 脚本（本地兜底与 Docker 沙箱共用）。

    注意：必须用标准多行缩进写法。旧版把 try:/except 挤在同一行（"try:f()\\nexcept:..."），
    在 python -c 下是 SyntaxError（单行复合语句不能跨行接 except 子句），会导致非相对导入入口主测直接崩。
    """
    return (
        "import os,sys\n"
        f"_WR={work_root!r}\n"
        "for _p in (_WR, os.path.join(_WR,'src'), '/vendor'):\n"
        "    if _p and _p not in sys.path:\n"
        "        sys.path.insert(0,_p)\n"
        f"_F=os.path.join(_WR,{entry_rel!r})\n"
        "import importlib.util as _U\n"
        "_S=_U.spec_from_file_location('__vibe_entry__', _F)\n"
        "_M=_U.module_from_spec(_S)\n"
        "sys.modules['__main__']=_M\nsys.modules['__vibe_entry__']=_M\n"
        "setattr(_M,'__file__',_F)\nsetattr(_M,'__loader__',_S.loader)\nsetattr(_M,'__package__','')\n"
        "setattr(_M,'__builtins__',__builtins__)\n"
        "try:\n"
        "    _S.loader.exec_module(_M)\n"
        "except SystemExit:\n"
        "    pass\n"
        "except SyntaxError as _e:\n"
        "    print(f'SyntaxError: {_e.msg} ({_e.filename}:{_e.lineno})',file=sys.stderr)\n"
        "    sys.exit(1)\n"
        "except Exception as _e:\n"
        "    import traceback\n"
        "    traceback.print_exc()\n"
        "    sys.exit(1)\n"
    )


def _run_main_entry_locally(f_path: str) -> dict:
    """Docker 不可用时的兜底：用本地 Python 直接在项目目录跑顶层入口。

    与 Docker 沙箱返回结构保持一致：{success, return_code, stdout, stderr}，
    这样下游 main_test_judge_node 无需区分来源，照常判定（真通过就通过，代码错就走修复重试，
    只有本地环境本身缺解释器/超时才算环境问题）。

    安全说明：modify 模式改的本就是用户本机已有的项目，用用户自己的 venv 在项目目录里跑，
    和用户手动 python main.py 等价；加 timeout 防死循环，cwd 锁在项目根，PYTHONPATH 注入项目根/src。
    """
    import subprocess
    import sys as _sys

    python_bin = LOCAL_PYTHON or _sys.executable
    work_root = os.path.abspath(absolute_project_path)
    rel = f_path.replace("\\", "/").lstrip("/")
    abs_file = os.path.join(work_root, rel)

    env = os.environ.copy()
    extra_pp = [work_root, os.path.join(work_root, "src")]
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in extra_pp if os.path.isdir(p)] + ([existing_pp] if existing_pp else [])
    )
    env["WORKSPACE_ROOT"] = work_root

    # 提前扫描入口文件是否用相对导入（决定 -m 包启动 还是内联 exec），逻辑与 Docker 分支一致
    has_relative_import = False
    try:
        with open(abs_file, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(4096)
        if re.search(r'(?m)^\s*from\s+\.{1,}[\w\s]', head):
            has_relative_import = True
    except Exception:
        pass

    try:
        if has_relative_import:
            module_name = rel[:-3] if rel.endswith(".py") else rel
            module_name = module_name.replace("/", ".")
            cmd = [python_bin, "-m", module_name]
            log_info("子图2", "主测执行", f"本地兜底(-m)：{os.path.basename(python_bin)} -m {module_name} (cwd={work_root})")
        else:
            # 内联 bootstrap（与 Docker 沙箱共用同一套多行脚本，work_root 用本地绝对路径）
            bootstrap = _build_entry_bootstrap(work_root, rel)
            cmd = [python_bin, "-c", bootstrap]
            log_info("子图2", "主测执行", f"本地兜底(exec)：{os.path.basename(python_bin)} -c <bootstrap> (cwd={work_root})")

        proc = subprocess.run(
            cmd, cwd=work_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=MAIN_SANDBOX_TIMEOUT,
        )
        out = (proc.stdout or b"").decode("utf-8", errors="ignore").strip()
        rc = proc.returncode
        log_info("子图2", "主测执行", f"本地兜底执行完成 return_code={rc}")
        # 与沙箱结构对齐：失败时把合并输出同时放到 stderr，方便判定模型看到 traceback
        return {
            "success": rc == 0,
            "return_code": rc,
            "stdout": out,
            "stderr": "" if rc == 0 else out,
        }
    except subprocess.TimeoutExpired:
        msg = f"本地兜底执行超时（>{MAIN_SANDBOX_TIMEOUT}s），已终止"
        log_error("子图2", "主测执行", msg)
        return {"success": False, "return_code": -1, "stdout": "", "stderr": msg}
    except FileNotFoundError:
        msg = f"本地 Python 解释器不存在：{python_bin}（请检查配置 shared.local_python）"
        log_error("子图2", "主测执行", msg)
        return {"success": False, "return_code": -3, "stdout": "", "stderr": msg}
    except Exception as e:
        msg = f"本地兜底执行失败：{e}"
        log_error("子图2", "主测执行", msg)
        return {"success": False, "return_code": -1, "stdout": "", "stderr": msg}


def main_test_fix_and_run_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "主测执行", "开始...")
    top_entry = state.get("top_entry_file")
    last_error = state.get("last_main_error")
    main_retry = state.get("main_retry_count") or 0
    max_retry = state.get("max_main_retry") or MAIN_MAX_RETRY
    modify_summary = state.get("modify_summary") or []

    if not top_entry:
        log_warn("子图2", "主测执行", "无顶层入口文件，跳过主测")
        return {
            "raw_main_sandbox_result": {"success": False, "return_code": 0, "stdout": "", "stderr": "NO_TOP_ENTRY"},
            "last_main_error": None
        }

    f_path = top_entry["file_path"]
    f_path = _normalize_path(f_path)
    f_path = _remove_top_dir_if_present(f_path, PROJECT_TOP_DIR)
    f_path = _fix_path_for_project(f_path)
    abs_file = os.path.join(absolute_project_path, f_path)

    if not os.path.exists(abs_file):
        log_error("子图2", "主测执行", f"顶层入口文件不存在：{f_path}")
        return {
            "raw_main_sandbox_result": {"success": False, "return_code": -1, "stdout": "", "stderr": f"顶层入口文件不存在: {f_path}"},
            "last_main_error": f"文件不存在: {f_path}"
        }

    if last_error and main_retry < max_retry:
        log_info("子图2", "主测执行", f"第 {main_retry + 1}/{max_retry} 次修复")
        fix_prompt = safe_format(
            main_code_fix_prompt,
            project_root=absolute_project_path,
            entry_file=f_path,
            file_purpose=top_entry.get("file_purpose", ""),
            error_log=last_error
        )

        model_output_retry_cfg = shared.get("model_output_retry", {})
        default_retry = model_output_retry_cfg.get("default", 2)
        judge_retry = model_output_retry_cfg.get("judge_retry", default_retry)
        main_fix_retry = get_retry_count_for_node("main_fix", judge_retry, model_output_retry_cfg)

        try:
            validated = invoke_with_validation(
                model=model_change_test_fix,
                prompt=fix_prompt,
                pydantic_model=MainFixOutput,
                max_retries=main_fix_retry,
                node_name="主测修复",
                logic_validator=validate_main_fix_logic,
                validator_context="输出必须包含 is_environment_error (bool)、error_reason (str)、modified_files (list[dict]) 三个字段。"
            )
            fix_result = validated.dict()
        except RuntimeError as e:
            log_warn("子图2", "主测执行", f"修复模型失败，标记为环境错误: {e}")
            return {
                "is_environment_error": True,
                "last_main_error": str(e),
                "raw_main_sandbox_result": {"success": False, "return_code": 0, "stdout": "", "stderr": f"修复模型校验失败: {e}"},
                "modify_summary": modify_summary
            }

        is_env_error = fix_result.get("is_environment_error", False)
        if is_env_error:
            err_reason = fix_result.get("error_reason", "环境依赖不足")
            log_warn("子图2", "主测执行", f"环境错误：{err_reason}")
            modify_summary.append({"file": f_path, "status": "skip_env", "remark": err_reason})
            return {
                "is_environment_error": True,
                "last_main_error": None,
                "raw_main_sandbox_result": {"success": False, "return_code": 0, "stdout": "", "stderr": "ENV_ERROR_SKIP"},
                "modify_summary": modify_summary
            }

        modified_files = fix_result.get("modified_files", [])
        pending_dict = {}
        for item in modified_files:
            fp = item.get("file_path", "")
            fp = _normalize_path(fp)
            fp = _remove_top_dir_if_present(fp, PROJECT_TOP_DIR)
            fp = _fix_path_for_project(fp)
            code = item.get("fixed_code", "")
            desc = item.get("change_desc", "")
            pending_dict[fp] = code
            modify_summary.append({"file": fp, "status": "main_modified", "remark": desc})

        for fp, code in pending_dict.items():
            full_path = os.path.join(absolute_project_path, fp)
            full_path = os.path.abspath(full_path)
            base_abs = os.path.abspath(absolute_project_path)
            if not full_path.startswith(base_abs):
                log_warn("子图2", "主测执行", f"路径越界跳过：{fp}")
                continue
            try:
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(code)
                log_info("子图2", "主测执行", f"修复文件已写入：{fp}")
            except Exception as e:
                log_error("子图2", "主测执行", f"写入失败 {fp}：{e}")

        return {
            "pending_main_code_update": pending_dict,
            "main_retry_count": main_retry + 1,
            "modify_summary": modify_summary
        }

    client = get_docker_client()
    if client is None:
        # Docker 不可用：默认改用本地 Python 兜底真实跑一遍顶层主测，而不是直接跳过
        # （可在配置 shared.local_run_fallback: false 关闭此兜底）
        if LOCAL_RUN_FALLBACK:
            log_warn("子图2", "主测执行",
                     "Docker 不可用，改用本地 Python 兜底执行顶层主测（配置 shared.local_run_fallback=false 可关闭）")
            _conv = state.get("conv_id")
            if _conv:
                try:
                    notify_progress(_conv, "🧪 [修改模式] Docker 未启动，改用本机 Python 直接跑顶层主测...")
                except Exception:
                    pass
            local_result = _run_main_entry_locally(f_path)
            return {
                "raw_main_sandbox_result": local_result,
                "pending_main_code_update": None
            }
        log_warn("子图2", "主测执行", "Docker 不可用且已关闭本地兜底（local_run_fallback=false），跳过主测")
        return {
            "raw_main_sandbox_result": {"success": False, "return_code": -3, "stdout": "",
                                        "stderr": "Docker 不可用（且 local_run_fallback=false，已跳过本地兜底）"}
        }

    container = None
    try:
        # ======================================================================
        # ★★★ 模块路径 3 层兜底（解决图书管理系统 cli/main.py 找不到 models/services 的问题）★★★
        #   第一层: Docker 环境变量 PYTHONPATH，覆盖 95% 情况
        #       追加 /workspace/src 是兼容 "src 里的内容是顶层包" 这种老工程（src 作根）
        #   第二层: 启动命令改为 python -c "sys.path.insert(0,...); exec(open(...).read())"
        #       不依赖环境变量生效时机，也不依赖 PYTHONPATH 传递是否 OK
        #   第三层: 若入口文件包含相对 import (from ..  / from . )，
        #       则改用包启动模式 python -m a.b.c（去掉 .py 后缀 + / 改 .），这是 Python 官方唯一能跑相对 import 的方式
        # ======================================================================
        work_root = "/workspace"

        # ---------- 提前扫描入口文件内容，看是否有相对 import（决定是否切 -m 模式） ----------
        has_relative_import = False
        try:
            with open(os.path.join(absolute_project_path, f_path.lstrip("/\\")), "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            # 匹配: from .xxx / from ..xxx / from . import yyy / from .. import yyy
            if re.search(r'(?m)^\s*from\s+\.{1,}[\w\s]', head):
                has_relative_import = True
                log_info("子图2", "主测执行", "入口文件使用相对导入 → 切换为 python -m 包启动模式")
        except Exception:
            pass

        # ---------- 构建工作目录下相对的入口路径（去掉开头 / 或 \）----------
        entry_rel = f_path.replace("\\", "/").lstrip("/")
        # 第一层 env（即使容器内不支持，启动命令里的 sys.path 也会兜底）
        env = {
            "PYTHONPATH": f"{work_root}:{work_root}/src:/vendor",
            "WORKSPACE_ROOT": work_root,
        }

        if has_relative_import:
            # -m 模式： entry_rel = "cli/main.py" → 模块名 "cli.main"
            module_name = entry_rel
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            module_name = module_name.replace("/", ".")
            command = [
                "python", "-m", module_name,
            ]
            log_info("子图2", "主测执行", f"包启动模式: python -m {module_name}  (working_dir={work_root})")
        else:
            # 内联启动模式：双重注入 sys.path + exec()，比直接 python 入口文件更稳
            #   原因：python a/b/c.py 会把 a/b 放入 sys.path，
            #   但 exec(open(a/b/c.py).read()) 工作目录是 /workspace，我们自己控制 sys.path。
            #   bootstrap 用标准多行缩进（与本地兜底共用 _build_entry_bootstrap），避免单行 try/except 语法错误。
            bootstrap = _build_entry_bootstrap(work_root, entry_rel)
            command = ["python", "-c", bootstrap]

        container = client.containers.run(
            image=SANDBOX_IMAGE,
            command=command,
            environment=env,
            stdin_open=True,
            detach=True,
            remove=True,
            volumes={absolute_project_path: {"bind": "/workspace", "mode": "ro"}},
            working_dir="/workspace",
        )

        container.wait(timeout=MAIN_SANDBOX_TIMEOUT)
        all_logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore").strip()

        result = {"success": True, "return_code": 0, "stdout": all_logs, "stderr": ""}
        log_success("子图2", "主测执行", "执行完成")
    except docker.errors.ImageNotFound:
        result = {"success": False, "return_code": -2, "stdout": "", "stderr": f"镜像 {SANDBOX_IMAGE} 不存在"}
        log_error("子图2", "主测执行", f"镜像 {SANDBOX_IMAGE} 不存在")
    except Exception as e:
        result = {"success": False, "return_code": -1, "stdout": "", "stderr": f"主测沙箱调用失败：{str(e)}"}
        log_error("子图2", "主测执行", f"沙箱调用失败: {e}")
    finally:
        if container:
            try:
                container.kill()
            except Exception:
                pass

    return {
        "raw_main_sandbox_result": result,
        "pending_main_code_update": None
    }


################################################################################################################
# ====================== 节点15：main_test_judge_node ======================
def main_test_judge_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "主测判定", "开始...")
    top_entry = state.get("top_entry_file")
    raw_result = state.get("raw_main_sandbox_result")
    is_env_error = state.get("is_environment_error", False)
    main_retry = state.get("main_retry_count") or 0
    max_retry = state.get("max_main_retry") or MAIN_MAX_RETRY
    modify_summary = state.get("modify_summary") or []

    if not top_entry:
        log_warn("子图2", "主测判定", "无顶层入口，跳过")
        return {"last_main_error": None}

    f_path = top_entry["file_path"]
    f_path = _normalize_path(f_path)
    f_path = _remove_top_dir_if_present(f_path, PROJECT_TOP_DIR)
    f_path = _fix_path_for_project(f_path)

    if raw_result and raw_result.get("stderr") in ["NO_TOP_ENTRY", "ENV_ERROR_SKIP"]:
        log_info("子图2", "主测判定", "跳过主测（无入口或环境错误）")
        return {"last_main_error": None, "raw_main_sandbox_result": None}

    if is_env_error:
        log_warn("子图2", "主测判定", "环境错误跳过")
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": "主测环境错误"})
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "modify_summary": modify_summary
        }

    judge_prompt = safe_format(
        sandbox_judge_prompt,
        file_path=f_path,
        file_purpose=top_entry.get("file_purpose", ""),
        return_code=raw_result.get("return_code", -1) if raw_result else -1,
        stdout=raw_result.get("stdout", "") if raw_result else "",
        stderr=raw_result.get("stderr", "") if raw_result else ""
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    judge_retry = model_output_retry_cfg.get("judge_retry", default_retry)
    sandbox_judge_retry = get_retry_count_for_node("sandbox_judge", judge_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_change_test_judge,
            prompt=judge_prompt,
            pydantic_model=SandboxJudgeOutput,
            max_retries=sandbox_judge_retry,
            node_name="主测判定",
            validator_context="输出必须包含 success (bool)、is_environment_error (bool)、brief_reason (str)、error_detail (str) 四个字段。"
        )
        judge_dict = validated.dict()
    except RuntimeError as e:
        log_warn("子图2", "主测判定", f"判定模型失败，保守降级: {e}")
        judge_dict = {
            "success": False,
            "is_environment_error": False,
            "brief_reason": "判定模型校验失败，保守判定为失败",
            "error_detail": str(e)
        }

    success = judge_dict.get("success", False)
    judge_is_env = judge_dict.get("is_environment_error", False)
    error_detail = judge_dict.get("error_detail", "主测失败")
    brief_reason = judge_dict.get("brief_reason", "")

    if success:
        log_success("子图2", "主测判定", f"主测通过：{f_path}")
        modify_summary.append({"file": f_path, "status": "main_pass", "remark": brief_reason})
        print("\n📋 最终执行摘要：")
        for item in modify_summary:
            print(f"  - {item.get('file', '')} | {item.get('status', '')} | {item.get('remark', '')}")
        print("\n🎉 全部测试通过，修改完成！")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "modify_summary": modify_summary
        }
    elif judge_is_env:
        log_warn("子图2", "主测判定", f"环境错误跳过：{f_path}")
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": brief_reason})
        print("\n📋 最终执行摘要：")
        for item in modify_summary:
            print(f"  - {item.get('file', '')} | {item.get('status', '')} | {item.get('remark', '')}")
        print("\n🎉 修改完成（顶层业务因环境依赖跳过测试）")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "is_environment_error": True,
            "modify_summary": modify_summary
        }
    else:
        if main_retry >= max_retry:
            log_warn("子图2", "主测判定", f"达到最大重试次数({max_retry})，跳过主测")
            modify_summary.append({"file": f_path, "status": "skip_max_retry", "remark": f"重试{max_retry}次仍失败"})
            print("\n📋 最终执行摘要：")
            for item in modify_summary:
                print(f"  - {item.get('file', '')} | {item.get('status', '')} | {item.get('remark', '')}")
            print("\n⚠️ 修改完成（主测因重试次数耗尽而跳过）")
            return {
                "last_main_error": None,
                "raw_main_sandbox_result": None,
                "modify_summary": modify_summary
            }
        log_warn("子图2", "主测判定", f"主测失败，准备重试：{f_path}")
        return {
            "last_main_error": error_detail,
            "raw_main_sandbox_result": None
        }


################################################################################################################
# ====================== 节点16：dump_summary_md_node ======================
def dump_summary_md_node(state: CheckAndChangeCode) -> dict:
    log_info("子图2", "保存摘要", "保存摘要到磁盘...")
    os.makedirs(ai_summary_dir, exist_ok=True)
    memory_summary_list = state.get("file_summary_list", [])
    old_disk_list = []
    if state["has_local_summary_md"] and os.path.exists(summary_md_path):
        old_disk_list = parse_local_summary_md(summary_md_path)

    old_disk_map = {item["rel_path"]: item for item in old_disk_list}
    memory_map = {item["rel_path"]: item for item in memory_summary_list}
    merged_map = {**old_disk_map, **memory_map}
    merged_final_list = list(merged_map.values())

    # ★★★ 标准化所有路径 ★★★
    for item in merged_final_list:
        item["rel_path"] = _normalize_path(item.get("rel_path", ""))
        item["rel_path"] = _remove_top_dir_if_present(item["rel_path"], PROJECT_TOP_DIR)
        item["rel_path"] = _fix_path_for_project(item["rel_path"])

    def sort_key(item):
        return item.get("rel_path", "")

    sorted_new = sorted(merged_final_list, key=sort_key)
    sorted_old = sorted(old_disk_list, key=sort_key)

    summary_package = {
        "version": 2,
        "generated_at": datetime.now().isoformat(),
        "files": merged_final_list
    }

    if sorted_new != sorted_old:
        existing_version = None
        if old_disk_list:
            try:
                with open(summary_md_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                existing_data = json.loads(content)
                if "version" in existing_data:
                    existing_version = existing_data.get("version")
            except:
                pass

        if existing_version is not None and existing_version != summary_package["version"]:
            log_info("子图2", "保存摘要", f"版本从 {existing_version} 升级到 {summary_package['version']}")

        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary_package, ensure_ascii=False, indent=2))
        log_success("子图2", "保存摘要", f"已保存 {len(merged_final_list)} 条摘要 (版本 {summary_package['version']})")
    else:
        log_debug("子图2", "保存摘要", "摘要无变化，跳过写入")

    return {"file_summary_list": merged_final_list}


################################################################################################################
# ====================== 路由函数 ======================
def route_after_execute_change(state: CheckAndChangeCode):
    is_enough = state.get("is_enough_info", False)
    loop_count = state.get("loop_count", 0)
    max_loop = state.get("max_loop", 5)
    zero_streak = int(state.get("_zero_read_streak") or 0)

    # ★★★ HAS_SCHEMA 策略开关：仅在完整模式项目（有 Schema 背书）才触发 AST 回退重试 ★★★
    ast_errors = state.get("ast_validation_errors")
    if HAS_SCHEMA and ast_errors:
        log_info("子图2", "路由", f"AST 校验失败 ({len(ast_errors)} 个错误)，回到分析节点")
        return "analyze_task_node"
    if (not HAS_SCHEMA) and ast_errors:
        log_info("子图2", "路由", f"HAS_SCHEMA=False → 忽略 {len(ast_errors)} 个 AST 错误（无 Schema 背书），继续后续流程")

    # ★★★ 熔断：连续 2 轮读取 0 个文件 + 仍然信息不足 → 立即 END，不等到 max_loop 烧 3 次 LLM ★★★
    # 还原你遇到的场景：LLM 每次都请求 crawler.py / "py"（不存在）→ 读取 0 个 → execute 说信息不足 → 下轮又一样
    if zero_streak >= 2 and not is_enough:
        log_warn("子图2", "路由", f"连续 {zero_streak} 轮读取 0 个文件且信息不足 → 熔断 END（为了避免 5 次死循环浪费 LLM 调用）")
        notify_progress(state.get("conv_id"),
                        f"⚠️ [修改模式] 连续多轮没能读到需要的文件（可能路径/文件不存在），已自动结束，避免空转烧钱。建议：确认要修改的文件路径是否真的存在后重新发起修改需求。")
        return END

    if not is_enough:
        if loop_count < max_loop:
            log_info("子图2", "路由", "信息不足，继续循环")
            return "inc_loop_count_node"
        else:
            log_warn("子图2", "路由", "信息不足且达到最大循环，结束")
            return END
    else:
        log_info("子图2", "路由", "信息充足，进入影响分析")
        return "analyze_impact_node"


def route_after_impact(state: CheckAndChangeCode) -> str:
    impacted_files = state.get("impacted_files") or []
    if impacted_files:
        log_info("子图2", "路由", f"发现 {len(impacted_files)} 个受影响的文件，回到分析节点")
        return "analyze_task_node"
    else:
        log_info("子图2", "路由", "无受影响的文件，进入刷新摘要")
        return "re_summary_modified_node"


def route_after_collect(state: CheckAndChangeCode) -> str:
    need_re_eval = state.get("need_re_evaluate", False)
    if need_re_eval:
        log_info("子图2", "路由", "顶层入口有多个候选，回到 analyze_task_node 重新分析")
        return "analyze_task_node"
    return "unit_test_fix_and_run_node"


def decide_unit_test_next(state: CheckAndChangeCode) -> str:
    last_error = state.get("last_test_error")
    unit_list = state.get("unit_test_files") or []
    current_file = state.get("current_test_file")
    is_env_error = state.get("is_environment_error", False)

    if is_env_error:
        return "main_test_fix_and_run_node"
    if last_error:
        return "unit_test_fix_and_run_node"
    if unit_list and current_file:
        return "unit_test_fix_and_run_node"

    log_info("子图2", "路由", "单测全部完成，进入主测")
    return "main_test_fix_and_run_node"


def decide_main_test_next(state: CheckAndChangeCode) -> str:
    last_error = state.get("last_main_error")
    if last_error:
        return "main_test_fix_and_run_node"
    return "dump_summary_md_node"


################################################################################################################
# ====================== 构建图 ======================
def build_change_graph():
    log_info("子图2", "构建图", "开始构建...")
    change_project_graph = StateGraph(CheckAndChangeCode)

    change_project_graph.add_node("read_local_project_tree_node", read_local_project_tree_node)
    change_project_graph.add_node("detect_exist_summary_md_node", detect_exist_summary_md_node)
    change_project_graph.add_node("analyze_task_node", analyze_task_node)
    change_project_graph.add_node("read_target_file_node", read_target_file_node)
    change_project_graph.add_node("generate_file_summary_node", generate_file_summary_node)
    change_project_graph.add_node("execute_change_node", execute_change_node)
    change_project_graph.add_node("inc_loop_count_node", inc_loop_count_node)
    change_project_graph.add_node("analyze_impact_node", analyze_impact_node)
    change_project_graph.add_node("re_summary_modified_node", re_summary_modified_node)
    change_project_graph.add_node("write_file_node", write_file_node)

    change_project_graph.add_node("collect_test_resources_node", collect_test_resources_node)
    change_project_graph.add_node("unit_test_fix_and_run_node", unit_test_fix_and_run_node)
    change_project_graph.add_node("unit_test_judge_node", unit_test_judge_node)
    change_project_graph.add_node("main_test_fix_and_run_node", main_test_fix_and_run_node)
    change_project_graph.add_node("main_test_judge_node", main_test_judge_node)

    change_project_graph.add_node("dump_summary_md_node", dump_summary_md_node)

    change_project_graph.add_edge(START, "read_local_project_tree_node")
    change_project_graph.add_edge("read_local_project_tree_node", "detect_exist_summary_md_node")
    change_project_graph.add_edge("detect_exist_summary_md_node", "analyze_task_node")
    change_project_graph.add_edge("analyze_task_node", "read_target_file_node")
    change_project_graph.add_edge("read_target_file_node", "generate_file_summary_node")
    change_project_graph.add_edge("generate_file_summary_node", "execute_change_node")

    change_project_graph.add_conditional_edges(
        "execute_change_node",
        route_after_execute_change
    )
    change_project_graph.add_edge("inc_loop_count_node", "analyze_task_node")

    change_project_graph.add_conditional_edges(
        "analyze_impact_node",
        route_after_impact
    )

    change_project_graph.add_edge("re_summary_modified_node", "write_file_node")
    change_project_graph.add_edge("write_file_node", "collect_test_resources_node")

    change_project_graph.add_conditional_edges(
        "collect_test_resources_node",
        route_after_collect
    )

    change_project_graph.add_edge("unit_test_fix_and_run_node", "unit_test_judge_node")
    change_project_graph.add_conditional_edges(
        "unit_test_judge_node",
        decide_unit_test_next
    )

    change_project_graph.add_edge("main_test_fix_and_run_node", "main_test_judge_node")
    change_project_graph.add_conditional_edges(
        "main_test_judge_node",
        decide_main_test_next
    )

    change_project_graph.add_edge("dump_summary_md_node", END)

    graph = change_project_graph.compile()
    log_success("子图2", "构建图", "完成")
    return graph


################################################################################################################
# ====================== 使用示例 ======================
if __name__ == "__main__":
    log_separator("子图2", "启动")
    change_graph_app = build_change_graph()

    init_state: CheckAndChangeCode = {
        "messages": [],
        "user_change_requirement": "实现模糊查询功能并更新在主函数中，实现功能，模糊查询，第五个功能",
        "change_is_ok": False,
        "file_name_list": [],
        "change_content_list": [],
        "project_file_tree": "",
        "file_summary_list": [],
        "next_file_name_list": [],
        "user_question": None,
        "loop_count": 0,
        "max_loop": 5,
        "has_local_summary_md": False,
        "is_enough_info": False,
        "insufficient_reason": None,
        "need_additional_file_list": None,
        "target_file_source_map": None,
        "changed_file_map": None,
        "processed_file_list": [],
        "inferred_can_unit_test": None,
        "inferred_top_entry": None,
        "top_entry_confidence": None,
        "unit_test_files": None,
        "top_entry_file": None,
        "current_test_file": None,
        "pending_file_code_update": None,
        "raw_sandbox_result": None,
        "last_test_error": None,
        "test_retry_count": 0,
        "max_test_retry": UNIT_MAX_RETRY,
        "pending_main_code_update": None,
        "raw_main_sandbox_result": None,
        "last_main_error": None,
        "main_retry_count": 0,
        "max_main_retry": MAIN_MAX_RETRY,
        "is_environment_error": False,
        "modify_summary": [],
        "test_phase": None,
        "need_re_evaluate": False,
        "candidate_top_entries": None,
        "impacted_files": None,
        "ast_validation_errors": None
    }

    result = change_graph_app.invoke(init_state)

    print("\n" + "=" * 60)
    print("📋 修改执行摘要")
    print("=" * 60)

    modify_summary = result.get("modify_summary") or []
    if modify_summary:
        print("\n📝 修改摘要:")
        for item in modify_summary:
            print(f"  - {item.get('file', '')} | {item.get('status', '')} | {item.get('remark', '')}")

    unit_list = result.get("unit_test_files") or []
    if unit_list:
        print(f"\n🔍 单测列表剩余: {len(unit_list)} 个文件")
        for item in unit_list:
            print(f"  - {item.get('file_path', '')}")

    top_entry = result.get("top_entry_file")
    if top_entry:
        print(f"\n📌 顶层入口: {top_entry.get('file_path', '')}")

    print("\n" + "=" * 60)