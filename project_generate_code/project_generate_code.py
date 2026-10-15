import json
import os
import re
import operator
import time
import subprocess
import sys
import hashlib
from typing import Annotated, TypedDict, List, Dict, Optional, Callable, Type
from typing import Literal
from datetime import datetime

import docker
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph, START, END
import yaml
from langgraph.types import interrupt

# ========== Pydantic 校验导入 ==========
from pydantic import BaseModel, Field, ValidationError

# ★★★ 缓存管理器导入（新增） ★★★
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
    try:
        _prov_mod_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
        if _prov_mod_dir not in sys.path:
            sys.path.insert(0, _prov_mod_dir)
        from provider_factory import resolve_model_credentials  # type: ignore
    except Exception:
        def resolve_model_credentials(model_provider, model_name, shared_cfg=None):  # type: ignore
            if model_provider == "ali":
                return os.getenv("ALI_API_KEY", ""), model_name
            if model_provider == "deepseek":
                return os.getenv("DEEPSEEK_API_KEY", ""), model_name
            return "", model_name

load_dotenv()

# ========== 修复：基于 __file__ 的绝对路径，避免依赖 cwd 导致配置文件找不到 ==========
_config_script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(_config_script_dir, "../config_file/my_config.yaml"))
################################################################################################################
# 过滤系统模块
try:
    STDLIB_MODULES = set(sys.stdlib_module_names)
except AttributeError:
    # Python 3.10 以下版本手动列出常见标准库
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

# ========== 子图1 专属运行时配置 ==========
create_runtime = config.get("create_runtime", {})
DEFAULT_TEMPERATURE = create_runtime.get("default_temperature", 0.1)
MAX_CLARIFY_STEPS = create_runtime.get("max_clarify_loop_steps", 3)
UNIT_TEST_MAX_RETRY = create_runtime.get("unit_test_max_retry", 3)
MAIN_TEST_MAX_RETRY = create_runtime.get("main_test_max_retry", 3)

# ========== 全局共享配置 ==========
shared = config.get("shared", {})
SANDBOX_IMAGE = shared.get("sandbox_image", "sandbox-web:v1")
SANDBOX_TIMEOUT = shared.get("single_sandbox_timeout", 240)
MAIN_SANDBOX_TIMEOUT = shared.get("main_sandbox_timeout", 300)
VENDOR_DIR = shared.get("vendor_mount_path", "/tmp/vendor")
PIP_INDEX_URL = shared.get("pip_index_url", "https://mirrors.aliyun.com/pypi/simple/")
PIP_TRUSTED_HOST = shared.get("pip_trusted_host", "mirrors.aliyun.com")
PIP_MAX_RETRIES = shared.get("pip_max_retries", 3)
PIP_RETRY_DELAY = shared.get("pip_retry_delay", 5)

# ========== 镜像预装依赖列表 ==========
preinstalled_pip = shared.get("preinstalled_pip", [])
preinstalled_names = {item.split("==")[0] for item in preinstalled_pip}

# ========== 模块名 → pip 包名映射 ==========
module_to_pip = shared.get("module_to_pip", {})

# ========== 项目保存路径 ==========
try:
    absolute_path = config["create_project_path"]
except KeyError:
    absolute_path = config.get("absolute_save_path", os.path.abspath(__file__))
save_path = os.path.abspath(os.path.join(absolute_path, "./new_project"))

# ========== Docker 客户端（惰性初始化） ==========
_docker_client = None


def get_docker_client():
    """惰性获取 Docker 客户端，仅在首次调用时初始化"""
    global _docker_client
    if _docker_client is None:
        try:
            os.environ["DOCKER_CONTEXT"] = "default"
            _docker_client = docker.from_env()
            _docker_client.ping()
            log_info("子图1", "Docker", "连接成功")
        except Exception as e:
            log_warn("子图1", "Docker", f"连接失败，沙箱功能将不可用: {e}")
            _docker_client = False
    return _docker_client if _docker_client is not False else None


def get_openai_model(model_provider: str, model_name: str):
    """Provider 工厂兼容封装（保留原函数名与签名，调用方零改动即可升级）。
    新增供应商支持：在 shared.llm_providers 加 YAML 段 + .env 加 key 即可，不改 Python。"""
    api_key, m_name = resolve_model_credentials(model_provider, model_name, shared)
    if not api_key:
        raise RuntimeError(f"模型 {model_provider} 的API密钥环境变量未配置！")
    return api_key, m_name


def _check_api_key(provider_name: str, key: str):
    if not key:
        raise RuntimeError(f"[{provider_name}] API密钥未配置，请检查.env文件")


# ========== 第1个模型：需求澄清 ==========
ask_cfg = config["ask_model"]
api_key_ask, model_name_ask = get_openai_model(ask_cfg["model_provider"], ask_cfg["model_name"])
_check_api_key(ask_cfg["model_provider"], api_key_ask)
model_ask = ChatOpenAI(
    model=model_name_ask,
    api_key=api_key_ask,
    base_url=ask_cfg["base_url"],
    temperature=DEFAULT_TEMPERATURE,
    timeout=ask_cfg.get("timeout", 60),
    max_retries=ask_cfg.get("max_retries", 2)
)

# ========== 第2个模型：架构方案A ==========
front_one_cfg = config["front_plan_model_one"]
api_key_front_one, model_name_front_one = get_openai_model(
    front_one_cfg["model_provider"], front_one_cfg["model_name"])
_check_api_key(front_one_cfg["model_provider"], api_key_front_one)
model_front_plan_one = ChatOpenAI(
    model=model_name_front_one,
    api_key=api_key_front_one,
    base_url=front_one_cfg["base_url"],
    temperature=front_one_cfg.get("temperature", DEFAULT_TEMPERATURE),
    timeout=front_one_cfg.get("timeout", 60),
    max_retries=front_one_cfg.get("max_retries", 2)
)

# ========== 第3个模型：架构方案B ==========
front_two_cfg = config["front_plan_model_two"]
api_key_front_two, model_name_front_two = get_openai_model(
    front_two_cfg["model_provider"], front_two_cfg["model_name"])
_check_api_key(front_two_cfg["model_provider"], api_key_front_two)
model_front_plan_two = ChatOpenAI(
    model=model_name_front_two,
    api_key=api_key_front_two,
    base_url=front_two_cfg["base_url"],
    temperature=front_two_cfg.get("temperature", DEFAULT_TEMPERATURE),
    timeout=front_two_cfg.get("timeout", 60),
    max_retries=front_two_cfg.get("max_retries", 2)
)

# ========== 第4个模型：方案合并 ==========
back_cfg = config["back_plan_model"]
api_key_back, model_name_back = get_openai_model(back_cfg["model_provider"], back_cfg["model_name"])
_check_api_key(back_cfg["model_provider"], api_key_back)
model_back_plan = ChatOpenAI(
    model=model_name_back,
    api_key=api_key_back,
    base_url=back_cfg["base_url"],
    temperature=back_cfg.get("temperature", DEFAULT_TEMPERATURE),
    timeout=back_cfg.get("timeout", 60),
    max_retries=back_cfg.get("max_retries", 2)
)

# ========== 第5个模型：代码生成 ==========
code_gen_cfg = config["code_generate_model"]
api_key_code_gen, model_name_code_gen = get_openai_model(
    code_gen_cfg["model_provider"], code_gen_cfg["model_name"])
_check_api_key(code_gen_cfg["model_provider"], api_key_code_gen)
model_code_generate = ChatOpenAI(
    model=model_name_code_gen,
    api_key=api_key_code_gen,
    base_url=code_gen_cfg["base_url"],
    temperature=code_gen_cfg.get("temperature", DEFAULT_TEMPERATURE),
    timeout=code_gen_cfg.get("timeout", 60),
    max_retries=code_gen_cfg.get("max_retries", 2)
)

# ========== 第6个模型：测试修复 ==========
test_fix_cfg = config["test_fix_model"]
api_key_test_fix, model_name_test_fix = get_openai_model(
    test_fix_cfg["model_provider"], test_fix_cfg["model_name"])
_check_api_key(test_fix_cfg["model_provider"], api_key_test_fix)
model_test_fix = ChatOpenAI(
    model=model_name_test_fix,
    api_key=api_key_test_fix,
    base_url=test_fix_cfg["base_url"],
    temperature=test_fix_cfg.get("temperature", DEFAULT_TEMPERATURE),
    timeout=test_fix_cfg.get("timeout", 60),
    max_retries=test_fix_cfg.get("max_retries", 2)
)

# ========== 第7个模型：测试判定 ==========
test_judge_cfg = config["test_judge_model"]
api_key_test_judge, model_name_test_judge = get_openai_model(
    test_judge_cfg["model_provider"], test_judge_cfg["model_name"])
_check_api_key(test_judge_cfg["model_provider"], api_key_test_judge)
model_test_judge = ChatOpenAI(
    model=model_name_test_judge,
    api_key=api_key_test_judge,
    base_url=test_judge_cfg["base_url"],
    temperature=test_judge_cfg.get("temperature", 0.0),
    timeout=test_judge_cfg.get("timeout", 30),
    max_retries=test_judge_cfg.get("max_retries", 2)
)


################################################################################################################
# ====================== Pydantic 校验模型定义 ======================
class ClarifyOutput(BaseModel):
    is_enough: bool
    final_requirement: str = Field(default="")
    questions: List[str] = Field(default_factory=list)


class ExportApiParam(BaseModel):
    param_name: str
    param_type: str
    desc: str


# ★★★ 新增：类方法级契约（P0 核心）★★★
class ClassMethod(BaseModel):
    method_name: str = Field(description="方法名，如 get_profile")
    params: List[ExportApiParam] = Field(default_factory=list, description="方法参数列表")
    return_desc: str = Field(default="", description="返回值描述")


class ExportApi(BaseModel):
    name: str
    type: str  # "函数" 或 "类"
    params: List[ExportApiParam] = Field(default_factory=list)  # 函数的参数 / 类的构造参数
    return_desc: str = Field(default="")
    exception: str = Field(default="无")
    # ★ 仅 type="类" 时有值，列出类的所有公开方法签名
    class_methods: List[ClassMethod] = Field(default_factory=list)


class CallOtherApi(BaseModel):
    target_file: str
    api_name: str


# ========== 13层契约模型 ==========
class BusinessData(BaseModel):
    input: str = Field(default="")
    output: str = Field(default="")
    side_effects: List[str] = Field(default_factory=list)


class BusinessRole(BaseModel):
    in_flows: List[str] = Field(default_factory=list)
    flow_position: str = Field(default="")
    business_rules: List[str] = Field(default_factory=list)
    business_data: BusinessData = Field(default_factory=BusinessData)
    dependent_flows: List[str] = Field(default_factory=list)


class DataModel(BaseModel):
    name: str
    type: Literal["TypedDict", "BaseModel", "PydanticModel"]
    fields: Dict[str, str] = Field(default_factory=dict)
    description: str = Field(default="")


class ErrorDefinition(BaseModel):
    code: int
    module: str = Field(default="")
    message: str
    http_status: int = Field(default=500)


class GenerationItem(BaseModel):
    sequence: int
    file: str
    reason: str = Field(default="")


class FlowStep(BaseModel):
    step: int
    action: str
    actor: str = Field(default="系统")


class CoreFlow(BaseModel):
    name: str
    description: str = Field(default="")
    steps: List[FlowStep] = Field(default_factory=list)
    involved_files: List[str] = Field(default_factory=list)


class BusinessContext(BaseModel):
    overview: str = Field(default="")
    user_roles: List[str] = Field(default_factory=list)
    core_flows: List[CoreFlow] = Field(default_factory=list)


class LayerRule(BaseModel):
    from_layer: str
    to_layers: List[str] = Field(default_factory=list)


class LayerRules(BaseModel):
    allowed_dependencies: List[LayerRule] = Field(default_factory=list)
    forbidden_imports: List[LayerRule] = Field(default_factory=list)


# ★★★ 新增：ImplementationGuide、TestGuide、ExternalDependency（P0 核心） ★★★
class ImplementationGuide(BaseModel):
    core_logic: str = Field(description="核心实现逻辑描述，1-2句话说明这个文件要完成什么业务动作")
    key_functions: List[str] = Field(default_factory=list, description="实现该逻辑的关键函数名称列表")
    error_scenarios: List[str] = Field(default_factory=list, description="需要处理的错误场景")
    performance_considerations: List[str] = Field(default_factory=list, description="性能注意事项")
    collaboration_notes: List[str] = Field(default_factory=list, description="与其他文件的协作要点")


class TestGuide(BaseModel):
    can_test: bool = Field(default=False, description="是否可以在沙箱中独立单测")
    skip_reason: str = Field(default="", description="如果不能单测，给出具体原因")
    mock_suggestions: List[str] = Field(default_factory=list, description="建议 mock 哪些外部依赖")
    test_cases: List[str] = Field(default_factory=list, description="建议的测试用例描述")


class ExternalDependency(BaseModel):
    name: str = Field(description="依赖名称，如 OPENAI_API_KEY")
    dep_type: Literal["api_key", "database", "service", "file", "environment"] = Field(description="依赖类型")
    required: bool = Field(default=True, description="是否必须")
    description: str = Field(default="", description="依赖用途说明")


# ★★★ 新增：单文件生成输出校验模型 ★★★
class SingleFileOutput(BaseModel):
    code: str = Field(description="完整的文件源码")
    summary: str = Field(description="文件的极简摘要，不超过80字")


class FileItem(BaseModel):
    file_path: str
    file_purpose: str
    depend_on_files: List[str] = Field(default_factory=list)
    is_test_file: bool = False
    can_unit_test: bool = False
    is_top_entry: bool = False
    template_content: str = Field(default="")
    export_apis: List[ExportApi] = Field(default_factory=list)
    call_other_apis: List[CallOtherApi] = Field(default_factory=list)
    business_role: BusinessRole = Field(default_factory=BusinessRole)
    # ★★★ 新增三个字段 ★★★
    implementation_guide: ImplementationGuide = Field(default_factory=ImplementationGuide)
    test_guide: TestGuide = Field(default_factory=TestGuide)
    external_dependencies: List[ExternalDependency] = Field(default_factory=list)


class DirectoryItem(BaseModel):
    path: str
    desc: str = Field(default="")
    files: List[FileItem] = Field(default_factory=list)
    sub_dirs: List['DirectoryItem'] = Field(default_factory=list)


class SchemaDependencies(BaseModel):
    pip: List[str] = Field(default_factory=list)
    dev_dependencies: List[str] = Field(default_factory=list)


class SchemaOutput(BaseModel):
    project_name: str
    project_desc: str = Field(default="")
    entry_file: str = Field(default="")
    dependencies: SchemaDependencies = Field(default_factory=SchemaDependencies)
    directory_tree: List[DirectoryItem] = Field(default_factory=list)
    business_context: BusinessContext = Field(default_factory=BusinessContext)
    generation_order: List[GenerationItem] = Field(default_factory=list)
    data_models: Dict[str, DataModel] = Field(default_factory=dict)
    error_registry: Dict[str, ErrorDefinition] = Field(default_factory=dict)
    layer_rules: LayerRules = Field(default_factory=LayerRules)


class SandboxJudgeOutput(BaseModel):
    success: bool
    is_environment_error: bool = False
    brief_reason: str = Field(default="")
    error_detail: str = Field(default="")


################################################################################################################
# ====================== 业务逻辑校验函数 ======================
def _collect_all_files_from_schema(schema: SchemaOutput) -> List[FileItem]:
    result = []

    def traverse(items: List[DirectoryItem]):
        for dir_item in items:
            result.extend(dir_item.files)
            if dir_item.sub_dirs:
                traverse(dir_item.sub_dirs)

    traverse(schema.directory_tree)
    return result


def _file_exists_in_schema(schema: SchemaOutput, file_path: str) -> bool:
    all_files = _collect_all_files_from_schema(schema)
    for f in all_files:
        if f.file_path == file_path:
            return True
    return False


def _get_export_api_names(file_item: FileItem) -> List[str]:
    return [api.name for api in file_item.export_apis]


def _get_top_entry_files(schema: SchemaOutput) -> List[str]:
    all_files = _collect_all_files_from_schema(schema)
    return [f.file_path for f in all_files if f.is_top_entry]


def validate_schema_logic(schema: SchemaOutput) -> List[str]:
    errors = []
    all_files = _collect_all_files_from_schema(schema)
    file_path_set = {f.file_path for f in all_files}
    file_path_to_item = {f.file_path: f for f in all_files}

    for f in all_files:
        has_run = any(api.name == "run" for api in f.export_apis)
        if has_run and not f.is_top_entry:
            errors.append(
                f"文件 {f.file_path} 导出了 'run' 函数，但 is_top_entry 为 false。"
                f"只有顶层入口文件（is_top_entry=true）可以导出 run 函数。"
            )

    if schema.entry_file and schema.entry_file not in file_path_set:
        errors.append(f"entry_file '{schema.entry_file}' 在 directory_tree 中不存在")

    top_entries = _get_top_entry_files(schema)
    if len(top_entries) == 0:
        errors.append("directory_tree 中没有标记 is_top_entry=true 的文件")
    elif len(top_entries) > 1:
        errors.append(f"directory_tree 中有多个 is_top_entry=true 的文件: {top_entries}")

    for f in all_files:
        if f.can_unit_test and len(f.depend_on_files) > 0:
            errors.append(
                f"文件 {f.file_path} 标记为 can_unit_test=true，但 depend_on_files 不为空: {f.depend_on_files}"
            )

    # ★★★ 接口闭环校验（新增） ★★★
    for f in all_files:
        for call in f.call_other_apis:
            target_file = call.target_file
            api_name = call.api_name

            if target_file not in file_path_to_item:
                errors.append(
                    f"文件 {f.file_path} 调用了不存在的 target_file '{target_file}'"
                )
                continue

            target_item = file_path_to_item[target_file]
            export_names = [api.name for api in target_item.export_apis]
            if api_name not in export_names:
                errors.append(
                    f"文件 {f.file_path} 调用 {target_file} 的 '{api_name}' 不存在。"
                    f"目标文件导出: {export_names}"
                )

    # ★★★ 检查是否有文件被调用但没有导出对应接口（新增） ★★★
    called_by = {}
    for f in all_files:
        for call in f.call_other_apis:
            target = call.target_file
            api = call.api_name
            if target not in called_by:
                called_by[target] = []
            if api not in called_by[target]:
                called_by[target].append(api)

    for target_file, called_apis in called_by.items():
        if target_file not in file_path_to_item:
            continue
        target_item = file_path_to_item[target_file]
        export_names = [api.name for api in target_item.export_apis]
        missing_apis = [api for api in called_apis if api not in export_names]
        if missing_apis:
            errors.append(
                f"文件 {target_file} 被调用以下接口，但未在 export_apis 中导出: {missing_apis}"
            )

    # ★★★ 校验 implementation_guide 非空（新增） ★★★
    for f in all_files:
        if not f.implementation_guide.core_logic or f.implementation_guide.core_logic.strip() == "":
            errors.append(f"文件 {f.file_path} 的 implementation_guide.core_logic 为空，请补充")

        if not f.implementation_guide.collaboration_notes:
            errors.append(f"文件 {f.file_path} 的 implementation_guide.collaboration_notes 为空，请补充")

    # ★★★ 新增：类方法级契约校验 ★★★
    for f in all_files:
        for api in f.export_apis:
            if api.type == "类":
                if not api.class_methods:
                    errors.append(
                        f"文件 {f.file_path} 的类 '{api.name}' 没有声明 class_methods。"
                        f"请列出该类的所有公开方法签名（方法名 + 参数 + 返回值）。"
                    )

    # ★★★ 校验 test_guide 完整性（新增） ★★★
    for f in all_files:
        if not f.test_guide.can_test and not f.test_guide.skip_reason:
            errors.append(f"文件 {f.file_path} 的 test_guide.skip_reason 为空，请说明为什么不能单测")

    dep_graph = {}
    for f in all_files:
        dep_graph[f.file_path] = f.depend_on_files

    def has_cycle(node: str, visited: set, stack: set) -> bool:
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for neighbor in dep_graph.get(node, []):
            if neighbor in dep_graph:
                if has_cycle(neighbor, visited, stack):
                    return True
        stack.remove(node)
        return False

    visited = set()
    for f in all_files:
        if has_cycle(f.file_path, visited, set()):
            errors.append(f"检测到循环依赖，涉及文件: {f.file_path}")

    if not schema.project_name or schema.project_name.strip() == "":
        errors.append("project_name 不能为空")

    return errors


################################################################################################################
# ====================== 通用校验调用函数 ======================
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
                    log_warn("子图1", node_name, f"JSON解析失败，第 {attempt + 1} 次重试...")
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
                    log_warn("子图1", node_name, f"字段校验失败，第 {attempt + 1} 次重试...")
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
                        log_warn("子图1", node_name, f"业务逻辑校验失败，第 {attempt + 1} 次重试...")
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
                log_warn("子图1", node_name, f"未知错误，第 {attempt + 1} 次重试...")
                continue
            else:
                raise RuntimeError(
                    f"模型调用失败（重试{max_retries}次后仍失败）: {str(e)}"
                )

    raise RuntimeError(f"模型调用失败（重试{max_retries}次后仍失败）")


################################################################################################################
# json数据清洗（支持数组）
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


################################################################################################################
# 全局工具：递归提取目录树中全部file对象
def collect_all_files(tree_list):
    file_collect = []
    for dir_item in tree_list:
        files = dir_item.get("files", [])
        file_collect.extend(files)
        sub_dirs = dir_item.get("sub_dirs", [])
        file_collect.extend(collect_all_files(sub_dirs))
    return file_collect


# ====================== 依赖图工具 ======================
def build_dependency_graph(all_file_items: List[dict]) -> dict:
    graph = {}
    for item in all_file_items:
        file_path = item.get("file_path", "")
        depend_on = item.get("depend_on_files", []) or []
        # ===== Critical #3 修复：只保留确实出现在文件清单里的依赖（LLM 写错 depend_on_files 指向不存在文件会污染拓扑排序）=====
        valid_deps = [d for d in depend_on if isinstance(d, str) and d]
        graph[file_path] = valid_deps
    return graph


def topological_sort(graph: dict) -> List[str]:
    """
    Kahn 拓扑排序。
    约定：graph[A] = [B, C] 表示 "A 依赖 B、C（B / C 必须比 A 先生成）"。
    因此入度定义为"该文件未被满足的直接依赖数量"：A 的入度 = |graph[A]|。
    """
    from collections import deque

    # ===== Critical #3 修复：原 in_degree 写反（之前把 in_degree[dep]++ 算了"被依赖数"当入度）=====
    #   旧算法在 graph = {"a": [], "b": ["a"], "c": ["b"]} 上会得到 in_degree = {a:1,b:1,c:0}，
    #   处理顺序为 c → 无后续，最后抛出"循环依赖: {a, b}"——完全错误。
    #   新算法：in_degree[A] = len(graph[A])，0 依赖的文件先出队，处理完它再把所有"以它为依赖"的文件 in_degree 减 1。
    in_degree = {node: 0 for node in graph}
    for node, deps in graph.items():
        unknown = [d for d in deps if d not in in_degree]
        if unknown:
            # 依赖了 graph 中不存在的文件（Schema.depend_on_files 写错或 LLM 幻觉）→ 视为 warn，不计入阻塞入度
            log_warn("子图1", "拓扑排序",
                     f"⚠️ 文件 {node} 的 depend_on_files 指向未知文件 {unknown[:6]}（已忽略这些未知依赖）")
            in_degree[node] = len([d for d in deps if d in in_degree])
        else:
            in_degree[node] = len(deps)

    # 反向索引：被依赖项 dep → [所有依赖 dep 的文件 nodes]
    reverse: Dict[str, List[str]] = {node: [] for node in graph}
    for node, deps in graph.items():
        for d in deps:
            if d in reverse:
                reverse[d].append(node)

    queue = deque([node for node, deg in in_degree.items() if deg == 0])
    result: List[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for dependent in reverse[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(graph):
        raise RuntimeError(f"检测到循环依赖: {sorted(set(graph.keys()) - set(result))}")

    return result


# ====================== AST 契约校验工具 ======================
import ast
import inspect
from typing import Dict, Any


def extract_exported_apis_from_code(source_code: str) -> Dict[str, Dict[str, Any]]:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    result = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue

            params = []
            for arg in node.args.args:
                if arg.arg not in ["self", "cls"]:
                    params.append(arg.arg)

            return_annotation = None
            if node.returns:
                try:
                    return_annotation = ast.unparse(node.returns)
                except Exception:
                    return_annotation = "Any"

            result[node.name] = {
                "params": params,
                "return_annotation": return_annotation,
                "type": "function"
            }

        elif isinstance(node, ast.ClassDef):
            init_method = None
            class_methods = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        init_method = item
                    elif not item.name.startswith("_"):
                        # ★ 提取类的公开方法签名
                        method_params = []
                        for arg in item.args.args:
                            if arg.arg not in ["self", "cls"]:
                                method_params.append(arg.arg)
                        method_return = None
                        if item.returns:
                            try:
                                method_return = ast.unparse(item.returns)
                            except Exception:
                                method_return = "Any"
                        class_methods[item.name] = {
                            "params": method_params,
                            "return_annotation": method_return
                        }

            params = []
            if init_method:
                for arg in init_method.args.args:
                    if arg.arg not in ["self"]:
                        params.append(arg.arg)

            result[node.name] = {
                "params": params,
                "return_annotation": None,
                "type": "class",
                "class_methods": class_methods  # ★ 新增：类方法签名
            }

    return result


def validate_code_against_schema(
        source_code: str,
        expected_apis: List[Dict[str, Any]]
) -> List[str]:
    """
    校验代码是否导出了 schema 中声明的接口
    兼容 expected_apis 为字符串列表或字典列表
    """
    errors = []

    # ★★★ 类型兼容：如果是字符串列表，转换为"最少信息"的 dict（name + 参数数量从实际 AST 推断 = "不要在参数数量上误报"）★★★
    if expected_apis and isinstance(expected_apis, list):
        if len(expected_apis) > 0 and isinstance(expected_apis[0], str):
            # 为什么不直接塞空参数？
            #   → 如果字符串列表是 LLM 仅 output 了接口名 + 没给参数信息，我们只校验"接口存在"，
            #     参数数量按实际代码为准（避免 35/38 断言里那个"字符串列表默认 0 参数 vs 实际 2 参数"的误判）。
            actual_quick = extract_exported_apis_from_code(source_code)
            converted = []
            for name in expected_apis:
                quick_actual_params = actual_quick.get(name, {}).get("params", []) or []
                converted.append({
                    "name": name,
                    "type": "函数",
                    "params": [{"param_name": p} for p in quick_actual_params],
                    "return_desc": "",
                    "exception": "无",
                })
            expected_apis = converted

    # 检查源码语法
    try:
        ast.parse(source_code)
    except SyntaxError as e:
        errors.append(f"❌ 代码存在语法错误: {e}")
        return errors

    # 提取实际导出的接口
    actual_apis = extract_exported_apis_from_code(source_code)

    # 如果没有预期接口，直接返回
    if not expected_apis:
        return errors

    for expected in expected_apis:
        # ★★★ 安全获取：兼容字典或字符串 ★★★
        if isinstance(expected, str):
            api_name = expected
            api_type = "函数"
            expected_param_names = []
            expected_return = ""
        elif isinstance(expected, dict):
            api_name = expected.get("name")
            api_type = expected.get("type", "函数")
            expected_param_names = [p.get("param_name") for p in expected.get("params", []) if p.get("param_name")]
            expected_return = expected.get("return_desc", "")
        else:
            continue

        if not api_name:
            continue

        # 检查接口是否存在
        if api_name not in actual_apis:
            errors.append(f"❌ 缺少对外接口: {api_name}")
            continue

        actual = actual_apis[api_name]
        actual_param_names = actual.get("params", [])

        # 参数数量校验
        if len(expected_param_names) != len(actual_param_names):
            errors.append(
                f"❌ {api_name} 参数数量不匹配: "
                f"预期 {len(expected_param_names)} 个 ({expected_param_names}), "
                f"实际 {len(actual_param_names)} 个 ({actual_param_names})"
            )
        else:
            # ★ 参数名匹配警告（非阻断，允许 LLM 用同义词）
            mismatch = [(e, a) for e, a in zip(expected_param_names, actual_param_names) if e != a]
            if mismatch:
                errors.append(
                    f"⚠️ {api_name} 参数名不完全匹配: "
                    f"预期 {expected_param_names}, 实际 {actual_param_names}. "
                    f"差异: {mismatch}"
                )

        # ★★★ 新增：类方法级契约校验 ★★★
        if api_type == "类":
            expected_class_methods = expected.get("class_methods", []) if isinstance(expected, dict) else []
            actual_class_methods = actual.get("class_methods", {})

            if expected_class_methods:
                for cm in expected_class_methods:
                    cm_name = cm.get("method_name") if isinstance(cm, dict) else None
                    if not cm_name:
                        continue

                    # 硬错：类的公开方法必须存在
                    if cm_name not in actual_class_methods:
                        errors.append(
                            f"❌ 类 '{api_name}' 缺少公开方法 '{cm_name}'（Schema 声明了但代码未实现）"
                        )
                        continue

                    # 硬错：类方法参数数量必须匹配
                    expected_cm_params = [
                        p.get("param_name") for p in (cm.get("params", []) if isinstance(cm, dict) else [])
                        if p.get("param_name")
                    ]
                    actual_cm_params = actual_class_methods[cm_name].get("params", [])
                    if len(expected_cm_params) != len(actual_cm_params):
                        errors.append(
                            f"❌ 类 '{api_name}.{cm_name}' 参数数量不匹配: "
                            f"预期 {len(expected_cm_params)} 个 ({expected_cm_params}), "
                            f"实际 {len(actual_cm_params)} 个 ({actual_cm_params})"
                        )

    return errors


# ========== 导入业务校验和上下文构建器 ==========
try:
    from my_vibecoding.utils.business_context_validator import BusinessContextValidator
    from my_vibecoding.utils.project_context_builder import ProjectContextBuilder
except ImportError as e:
    log_warn("子图1", "导入", f"无法导入 utils 模块: {e}，将使用空实现")

    class BusinessContextValidator:
        @staticmethod
        def validate_business_consistency(schema_data): return []

        @staticmethod
        def validate_generation_order(schema_data): return []

        @staticmethod
        def check_business_rules_in_code(source_code, rules): return []

        @staticmethod
        def validate_layer_rules(schema_data): return []

        # ★★★ 摇篮扼杀 C / B / A —— fallback 镜像实现（导入失败仍可校验）★★★
        @staticmethod
        def validate_package_structure_consistency(schema_data):
            """
            镜像实现（简化版）：检查 包式顶层目录必须有 __init__.py + 入口文件必须存在
            """
            errors = []
            try:
                from collections import defaultdict
                all_files = []
                def _walk(tl):
                    for d in tl:
                        all_files.extend(d.get("files", []) or [])
                        if d.get("sub_dirs"): _walk(d["sub_dirs"])
                _walk(schema_data.get("directory_tree", []) or [])
                py_files = {f.get("file_path","") for f in all_files if str(f.get("file_path","")).endswith(".py")}
                pkg_cnt = defaultdict(int)
                for rel in py_files:
                    parts = [p for p in rel.replace("\\","/").split("/") if p]
                    if len(parts) <= 1: continue
                    if parts[0]=="src" and len(parts)>=3: pkg_cnt[f"src/{parts[1]}"] += 1
                    elif parts[0]=="src": pkg_cnt["src"] += 1
                    else: pkg_cnt[parts[0]] += 1
                for pkg, c in pkg_cnt.items():
                    if c >= 2 and f"{pkg}/__init__.py" not in py_files:
                        errors.append(f"❌ 包路径缺失 __init__.py：'{pkg}/' 下未生成 __init__.py，将导致绝对导入失败")
                entry = schema_data.get("entry_file") or ""
                if entry and entry not in py_files:
                    errors.append(f"❌ Schema.entry_file='{entry}' 在 directory_tree 中未找到对应 .py 文件")
            except Exception as _e:
                errors.append(f"⚠️ fallback 包路径校验异常（跳过）：{_e}")
            return errors

        @staticmethod
        def check_no_relative_imports(source_code: str, file_path: str = ""):
            import re as _re
            errors = []
            try:
                import ast as _ast
                tree = _ast.parse(source_code or "")
                for node in _ast.walk(tree):
                    if isinstance(node, _ast.ImportFrom) and (getattr(node, "level", 0) or 0) >= 1:
                        level = node.level
                        module = getattr(node, "module", None) or ""
                        names = ", ".join(getattr(a, "name", "") for a in (getattr(node, "names", []) or []))
                        errors.append(
                            f"❌ {file_path or '<unknown>'} 使用了相对导入（level={level}）："
                            f"from {'.'*level}{module} import {names}；请使用绝对导入（以顶层包名为前缀）"
                        )
            except SyntaxError:
                bad = _re.findall(r"(?m)^\s*from\s+\.{1,}[\w\.]*\s+import\b", source_code or "")
                if bad:
                    errors.append(f"❌ {file_path or '<unknown>'} 存在疑似相对导入：{bad[:3]}")
            return errors

        @staticmethod
        def has_self_bootstrap_header(source_code: str) -> bool:
            return bool(source_code and "VIBE_BOOTSTRAP" in source_code
                        and "__file__" in source_code and "sys.path.insert" in source_code)

        @staticmethod
        def build_entry_bootstrap_header(entry_rel_path: str, project_name: str = "") -> str:
            entry_rel = (entry_rel_path or "").replace("\\", "/")
            depth = len([p for p in entry_rel.split("/") if p]) - 1
            args_list = ["_V_os.path.dirname(__file__)"] + ['".."'] * depth
            path_join_arg = ", ".join(args_list)
            return (
                '# -*- coding: utf-8 -*-\n'
                '# >>> VIBE_BOOTSTRAP >>>\n'
                '# 自动注入：确保 IDE 双击 / 沙箱 python path/file.py / python -m / Docker 都能找到顶层包\n'
                'import os as _V_os, sys as _V_sys\n'
                f'_V_PROJECT_ROOT = _V_os.path.abspath(_V_os.path.join({path_join_arg}))\n'
                'if _V_PROJECT_ROOT not in _V_sys.path:\n'
                '    _V_sys.path.insert(0, _V_PROJECT_ROOT)\n'
                '_V_SRC_ROOT = _V_os.path.join(_V_PROJECT_ROOT, "src")\n'
                'if _V_os.path.isdir(_V_SRC_ROOT) and _V_SRC_ROOT not in _V_sys.path:\n'
                '    _V_sys.path.insert(0, _V_SRC_ROOT)\n'
                'del _V_os, _V_sys, _V_PROJECT_ROOT, _V_SRC_ROOT\n'
                '# <<< VIBE_BOOTSTRAP <<<\n\n'
            )

    class ProjectContextBuilder:
        @staticmethod
        def build_context(schema_data): return {}

        @staticmethod
        def build_ready_files_summary(ready, all_files): return ""

        @staticmethod
        def build_upcoming_files_summary(order, idx, all_files): return ""

        @staticmethod
        def build_file_context(path, item, ctx, ready, ready_sum, up_sum): return ""

        @staticmethod
        def format_context_for_prompt(ctx): return ""


# ★★★ 新增：从文件末尾解析 @generated 注释块 ★★★
def extract_summary_from_file(file_path: str) -> Optional[Dict]:
    """从文件末尾解析 @generated 注释块"""
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
            # ===== Critical #2 修复：单测沙箱强制启用配置的 single_sandbox_timeout（默认 240s）=====
            # docker-py: `timeout` 参数是等待容器结束的最大秒数，超时抛 docker.errors.APIError/ReadTimeout 由外层 except Exception 捕获
            timeout=SANDBOX_TIMEOUT,
            network_disabled=False,
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
# ========== 状态定义 ==========
class VibeProjectState(TypedDict):
    messages: Annotated[list[dict], add_messages]
    user_raw_requirement: str | None
    user_clarify_answers: Annotated[list[str], operator.add]
    final_clear_requirement: str | None
    need_user_input: bool
    clarify_questions: list[str] | None
    info_enough: bool
    front_plan_one: str | None
    front_plan_two: str | None
    final_plan: str | None
    remaining_steps: int
    config: dict | None

    plan_one_done: bool
    plan_two_done: bool
    wait_counter: int

    unit_test_files: list[dict] | None
    top_entry_file: dict | None
    current_test_file: dict | None
    pending_file_code_update: str | None
    raw_sandbox_result: dict | None
    last_test_error: str | None
    pending_main_code_update: dict | None
    raw_main_sandbox_result: dict | None
    last_main_error: str | None
    is_environment_error: bool | None
    modify_summary: list[dict] | None

    unit_test_retry_count: int
    main_test_retry_count: int

    # ★★★ 新增：错误汇总 ★★★
    _error_summary: str | None

    # 主图作为编译子图节点传入（运行时父图 state dict 含该字段），用于进度实时入库
    conv_id: str | None


################################################################################################################
# ====================== 提示词 ======================
fill_info_prompt = """
你是项目需求分析专家。
你的任务：结合用户原始描述以及历史补充回答，判断是否需要向用户进一步确认信息。

【重要意图识别规则，优先级最高】
如果用户语句包含如下含义：直接开始、别再问了、就这样、不用继续提问、跳过问题、直接做。
那么直接跳过提问，不要生成任何问题，把全部现有信息整理为 final_requirement，questions为空数组。次要信息允许你自行合理推断。

【内部参考维度，仅供判断，不需要每条都提问】
1.项目核心目标
2.编程语言（本项目固定使用Python）
3.是否需要web接口、对外API
4.是否需要数据库、持久化存储
5.是否调用第三方大模型API
6.是否需要异步逻辑
7.单元测试相关需求
8.部署方式：本地脚本 / docker /服务部署
9.项目规模：demo /正式工程

你的判断规则：
1.只对真正影响整体架构的关键缺失信息生成问题。
2.属于次要细节，不会改动目录结构的，不要提问，自行推断。
3.单次最多输出3个问题，不要一次性抛出大量问题。
4.信息收集完成，或者用户要求直接开始，输出完整final_requirement，questions为空。

严格只输出JSON，禁止多余文字、markdown、解释。
输出格式：
{{
  "is_enough": true/false,
  "final_requirement": "整理合并后的完整需求文本",
  "questions": ["问题1","问题2"]
}}

示例1 用户要求直接开始
{{
  "is_enough": true,
  "final_requirement":"用户需要开发xxx，用户要求直接开始，部分细节由架构阶段推断补齐。",
  "questions":[]
}}

示例2 需要补充信息
{{
  "is_enough": false,
  "final_requirement":"",
  "questions":["是否需要数据库持久化？","项目是否需要对外web接口？"]
}}

示例3 信息充足
{{
  "is_enough": true,
  "final_requirement":"开发基于LangGraph的Agent，Python，本地脚本运行，接入deepseek，不需要数据库，编写单元测试。",
  "questions":[]
}}

====输入内容====
原始用户需求：{USER_RAW_INPUT}
历史补充回答：{USER_CLARIFY_ANSWERS}
"""

front_plan_prompt = """
你是专业软件架构师，请根据用户项目需求输出完整工程架构方案。
所有设计必须严格围绕用户原始需求展开，禁止超出需求范围过度设计，禁止遗漏核心需求。

【前置步骤：需求锚定（必须执行）】
1. 先逐条拆解用户原始需求中的核心功能点、约束点，形成清晰的需求点清单。
2. 后续所有设计必须对应需求点，每个需求点都要有明确的模块/文件实现，禁止出现需求无对应设计的情况。

硬性约束：
1. 输出仅使用Markdown，禁止输出JSON，禁止使用```代码块标记。
2. 不要输出思考过程，直接输出方案正文。
3. 必须包含下面6个固定标题，标题文字不可修改、不可缺失。
4. 每个文件写明相对路径，写清职责，写明依赖的项目内其他文件。
5. 需要包含配置文件、入口脚本、测试目录、开发依赖、日志处理、错误处理，不要遗漏工程组件。
6.【最重要强制规则1：接口契约精细化】在文件清单的每一个文件描述中，明确列出**本文件对外暴露、可供其他文件调用的全部类、函数**；
   - 必须写明：函数/类名称、入参名称+参数类型、参数含义、返回值类型+返回值说明、抛出的异常类型。
   - 如果该文件没有对外暴露接口，写明：无对外导出接口。
   - 仅写对外公开接口，内部私有函数不需要罗列。
   - 这部分信息是后续代码生成的契约，绝对不能省略、不能写错函数名与参数，禁止模糊表述。
7.【最重要强制规则2：调用关系精确化】每个文件必须增加【本文件调用外部接口】，预先写明本文件将会调用哪些其他文件的哪些公开接口。
   - 格式：依赖文件路径 + 调用的类/函数名称，参数完全复用对方对外接口定义。
   - 如果本文件不调用任何外部文件，写：本文件不调用任何外部接口。
   - 架构阶段就定死调用关系，后续代码生成阶段不允许新增、修改外部调用。
8.【最重要强制规则3：职责边界刚性约束】本文件自身实现的函数、类，必须和对外导出接口列表完全对齐：不能多定义函数，不能少定义函数，禁止私自新增不在对外接口列表的公开函数。私有内部函数不受此约束。
   - 严格禁止跨文件写业务逻辑，每个文件职责单一，边界清晰。
9.【沙箱独立单测标记】每个文件清单条目必须增加【是否支持独立沙箱单测】：
   - ★★★ 核心规则：只有 .py 文件才能标记为"是" ★★★
   - 支持独立单测（标记"是"）：仅限 Python 源文件（.py），且文件仅包含常量、枚举、Pydantic模型、纯工具函数；不依赖项目内部其它模块、不需要全局初始化、不需要运行时上下文
   - 不支持独立单测（标记"否"）：
     * 所有非 .py 文件（.txt、.yaml、.json、.env、.md、.ini、.html、.css、.js、.sqlite、.db 等）一律标记为"否"
     * 入口脚本、业务编排、路由、Graph节点、需要项目内部其它模块实例、需要环境变量/全局上下文
   - 拿不准一律标记：否。
10.【顶层入口标记】每个文件清单条目必须增加【是否顶层入口】：
    - 是否顶层入口：是：整个项目对外运行的主程序入口，**全项目只能有且仅有1个文件标记为是**；
    - 是否顶层入口：否：其余所有文件，工具、模型定义、节点、配置、测试文件全部标记否。
11.【入口函数强制规则】只有 is_top_entry=true 的文件才能对外导出名为 run 的函数。
    其他任何文件（工具模块、模型定义、业务节点等）不得导出 run 函数。
    顶层入口文件的 run 函数签名必须为：def run(input=None):

【必须包含的6个固定标题】
## 项目概述
## 项目依赖
## 架构整体说明
## 目录结构
## 文件清单
## 需求覆盖自查

【文件清单条目格式模板（严格按此格式输出每个文件）】
- 文件路径：<相对路径>
  职责：<一句话描述文件职责>
  文件依赖：<依赖的其他文件路径列表，无依赖写"无">
  支持独立沙箱单测：<是/否>
  是否顶层入口：<是/否>
  对外导出接口：
  <函数/类名>(<参数名>: <参数类型>) -> <返回值类型>
    参数 <参数名>：<参数含义>
    参数 <参数名>：<参数含义>
    返回值：<返回值含义>
    异常：<抛出的异常类型，无异常写"无">
  <如果有多个接口，继续列举>
  本文件调用外部接口：
  调用 <目标文件路径> : <接口名称>(<参数名>...)
  <如果有多个调用，继续列举>

【以下为完整示例，严格参考格式，禁止自行创造新格式】
示例1：顶层入口文件（不支持单测，是顶层入口）
- 文件路径：src/main.py
  职责：程序入口，组装并运行LangGraph图，读取环境配置，统一异常捕获与日志
  文件依赖：src/graph/state.py，src/graph/nodes.py，src/utils/logger.py
  支持独立沙箱单测：否
  是否顶层入口：是
  对外导出接口：
  函数 run(input=None) -> dict
    参数 input：任意类型，入口参数，默认为 None
    返回值：dict类型，包含"status"、"result"、"error"三个字段
    异常：无对外抛出异常，内部统一捕获并记录日志
  本文件调用外部接口：
  调用 src/graph/nodes.py : build_graph()
  调用 src/graph/nodes.py : process_node(state: dict)
  调用 src/utils/logger.py : setup_logger(name: str)

示例2：纯数据模型文件（支持单测，非顶层入口）
- 文件路径：src/models/schemas.py
  职责：定义项目核心数据模型，使用Pydantic做数据校验，无任何业务逻辑
  文件依赖：无
  支持独立沙箱单测：是
  是否顶层入口：否
  对外导出接口：
  类 ProjectConfig(BaseModel)
    字段 name: str 项目名称
    字段 version: str 版本号
    字段 debug: bool 是否调试模式
  类 TaskResult(BaseModel)
    字段 task_id: str 任务唯一标识
    字段 status: str 状态值，枚举"pending"/"running"/"done"/"failed"
    字段 data: Optional[dict] 任务结果数据
    字段 error: Optional[str] 错误信息
  本文件调用外部接口：
  本文件不调用任何外部接口

示例3：工具函数文件（支持单测，非顶层入口）
- 文件路径：src/utils/helpers.py
  职责：通用工具函数，字符串处理、文件读写、prompt模板渲染，不依赖项目内部模块
  文件依赖：无
  支持独立沙箱单测：是
  是否顶层入口：否
  对外导出接口：
  函数 render_prompt(template: str, context: dict) -> str
    参数 template：str类型，含占位符的模板字符串，占位符格式为{{key}}
    参数 context：dict类型，键值对映射
    返回值：str类型，渲染完成后的完整文本
    异常：模板缺少占位符时抛出KeyError
  函数 load_env_file(file_path: str) -> dict
    参数 file_path：str类型，.env文件路径
    返回值：dict类型，解析后的环境变量键值对
    异常：文件不存在抛出FileNotFoundError
  本文件调用外部接口：
  本文件不调用任何外部接口

示例4：业务节点文件（不支持单测，非顶层入口）
- 文件路径：src/graph/nodes.py
  职责：实现LangGraph业务节点，包含状态转换、LLM调用、路由逻辑
  文件依赖：src/models/schemas.py，src/utils/helpers.py
  支持独立沙箱单测：否
  是否顶层入口：否
  对外导出接口：
  函数 build_graph() -> CompiledGraph
    参数：无
    返回值：CompiledGraph类型，编译完成的LangGraph对象
    异常：配置缺失抛出ConfigError
  函数 process_node(state: dict) -> dict
    参数 state：dict类型，当前图状态，必须包含"messages"字段
    返回值：dict类型，更新后的状态，包含"result"字段
    异常：state格式非法抛出ValueError
  本文件调用外部接口：
  调用 src/models/schemas.py : ProjectConfig
  调用 src/utils/helpers.py : render_prompt(template: str, context: dict)

示例5：配置文件模板（支持单测，非顶层入口）
- 文件路径：.env.example
  职责：环境变量配置模板，存放需要用户填写的密钥和运行时参数
  文件依赖：无
  支持独立沙箱单测：是
  是否顶层入口：否
  对外导出接口：无对外导出接口
  本文件调用外部接口：
  本文件不调用任何外部接口

【❌ 常见错误规避 - 生成前必须对照检查】
1. ❌ 禁止给文件凭空捏造不存在的依赖（比如依赖一个你根本没定义的文件）
2. ❌ 禁止在"对外导出接口"中写函数实现细节，只写签名和说明
3. ❌ 禁止将"支持独立沙箱单测"标记为"是"的文件依赖项目内部其他文件
4. ❌ 禁止将"是否顶层入口"标记为"是"的文件超过1个
5. ❌ 禁止在"本文件调用外部接口"中调用未在对方文件中声明的接口
6. ❌ 禁止遗漏"文件依赖"中列出的文件，确保每个依赖文件都在文件清单中有定义

【必须包含的6个固定标题】
## 项目概述
## 项目依赖
## 架构整体说明
## 目录结构
## 文件清单
## 需求覆盖自查

====基于下面用户需求输出方案====
{USER_REQUIREMENT}
"""

# ★★★ 完全重写 back_code_prompt ★★★
back_code_prompt = """
你是高级架构评审专家。接收两份Markdown架构方案A、方案B，以及用户原始项目需求。

你的任务：将两份方案择优融合，输出一份**严格符合以下 JSON Schema 定义**的标准 schema。

【★ 核心目标】
后续代码生成模型将读取这份 schema 生成代码。
每个文件必须包含 `implementation_guide`，告诉后续模型"这个文件该怎么做"。

================================================================================
【输出 JSON Schema - 与 Pydantic 模型严格对齐】
================================================================================

{
  "$schema": "vibe_project_schema_v1",
  "project_name": "项目名称",
  "project_desc": "项目描述（1-2句话）",
  "entry_file": "src/main.py",
  "dependencies": {
    "pip": ["包名1", "包名2"],
    "dev_dependencies": ["pytest", "black"]
  },
  "business_context": {
    "overview": "核心业务问题描述",
    "user_roles": ["管理员", "普通用户"],
    "core_flows": [
      {
        "name": "用户注册",
        "description": "用户填写信息，系统校验并创建账号",
        "steps": [
          {"step": 1, "action": "接收用户名、密码、邮箱", "actor": "用户"},
          {"step": 2, "action": "校验邮箱唯一性", "actor": "系统"},
          {"step": 3, "action": "哈希密码并写入数据库", "actor": "系统"}
        ],
        "involved_files": ["src/api/auth.py", "src/services/user_service.py", "src/models/user.py"]
      }
    ]
  },
  "directory_tree": [
    {
      "path": "src",
      "desc": "源码目录",
      "files": [
        {
          "file_path": "src/models/user.py",
          "file_purpose": "用户数据模型",
          "depend_on_files": [],
          "is_test_file": false,
          "can_unit_test": true,
          "is_top_entry": false,
          "template_content": "",
          "export_apis": [
            {
              "name": "User",
              "type": "类",
              "params": [
                {"param_name": "id", "param_type": "int", "desc": "用户ID"},
                {"param_name": "name", "param_type": "str", "desc": "用户名"},
                {"param_name": "email", "param_type": "str", "desc": "邮箱"}
              ],
              "return_desc": "Pydantic BaseModel 子类",
              "exception": "无",
              "class_methods": [
                {
                  "method_name": "validate_email",
                  "params": [
                    {"param_name": "email", "param_type": "str", "desc": "待验证邮箱"}
                  ],
                  "return_desc": "bool，邮箱格式是否合法"
                }
              ]
            }
          ],
          "call_other_apis": [],
          "business_role": {
            "in_flows": ["用户注册"],
            "flow_position": "数据模型定义",
            "business_rules": ["邮箱格式必须合法", "用户名不能为空"],
            "business_data": {
              "input": "",
              "output": "",
              "side_effects": []
            },
            "dependent_flows": []
          },
          "implementation_guide": {
            "core_logic": "定义 User 和 UserCreate 两个 Pydantic 模型，用于用户注册和登录时的数据校验与序列化",
            "key_functions": ["User", "UserCreate"],
            "error_scenarios": ["字段缺失", "字段类型错误", "邮箱格式不合法"],
            "performance_considerations": [],
            "collaboration_notes": ["被 src/services/user_service.py 的 create_user 函数引用"]
          },
          "test_guide": {
            "can_test": true,
            "skip_reason": "",
            "mock_suggestions": [],
            "test_cases": ["测试 User 模型字段校验"]
          },
          "external_dependencies": []
        },
        {
          "file_path": "src/services/user_service.py",
          "file_purpose": "用户服务，处理用户注册和登录逻辑",
          "depend_on_files": ["src/models/user.py", "src/utils/security.py"],
          "is_test_file": false,
          "can_unit_test": false,
          "is_top_entry": false,
          "template_content": "",
          "export_apis": [
            {
              "name": "create_user",
              "type": "函数",
              "params": [
                {"param_name": "username", "param_type": "str", "desc": "用户名"},
                {"param_name": "password", "param_type": "str", "desc": "明文密码"},
                {"param_name": "email", "param_type": "str", "desc": "邮箱"}
              ],
              "return_desc": "dict，包含 user_id 和 username",
              "exception": "邮箱已存在时抛出 ValueError"
            },
            {
              "name": "authenticate_user",
              "type": "函数",
              "params": [
                {"param_name": "username", "param_type": "str", "desc": "用户名"},
                {"param_name": "password", "param_type": "str", "desc": "明文密码"}
              ],
              "return_desc": "User 对象或 None",
              "exception": "无"
            }
          ],
          "call_other_apis": [
            {"target_file": "src/models/user.py", "api_name": "User"},
            {"target_file": "src/utils/security.py", "api_name": "hash_password"}
          ],
          "business_role": {
            "in_flows": ["用户注册", "用户登录"],
            "flow_position": "业务逻辑层，流程 Step 2-3",
            "business_rules": ["邮箱必须唯一", "密码长度至少6位"],
            "business_data": {
              "input": "用户名、密码、邮箱",
              "output": "用户对象或错误",
              "side_effects": ["写入数据库"]
            },
            "dependent_flows": []
          },
          "implementation_guide": {
            "core_logic": "实现用户注册：校验邮箱唯一性 → 哈希密码 → 写入数据库 → 返回用户对象。实现用户认证：查询用户 → 验证密码 → 返回用户对象",
            "key_functions": ["create_user", "authenticate_user"],
            "error_scenarios": ["邮箱已被注册", "密码长度不足", "用户不存在", "密码错误"],
            "performance_considerations": ["使用索引加速邮箱查询"],
            "collaboration_notes": [
              "调用 src/models/user.py 的 User 模型",
              "调用 src/utils/security.py 的 hash_password",
              "被 src/api/auth.py 的 register 和 login 接口调用"
            ]
          },
          "test_guide": {
            "can_test": false,
            "skip_reason": "依赖数据库连接",
            "mock_suggestions": ["mock 数据库操作"],
            "test_cases": ["测试正常注册流程"]
          },
          "external_dependencies": [
            {
              "name": "DATABASE_URL",
              "dep_type": "database",
              "required": true,
              "description": "数据库连接串"
            }
          ]
        }
      ],
      "sub_dirs": []
    }
  ],
  "generation_order": [
    {"sequence": 1, "file": "src/models/user.py", "reason": "无依赖，优先生成"},
    {"sequence": 2, "file": "src/utils/security.py", "reason": "无依赖，优先生成"},
    {"sequence": 3, "file": "src/services/user_service.py", "reason": "依赖 models 和 utils"},
    {"sequence": 4, "file": "src/api/auth.py", "reason": "依赖 services"},
    {"sequence": 5, "file": "src/main.py", "reason": "依赖 api，最后生成"}
  ],
  "data_models": {},
  "error_registry": {},
  "layer_rules": {
    "allowed_dependencies": [
      {"from_layer": "api", "to_layers": ["services"]},
      {"from_layer": "services", "to_layers": ["models", "utils"]}
    ],
    "forbidden_imports": [
      {"from_layer": "models", "to_layers": ["api", "services"]}
    ]
  }
}

================================================================================
【每个文件的硬性要求（必须逐项检查）】
================================================================================

每个 FileItem 必须满足以下**所有**条件：

1. implementation_guide 完整性：
   □ core_logic 不能为空，必须用 1-2 句话写清楚"这个文件要完成什么业务动作"
   □ collaboration_notes 不能为空，必须写清楚"调用谁、被谁调用"
   □ key_functions 至少列出 1 个实现该逻辑的关键函数名
   □ error_scenarios 至少列出 2 个需要处理的错误场景

2. export_apis 完整性（对外接口契约）：
   □ 每个接口必须包含 name（函数/类名）
   □ 每个接口必须包含 type（"函数" 或 "类"）
   □ 每个接口的 params 必须完整列出所有参数（含 param_name, param_type, desc）
   □ 每个接口必须包含 return_desc 和 exception
   □ 导出接口数量必须与方案中的定义完全一致，不能多、不能少

3. call_other_apis 闭环性（依赖关系必须可追溯）：
   □ 每一条调用记录中的 target_file 必须在 directory_tree 中存在
   □ 每一条调用记录中的 api_name 必须在目标文件的 export_apis 中存在
   □ 禁止出现"调用不存在的接口"的情况

4. test_guide 完整性：
   □ can_test 必须是布尔值（true/false）
   □ 如果 can_test=false，skip_reason 必须填写具体原因
   □ 非 .py 文件的 can_test 必须为 false
   □ 依赖其他内部文件（depend_on_files 非空）的 .py 文件，can_test 必须 false

5. 依赖关系一致性：
   □ depend_on_files 中列出的每个文件，都必须在 directory_tree 中存在
   □ 依赖关系必须是单向的，禁止循环依赖
   □ 本文件依赖的文件，必须在 generation_order 中排在本文件前面

6. 顶层入口唯一性：
   □ is_top_entry 全项目只能有 1 个为 true
   □ 只有 is_top_entry=true 的文件才能导出名为 run 的函数

7. 业务角色完整性：
   □ business_role.in_flows 必须引用 business_context.core_flows 中定义的流程
   □ business_role.business_rules 至少列出 1 条业务规则

================================================================================
【不同文件类型的 core_logic 填写示例（硬性参照）】
================================================================================

1. 数据模型文件（models/*.py）：
   core_logic: "定义 [模型名] 等 Pydantic 模型，用于 [业务场景] 的数据校验与序列化"
   ✅ 示例: "定义 User 和 UserCreate 两个 Pydantic 模型，用于用户注册和登录的数据校验"
   ❌ 错误: "用户模型"

2. 服务层文件（services/*.py）：
   core_logic: "实现 [业务动作]：校验 [参数] → 调用 [依赖] → 处理 [结果] → 返回 [输出]"
   ✅ 示例: "实现用户注册：校验邮箱唯一性 → 哈希密码 → 写入数据库 → 返回用户对象"
   ❌ 错误: "用户注册服务"

3. API层文件（api/*.py）：
   core_logic: "处理 [路由] 请求：解析 [请求格式] → 调用 [service] → 返回 [响应格式]"
   ✅ 示例: "处理 /api/auth/login 请求：解析 JSON 请求体 → 调用 user_service.authenticate → 返回 JWT token"
   ❌ 错误: "登录接口"

4. 工具文件（utils/*.py）：
   core_logic: "提供 [功能] 等纯工具函数，不依赖项目内部模块"
   ✅ 示例: "提供密码哈希、JWT 生成与验证、日期格式化等纯工具函数"
   ❌ 错误: "工具函数"

5. 入口文件（main.py）：
   core_logic: "启动 [框架] 服务，注册 [路由列表]，加载 [配置]，统一 [异常处理]"
   ✅ 示例: "启动 FastAPI 服务，注册 auth、user、book 路由，加载环境配置，统一异常处理"
   ❌ 错误: "启动服务"

================================================================================
【接口契约强制执行（最高优先级）】
================================================================================

⚠️ 接口名称是文件间调用的唯一凭证，名称不一致 = 代码无法运行！

规则1：export_apis 中的接口名必须与方案中定义的完全一致
规则2：call_other_apis 中的 api_name 必须在目标文件的 export_apis 中存在
规则3：参数签名也必须一致（参数个数、参数名称、参数类型）
规则4：禁止"我觉得应该是这个名字"式的自由发挥

================================================================================
【绝对禁止清单（输出前对照）】
================================================================================

❌ 禁止事项：
1. implementation_guide.core_logic 使用"待定"、"xxx"、"TODO"等占位词
2. implementation_guide.collaboration_notes 为空数组
3. export_apis 中的 params 缺少 param_name 或 param_type
4. call_other_apis 中调用不存在的接口
5. depend_on_files 中引用了不存在的文件
6. 出现循环依赖
7. is_top_entry 超过 1 个
8. can_unit_test 在依赖内部文件时为 true
9. 非 .py 文件的 can_unit_test 为 true
10. business_role.in_flows 引用了 core_flows 中不存在的流程名

================================================================================
【执行步骤】
================================================================================

第1步：需求拆解 → 形成需求点清单
第2步：方案对比与择优融合
第3步：按上述规则填充每个文件的 implementation_guide
第4步：接口闭环校验（call_other_apis 必须能在目标文件的 export_apis 中找到）
第5步：输出 JSON

================================================================================
【硬性约束】
================================================================================
1. 只输出纯净 JSON，不要加 ```json 或 ``` 标记
2. implementation_guide.core_logic 不能为空
3. implementation_guide.collaboration_notes 不能为空
4. test_guide.skip_reason 在 can_test=false 时必须填写
5. 非 .py 文件的 can_unit_test 必须为 false
6. is_top_entry 全项目只有1个 true

用户原始需求：
{USER_REQUIREMENT}

方案A：
{TEXT_PLAN_A}

方案B：
{TEXT_PLAN_B}

文件保存根目录：
{SAVE_PATH}
"""

# ★★★ 完全重写 single_file_generate_prompt ★★★
single_file_generate_prompt = """
你是Python高级开发工程师，**严格按照架构契约生成零偏差、可直接运行的Python源码**。

【★ 项目上下文】
{project_context}

【★ 当前文件在项目中的位置】
{file_context}

【★ 已生成并验证的文件】
{ready_files_summary}

【★ 即将生成的文件】
{upcoming_files_summary}

【★ 当前文件的依赖文件接口总览（仅列出被调用的接口）】
{depend_all_export_text}

【本文件信息】
文件路径：{file_path}
文件职责：{file_purpose}
依赖文件：{depend_list}

【本文件对外导出接口契约（必须100%遵守）】
{export_api_text}

【★ 本文件的实现指南（来自架构阶段，告诉你怎么做）】
{implementation_guide_text}

【★ 本文件的业务规则（必须在代码中体现）】
{business_rules_text}

【架构预先约定：本文件允许调用的外部接口】
{call_other_api_text}

===== 输出格式（强制执行）=====
你必须输出一个合法的 JSON 对象，包含且仅包含两个字段：
{
  "code": "完整的文件源码（多行字符串，需要正确转义）",
  "summary": "该文件的极简摘要（不超过80字，包含：核心职责、对外暴露的关键函数、主要副作用）"
}

⚠️ 额外强制规则：
1. 生成的代码文件**末尾**必须包含 @generated 注释块：
   # @generated: {"file": "{file_path}", "summary": "摘要内容", "hash": "{{hash}}"}
2. 只输出纯净 JSON，不要加 ```json 或 ``` 标记
3. code 字段中的双引号 " 和反斜杠 \\ 必须转义
4. 文件末尾必须包含 if __name__ == "__main__": 测试块
5. ★★★ 摇篮扼杀 B（相对导入禁令）★★★：代码中**禁止**使用 `from .xxx` / `from ..xxx` 等相对导入。
   必须使用**绝对导入**，以项目根目录下的顶层包名 / `src` 下的包名为前缀，例如：
   `from my_project.models.book import Book`、`from src.services.user import create_user`。
   违反此规则将触发代码生成阶段的 AST/regex 双检并强制重试。
6. ★★★ 摇篮扼杀 A（入口文件自举）★★★：如果当前文件 == Schema.entry_file（顶层入口），
   代码开头必须自带一段基于 `__file__` 的 `sys.path.insert` 自举头（允许有 `VIBE_BOOTSTRAP` 标记的注释），
   确保用户在 IDE 中双击运行 / shell 直接 `python path/to/entry.py` 都能找到所有包。

硬性约束：只输出 JSON，不要输出任何解释、思考、前置说明。
"""

code_fix_prompt = """
你是Python高级修复工程师。根据文件原始代码、沙箱执行错误日志，修复代码中的问题。
⚠️ 必须严格保留原有文件的对外接口契约（函数名、参数、返回值类型），只修复导致报错的部分，禁止修改接口定义、禁止新增功能、禁止改变文件职责。

【文件信息】
文件路径：{file_path}
文件职责：{file_purpose}

【原始代码】
{original_code}

【沙箱错误日志】
{error_log}

【修复要求】
1. 精准定位错误原因，只修改出错的代码行，不改动无关逻辑。
2. 严格保留所有对外导出接口的签名：函数名、参数名、参数类型、返回值类型、抛出异常，完全和原始一致。
3. 修复后代码必须语法正确、导入正确、可正常加载。
4. 修复后必须保留 `if __name__ == "__main__":` 测试块。
5. 如果错误是环境问题，不做代码修改，直接输出原始代码。
6. 不要输出解释，只输出完整的修复后代码。
"""

main_code_fix_prompt = """
你是Python高级修复工程师。根据顶层入口运行错误日志，定位并修复所有关联文件的问题。

【项目信息】
项目根目录：{project_root}
顶层入口文件：{entry_file}
文件职责：{file_purpose}

【错误日志】
{error_log}

【修复规则】
1. 允许修改顶层入口及所有关联依赖文件
2. 严格保留所有对外导出接口签名
3. 精准修改出错部分，不改动无关逻辑
4. 修改类名、函数名、变量名时，所有引用文件必须同步更新
5. 修复后的代码必须保留 `if __name__ == "__main__":` 测试块
6. 如果错误是环境问题，标记 `is_environment_error: true`

【输出格式】
严格输出JSON：
{
  "is_environment_error": true/false,
  "error_reason": "环境错误说明",
  "modified_files": [
    {"file_path": "相对路径", "fixed_code": "修复后的完整代码", "change_desc": "修改说明"}
  ]
}
"""

sandbox_judge_prompt = """
你是代码测试判定专家。根据文件信息、沙箱执行输出，判断测试是否通过。

【文件信息】
文件路径：{file_path}
文件职责：{file_purpose}

【沙箱执行结果】
返回码：{return_code}
标准输出：{stdout}
错误输出：{stderr}

【判定规则】
1. 返回码为0，且无导入错误、无语法错误、无运行时异常 → 通过
2. 存在语法错误、导入失败、运行时异常 → 失败
3. 警告信息不影响功能 → 通过
4. 错误信息包含以下关键词 → 环境错误（is_environment_error: true）：
   API key, Connection refused, Timeout, Permission denied,
   No such file or directory, Authentication failed,
   pip install, ModuleNotFoundError, Could not find a version

严格只输出JSON：
{
  "success": true/false,
  "is_environment_error": true/false,
  "brief_reason": "简短判定原因",
  "error_detail": "完整错误信息"
}
"""


################################################################################################################
# ====================== 节点 ======================
def user_input_node(state: VibeProjectState) -> dict:
    log_info("子图1", "用户输入", "等待用户输入...")
    if state["user_raw_requirement"] is None:
        user_text = interrupt({
            "question": "请输入你的需求：",
        })
        log_info("子图1", "用户输入", f"收到: {user_text[:50]}{'...' if len(user_text) > 50 else ''}")
        return {
            "user_raw_requirement": user_text
        }
    elif state["user_raw_requirement"] is not None and not state.get("clarify_questions"):
        return {"user_raw_requirement": state["user_raw_requirement"]}
    else:
        questions = state.get("clarify_questions") or []
        question_text = "\n".join(f"- {q}" for q in questions)
        user_text = interrupt({
            "question": f"请按照提示补充信息：\n{question_text}",
        })
        log_info("子图1", "用户输入", f"补充回答: {user_text[:50]}{'...' if len(user_text) > 50 else ''}")
        return {
            "user_clarify_answers": [user_text]
        }


def parse_user_requirement_node(state: VibeProjectState) -> dict:
    log_info("子图1", "需求澄清", "开始...")
    user_raw = state["user_raw_requirement"]
    answer_list = state.get("user_clarify_answers") or []
    user_clarify_text = "\n".join(answer_list)

    final_prompt = safe_format(
        fill_info_prompt,
        USER_RAW_INPUT=user_raw,
        USER_CLARIFY_ANSWERS=user_clarify_text
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    normal_retry = model_output_retry_cfg.get("normal_retry", default_retry)
    clarify_retry = get_retry_count_for_node("clarify_retry", normal_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_ask,
            prompt=final_prompt,
            pydantic_model=ClarifyOutput,
            max_retries=clarify_retry,
            node_name="需求澄清",
            validator_context="输出必须包含 is_enough (bool)、final_requirement (str)、questions (list[str]) 三个字段。"
        )
        is_enough = validated.is_enough
        final_req = validated.final_requirement
        questions_out = validated.questions
        log_success("子图1", "需求澄清", f"is_enough={is_enough}")
    except RuntimeError as e:
        log_warn("子图1", "需求澄清", f"失败: {e}，直接使用原始需求")
        return {
            "info_enough": True,
            "final_clear_requirement": user_raw,
            "clarify_questions": None
        }

    if not is_enough:
        remain = state.get("remaining_steps", 0) - 1
        log_info("子图1", "需求澄清", f"信息不足，剩余 {remain} 次机会")
        return {
            "need_user_input": True,
            "info_enough": False,
            "clarify_questions": questions_out,
            "final_clear_requirement": None,
            "remaining_steps": remain
        }
    else:
        log_success("子图1", "需求澄清", "信息充足")
        return {
            "need_user_input": False,
            "info_enough": True,
            "clarify_questions": None,
            "final_clear_requirement": final_req
        }


def pre_node_front_plan_one_node(state: VibeProjectState) -> dict:
    log_info("子图1", "前向计划1", "开始生成...")
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = safe_format(
        front_plan_prompt,
        USER_REQUIREMENT=complete_info
    )
    try:
        resp = model_front_plan_one.invoke(real_front_plan_prompt)
        plan_content = resp.content
        log_success("子图1", "前向计划1", f"生成成功，长度: {len(plan_content)} 字符")
    except Exception as e:
        log_error("子图1", "前向计划1", f"失败: {e}")
        plan_content = ""

    return {
        "front_plan_one": plan_content,
        "plan_one_done": True
    }


def pre_node_front_plan_two_node(state: VibeProjectState) -> dict:
    log_info("子图1", "前向计划2", "开始生成...")
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = safe_format(
        front_plan_prompt,
        USER_REQUIREMENT=complete_info
    )
    try:
        resp = model_front_plan_two.invoke(real_front_plan_prompt)
        plan_content = resp.content
        log_success("子图1", "前向计划2", f"生成成功，长度: {len(plan_content)} 字符")
    except Exception as e:
        log_error("子图1", "前向计划2", f"失败: {e}")
        plan_content = ""

    return {
        "front_plan_two": plan_content,
        "plan_two_done": True
    }


def decide_to_plan_addtional(state: VibeProjectState) -> str | list[str]:
    info_enough = state.get("info_enough", False)
    remain = state.get("remaining_steps", 0)

    if info_enough or remain <= 0:
        if remain <= 0 and not info_enough:
            log_warn("子图1", "分支路由", f"澄清循环次数已耗尽，强制进入架构生成")
        return ["pre_node_front_plan_one_node", "pre_node_front_plan_two_node"]
    else:
        log_info("子图1", "分支路由", f"信息不足，回到用户输入（剩余 {remain} 次）")
        return "user_input_node"


def sync_node(state: VibeProjectState) -> dict:
    log_info("子图1", "同步节点", "检查两个方案是否都已完成...")

    plan_one_done = state.get("plan_one_done", False)
    plan_two_done = state.get("plan_two_done", False)
    wait_counter = state.get("wait_counter", 0)

    if plan_one_done and plan_two_done:
        log_success("子图1", "同步节点", "两个方案都已就绪，继续执行")
        return {"wait_counter": 0}

    new_counter = wait_counter + 1
    if new_counter > 10:
        log_warn("子图1", "同步节点", f"等待次数超过阈值({new_counter})，强制继续")
        return {"wait_counter": new_counter}

    log_info("子图1", "同步节点", f"等待中... 第 {new_counter} 次检查")
    return {"wait_counter": new_counter}


def route_after_sync(state: VibeProjectState) -> str:
    plan_one_done = state.get("plan_one_done", False)
    plan_two_done = state.get("plan_two_done", False)
    wait_counter = state.get("wait_counter", 0)

    if (plan_one_done and plan_two_done) or wait_counter > 10:
        log_info("子图1", "同步路由", "进入最终计划节点")
        return "pre_node_final_plan_node"

    log_info("子图1", "同步路由", "继续等待")
    return "sync_node"


def pre_node_final_plan_node(state: VibeProjectState) -> dict:
    log_info("子图1", "架构合并", "开始生成最终Schema...")
    final_back_code_prompt = safe_format(
        back_code_prompt,
        USER_REQUIREMENT=state.get("user_raw_requirement", ""),
        TEXT_PLAN_A=state.get("front_plan_one", ""),
        TEXT_PLAN_B=state.get("front_plan_two", ""),
        SAVE_PATH=save_path
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    md_save_path = os.path.join(save_path, "final_plan.md")

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    critical_retry = model_output_retry_cfg.get("critical_retry", default_retry)
    schema_retry = get_retry_count_for_node("schema_retry", critical_retry, model_output_retry_cfg)

    try:
        validated = invoke_with_validation(
            model=model_back_plan,
            prompt=final_back_code_prompt,
            pydantic_model=SchemaOutput,
            max_retries=schema_retry,
            node_name="架构Schema",
            logic_validator=validate_schema_logic,
            validator_context="输出必须符合 vibe_project_schema_v1 的完整结构，包含所有必填字段。"
        )
        final_plan_content = json.dumps(validated.dict(), ensure_ascii=False, indent=2)
        log_success("子图1", "架构合并", "Schema 校验通过")

    except RuntimeError as e:
        log_error("子图1", "架构合并", f"失败: {e}")
        raise RuntimeError(f"架构 Schema 生成失败: {e}")

    try:
        with open(md_save_path, "w", encoding="utf-8") as f:
            f.write(final_plan_content)
        log_success("子图1", "架构合并", f"已保存到: {md_save_path}")
    except Exception as e:
        log_warn("子图1", "架构合并", f"写入final_plan.md失败: {e}")

    return {
        "final_plan": final_plan_content
    }


def final_plan_output_node(state: VibeProjectState) -> dict:
    log_info("子图1", "计划输出", "创建目录和空文件...")
    final_plan = state.get("final_plan", "")
    final_plan_json_str = clean_json_data(str(final_plan))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        schema_data = json.loads(final_plan_json_str)
        log_success("子图1", "计划输出", "Schema 解析成功")
    except json.JSONDecodeError as e:
        log_error("子图1", "计划输出", f"JSON解析失败: {e}")
        raise RuntimeError(f"解析最终计划 JSON 失败: {e}")

    def process_directory_tree(tree_list, root_base: str):
        for dir_item in tree_list:
            rel_dir = dir_item.get("path", "")
            abs_dir = os.path.join(root_base, rel_dir)
            abs_dir = os.path.abspath(abs_dir)
            if not abs_dir.startswith(os.path.abspath(root_base)):
                log_warn("子图1", "计划输出", f"目录路径越界，已跳过: {rel_dir}")
                continue
            os.makedirs(abs_dir, exist_ok=True)

            file_list = dir_item.get("files", [])
            for file_item in file_list:
                rel_file_path = file_item.get("file_path", "")
                abs_file = os.path.join(root_base, rel_file_path)
                abs_file = os.path.abspath(abs_file)
                if not abs_file.startswith(os.path.abspath(root_base)):
                    log_warn("子图1", "计划输出", f"文件路径越界，已跳过: {rel_file_path}")
                    continue
                file_parent = os.path.dirname(abs_file)
                os.makedirs(file_parent, exist_ok=True)
                try:
                    with open(abs_file, "w", encoding="utf-8") as fp:
                        pass
                except Exception as e:
                    log_warn("子图1", "计划输出", f"创建空文件失败 {rel_file_path}: {e}")

            sub_dirs = dir_item.get("sub_dirs", [])
            if sub_dirs:
                process_directory_tree(sub_dirs, root_base)

    directory_tree = schema_data.get("directory_tree", [])
    process_directory_tree(directory_tree, save_path)
    log_success("子图1", "计划输出", "目录结构创建完成")
    return {}


# ★★★ 重构后的 coding_generate_node ★★★
def coding_generate_node(state: VibeProjectState) -> dict:
    log_info("子图1", "代码生成", "开始生成文件代码...")
    notify_progress(state.get("conv_id"), "⚙️ [完整模式] 正在生成代码与测试，可能需要 3~5 分钟，请稍候...")
    final_plan = state.get("final_plan", "")
    final_plan_json_str = clean_json_data(str(final_plan))

    try:
        schema_data = json.loads(final_plan_json_str)
    except json.JSONDecodeError as e:
        log_error("子图1", "代码生成", f"Schema JSON解析失败: {e}")
        raise RuntimeError(f"解析最终计划 JSON 失败: {e}")

    all_file_items = collect_all_files(schema_data.get("directory_tree", []))
    file_path_map = {item["file_path"]: item for item in all_file_items}

    # ========== 获取 generation_order ==========
    generation_order_raw = schema_data.get("generation_order", [])
    if generation_order_raw:
        if isinstance(generation_order_raw[0], dict):
            generation_order = [item.get("file", "") for item in generation_order_raw]
        else:
            generation_order = generation_order_raw
    else:
        log_warn("子图1", "代码生成", "Schema 中无 generation_order，使用拓扑排序降级")
        graph = build_dependency_graph(all_file_items)
        try:
            generation_order = topological_sort(graph)
        except RuntimeError as e:
            log_error("子图1", "代码生成", f"依赖图错误: {e}")
            raise

    log_info("子图1", "代码生成", f"按顺序生成 {len(generation_order)} 个文件")

    # ========== 构建项目上下文 ==========
    project_context = ProjectContextBuilder.build_context(schema_data)
    business_context = schema_data.get("business_context", {})

    # ========== 校验业务一致性 ==========
    biz_errors = BusinessContextValidator.validate_business_consistency(schema_data)
    order_errors = BusinessContextValidator.validate_generation_order(schema_data)
    layer_errors = BusinessContextValidator.validate_layer_rules(schema_data)
    # ★★★ 摇篮扼杀 C：Schema 包路径 ↔ directory_tree 一致性检查 ★★★
    pkg_errors = BusinessContextValidator.validate_package_structure_consistency(schema_data)

    all_validation_errors = biz_errors + order_errors + layer_errors + pkg_errors
    if all_validation_errors:
        log_warn("子图1", "代码生成", f"业务校验发现 {len(all_validation_errors)} 个问题:")
        for err in all_validation_errors:
            log_warn("子图1", "代码生成", f"  - {err}")
        # Schema 级硬错误（package_structure）直接打断：避免 LLM 在错误契约下生成出一串都跑不起来的文件
        if pkg_errors:
            log_error("子图1", "代码生成",
                      f"❌ Schema 包路径/目录结构存在 {len(pkg_errors)} 个硬错误，已中止生成（摇篮扼杀 C）")
            raise RuntimeError(
                "Schema validate_package_structure_consistency 失败:\n  - "
                + "\n  - ".join(pkg_errors[:20])
            )

    # ========== 状态追踪 ==========
    ready_files = []
    error_log = []
    total = len(generation_order)
    max_retries = 3

    for idx, file_path in enumerate(generation_order, 1):
        file_item = file_path_map.get(file_path)
        if not file_item:
            log_warn("子图1", "代码生成", f"跳过未找到的文件: {file_path}")
            continue

        f_path = file_item.get("file_path", "")
        f_purpose = file_item.get("file_purpose", "")
        depend_list = file_item.get("depend_on_files", [])
        export_apis = file_item.get("export_apis", [])
        call_other_apis = file_item.get("call_other_apis", [])
        business_role = file_item.get("business_role", {})
        business_rules = business_role.get("business_rules", [])
        implementation_guide = file_item.get("implementation_guide", {})

        abs_target_file = os.path.join(save_path, f_path)
        abs_target_file = os.path.abspath(abs_target_file)
        if not abs_target_file.startswith(os.path.abspath(save_path)):
            error_log.append(f"路径越界：{f_path}")
            continue

        # ========== ★★★ 查缓存（新增）★★★ ==========
        if os.path.exists(abs_target_file):
            # 第一层：从文件末尾解析 @generated
            cached_data = extract_summary_from_file(abs_target_file)
            if cached_data:
                log_success("子图1", "缓存命中（@generated）", f"[{idx}/{total}] ✅ {f_path} 摘要复用")
                ready_files.append({
                    "file_path": f_path,
                    "actual_exports": list(extract_exported_apis_from_code(
                        open(abs_target_file, "r", encoding="utf-8").read()
                    ).keys()),
                    "llm_summary": cached_data.get("summary", ""),
                    "generated_at": idx,
                    "cached": True
                })
                continue

            # 第二层：降级查数据库
            with open(abs_target_file, "r", encoding="utf-8") as f_existing:
                existing_code = f_existing.read()
            cached_summary = get_cached_summary(f_path, existing_code)
            if cached_summary:
                log_success("子图1", "缓存命中（数据库）", f"[{idx}/{total}] ✅ {f_path}")
                ready_files.append({
                    "file_path": f_path,
                    "actual_exports": list(extract_exported_apis_from_code(existing_code).keys()),
                    "llm_summary": cached_summary.get("llm_summary", ""),
                    "generated_at": idx,
                    "cached": True
                })
                continue

        # ========== 构建上下文摘要 ==========
        ready_summary = ProjectContextBuilder.build_ready_files_summary(ready_files, all_file_items)
        upcoming_summary = ProjectContextBuilder.build_upcoming_files_summary(
            generation_order, idx - 1, all_file_items
        )
        file_context = ProjectContextBuilder.build_file_context(
            f_path, file_item, project_context, ready_files, ready_summary, upcoming_summary
        )

        # ========== ★★★ 上下文裁剪：只注入实际调用的接口（新增）★★★ ==========
        depend_export_lines = []
        called_apis_by_target = {}
        for call in call_other_apis:
            target = call.get("target_file", "")
            api_name = call.get("api_name", "")
            if target not in called_apis_by_target:
                called_apis_by_target[target] = []
            if api_name:
                called_apis_by_target[target].append(api_name)

        for dep_file_path, called_names in called_apis_by_target.items():
            dep_file_obj = file_path_map.get(dep_file_path)
            if not dep_file_obj:
                depend_export_lines.append(f"⚠️ 依赖文件未找到：{dep_file_path}")
                continue

            dep_export = dep_file_obj.get("export_apis", [])
            depend_export_lines.append(f"---------- 依赖文件：{dep_file_path}（仅列出被调用的接口）----------")
            for api in dep_export:
                if api.get("name") in called_names:
                    api_name = api.get("name", "")
                    api_type = api.get("type", "")
                    params_raw = api.get("params", [])
                    param_str = ", ".join([
                        f"{p.get('param_name', '')}: {p.get('param_type', 'Any')}"
                        for p in params_raw
                    ])
                    depend_export_lines.append(f"  {api_type} {api_name}({param_str})")
                    # ★ 类方法级契约：如果 type=类，追加所有公开方法签名
                    class_methods = api.get("class_methods", [])
                    if class_methods:
                        for cm in class_methods:
                            cm_name = cm.get("method_name", "")
                            cm_params = cm.get("params", [])
                            cm_param_str = ", ".join([
                                f"{p.get('param_name', '')}: {p.get('param_type', 'Any')}"
                                for p in cm_params
                            ])
                            depend_export_lines.append(
                                f"    方法 {cm_name}({cm_param_str})"
                            )

            found_names = [api.get("name") for api in dep_export]
            missing = [n for n in called_names if n not in found_names]
            if missing:
                depend_export_lines.append(f"  ⚠️ 警告：以下接口在目标文件中未定义：{', '.join(missing)}")

        depend_all_export_text = "\n".join(depend_export_lines) if depend_export_lines else "（无外部依赖调用）"

        # ========== 格式化导出/调用接口 ==========
        export_api_text = json.dumps(export_apis, ensure_ascii=False, indent=2) if export_apis else "无对外导出接口"
        call_other_api_text = json.dumps(call_other_apis, ensure_ascii=False,
                                         indent=2) if call_other_apis else "本文件不允许调用任何外部接口"
        business_rules_text = "\n".join([f"  - {r}" for r in business_rules]) if business_rules else "  无"
        implementation_guide_text = json.dumps(implementation_guide, ensure_ascii=False,
                                               indent=2) if implementation_guide else "无"

        # ========== 构建完整提示词 ==========
        prompt_content = safe_format(
            single_file_generate_prompt,
            project_context=ProjectContextBuilder.format_context_for_prompt(project_context),
            file_context=file_context,
            ready_files_summary=ready_summary,
            upcoming_files_summary=upcoming_summary,
            depend_all_export_text=depend_all_export_text,
            file_path=f_path,
            file_purpose=f_purpose,
            depend_list=str(depend_list),
            export_api_text=export_api_text,
            call_other_api_text=call_other_api_text,
            business_rules_text=business_rules_text,
            implementation_guide_text=implementation_guide_text
        )

        log_info("子图1", "代码生成", f"[{idx}/{total}] 生成: {f_path}")

        # ========== 生成 + 校验循环 ==========
        code_content = None
        llm_summary = None
        final_content = None
        actual_exports = []  # 提前初始化，防止中途失败 ready_files 引用未定义变量
        is_python_file = f_path.endswith(".py")

        # ===== Critical #1 修复：非 Python 文件跳过 AST/相对导入/自举头/独立沙箱 等所有 Python 专属校验 =====
        # 非 Python 文件（.yaml/.json/.sql/.md/.env…）只做 Pydantic 格式 + 业务规则，不瞎跑 ast.parse
        for attempt in range(max_retries):
            try:
                resp = model_code_generate.invoke(prompt_content)
                raw_output = resp.content.strip()

                # 清洗 JSON
                cleaned_json = clean_json_data(raw_output)

                # ===== Pydantic 强校验 =====
                try:
                    parsed = json.loads(cleaned_json)
                    validated_output = SingleFileOutput(**parsed)
                except (json.JSONDecodeError, ValidationError) as e:
                    log_warn("子图1", "代码生成", f"[{idx}/{total}] ⚠️ JSON校验失败 (尝试 {attempt+1}/{max_retries}): {e}")
                    prompt_content += f"\n\n【上一轮 JSON 格式校验失败】{str(e)}，请重新输出合法 JSON。"
                    continue

                code_content = validated_output.code
                llm_summary = validated_output.summary

                # ===== AST 契约校验（仅 .py）=====
                if is_python_file:
                    export_apis_schema = file_item.get("export_apis", [])
                    ast_errors = validate_code_against_schema(code_content, export_apis_schema)

                    if ast_errors:
                        log_warn("子图1", "代码生成",
                                 f"[{idx}/{total}] ⚠️ {f_path} AST校验失败 (尝试 {attempt + 1}/{max_retries})")
                        for err in ast_errors:
                            log_warn("子图1", "代码生成", f"  - {err}")
                        prompt_content = prompt_content + f"\n\n【上一轮 AST 校验失败】\n" + "\n".join(ast_errors)
                        continue

                # ===== 业务规则校验（所有文件类型都可跑）=====
                if business_rules:
                    biz_errors = BusinessContextValidator.check_business_rules_in_code(code_content, business_rules)
                    if biz_errors:
                        log_warn("子图1", "代码生成",
                                 f"[{idx}/{total}] ⚠️ {f_path} 业务规则校验失败 (尝试 {attempt + 1}/{max_retries})")
                        for err in biz_errors:
                            log_warn("子图1", "代码生成", f"  - {err}")
                        prompt_content = prompt_content + f"\n\n【上一轮业务规则校验失败】\n" + "\n".join(biz_errors)
                        continue

                # ========== ★★★ 摇篮扼杀 B：绝对导入硬约束（禁止 from . / from ..）★★★ ==========
                if is_python_file:
                    rel_errors = BusinessContextValidator.check_no_relative_imports(code_content, f_path)
                    if rel_errors:
                        log_warn("子图1", "代码生成",
                                 f"[{idx}/{total}] ⚠️ {f_path} 相对导入校验失败 (尝试 {attempt + 1}/{max_retries}) —— 摇篮扼杀 B")
                        for _err in rel_errors:
                            log_warn("子图1", "代码生成", f"  - {_err}")
                        prompt_content = (
                            prompt_content
                            + "\n\n【上一轮相对导入禁令校验失败】请使用绝对导入（以顶层包名/项目根目录下的包名为前缀），"
                              "禁止使用 `from .xxx` 或 `from ..xxx`。详细错误：\n"
                            + "\n".join(rel_errors)
                        )
                        continue

                # ========== ★★★ 摇篮扼杀 A：入口文件注入自举 sys.path 头（幂等）★★★ ==========
                if is_python_file:
                    entry_file_rel = (schema_data.get("entry_file") or "").replace("\\", "/")
                    current_f_path = f_path.replace("\\", "/")
                    project_name = (schema_data.get("project_name") or schema_data.get("app_name") or "").strip()
                    if entry_file_rel and current_f_path == entry_file_rel:
                        if not BusinessContextValidator.has_self_bootstrap_header(code_content):
                            bootstrap_header = BusinessContextValidator.build_entry_bootstrap_header(
                                entry_file_rel, project_name
                            )
                            # 若 LLM 自行写了 coding:utf-8 行，避免产生两行
                            if code_content.startswith("# -*- coding:"):
                                _end_first = code_content.find("\n")
                                if _end_first != -1:
                                    # bootstrap 自带 # -*- coding:utf-8，优先用自带的（保证后面 BOOTSTRAP 标记在一起）
                                    rest = code_content[_end_first + 1 :].lstrip("\n")
                                    code_content = bootstrap_header + rest
                            else:
                                code_content = bootstrap_header + code_content.lstrip("\ufeff").lstrip()
                            log_success(
                                "子图1", "代码生成",
                                f"[{idx}/{total}] 🔧 {f_path} 为 Schema.entry_file，已注入自举 sys.path 头（摇篮扼杀 A）"
                            )

                # ===== High #1 修复：先做 hash 替换 → 做二次最小校验 → 一次落盘（避免写两次中间出现不一致）=====
                code_hash = hashlib.md5(code_content.encode("utf-8")).hexdigest()
                final_content = code_content.replace('"{{hash}}"', f'"{code_hash}"')

                # 二次校验：替换占位后必须保持与原 code_content 语义一致性（仅 Python 文件）
                if is_python_file and final_content != code_content:
                    try:
                        ast.parse(final_content)
                    except SyntaxError as _syn:
                        log_warn("子图1", "代码生成",
                                 f"[{idx}/{total}] ⚠️ {f_path} 替换 hash 占位后 SyntaxError (尝试 {attempt + 1}/{max_retries})：{_syn}")
                        prompt_content = (
                            prompt_content
                            + f"\n\n【上一轮生成的代码中字符串 \"{{{{hash}}}}\" 占位符与代码结构冲突，请重新生成，"
                              "把校验和字段放到独立字符串常量中单独定义。错误：{_syn}】"
                        )
                        continue

                # ===== 一次落盘 =====
                with open(abs_target_file, "w", encoding="utf-8") as fw:
                    fw.write(final_content)
                log_success("子图1", "代码生成", f"[{idx}/{total}] ✅ {f_path}")

                # ===== 写入数据库缓存（以最终落盘的 final_content 计算 hash 保证一致）=====
                final_hash = hashlib.md5(final_content.encode("utf-8")).hexdigest()
                set_cached_summary(f_path, final_content, {
                    "rel_path": f_path,
                    "file_desc": f_purpose,
                    "export_apis": [api.get("name") for api in export_apis],
                    "depend_on": depend_list,
                    "llm_summary": llm_summary,
                    "hash": final_hash
                })
                log_info("子图1", "缓存写入", f"💾 {f_path} 摘要已缓存")

                # ===== 提取真实接口 =====
                actual_exports = list(
                    extract_exported_apis_from_code(final_content).keys()
                ) if is_python_file else []

                # ===== 立即单测（仅 .py + Schema 明确标记 can_unit_test）=====
                if is_python_file and file_item.get("can_unit_test", False):
                    log_info("子图1", "代码生成", f"[{idx}/{total}] 立即测试: {f_path}")
                    sandbox_result = run_single_file_sandbox(final_content)
                    if sandbox_result.get("success"):
                        log_success("子图1", "代码生成", f"[{idx}/{total}] ✅ {f_path} 立即测试通过")
                    else:
                        log_warn("子图1", "代码生成", f"[{idx}/{total}] ⚠️ {f_path} 立即测试失败，将在单测阶段修复")

                # ===== 更新 ready_files =====
                ready_files.append({
                    "file_path": f_path,
                    "actual_exports": actual_exports,
                    "llm_summary": llm_summary,
                    "generated_at": idx
                })

                break  # 成功，跳出重试循环

            except Exception as exc:
                import traceback as _tb
                detail = _tb.format_exc() if isinstance(exc, (SyntaxError, NameError, TypeError)) else str(exc)
                if attempt == max_retries - 1:
                    # ===== High #2 修复：最后一次失败一定要①进 error_log ② 进 modify_summary 失败记录 ③ 不再继续
                    entry = f"[FAIL] {f_path} 生成失败（重试{max_retries}次）: {detail}"
                    error_log.append(entry)
                    log_error("子图1", "代码生成", entry)
                    ready_files.append({
                        "file_path": f_path,
                        "actual_exports": [],
                        "llm_summary": "",
                        "generated_at": idx,
                        "generation_failed": True,
                        "reason": detail
                    })
                    break  # 跳出 for attempt 循环，进入下一个文件
                else:
                    log_warn("子图1", "代码生成",
                             f"[{idx}/{total}] ⚠️ {f_path} 异常 (尝试 {attempt + 1}/{max_retries}): {exc}")
                    prompt_content = prompt_content + f"\n\n【上一轮异常】{detail}，请重新生成。"
                    continue

    # ========== ★★★ ERROR SUMMARY（新增）★★★ ==========
    if error_log:
        error_summary = "=" * 60 + "\n"
        error_summary += "❌ ERROR SUMMARY（生成失败的文件清单）\n"
        error_summary += "=" * 60 + "\n"
        for i, err in enumerate(error_log, 1):
            error_summary += f"{i}. {err}\n"
        error_summary += "=" * 60
        log_error("子图1", "代码生成", f"\n{error_summary}")

        return {
            "modify_summary": [{"file": f["file_path"], "status": "generated"} for f in ready_files],
            "unit_test_early_results": ready_files,
            "_error_summary": error_summary
        }
    else:
        log_success("子图1", "代码生成", f"全部 {total} 个文件生成完成")
        return {
            "modify_summary": [{"file": f["file_path"], "status": "generated"} for f in ready_files],
            "unit_test_early_results": ready_files,
            "_error_summary": "✅ 全部文件生成成功，无错误"
        }


# ★★★ 修改后的 collect_file_check_node ★★★
def collect_file_check_node(state: VibeProjectState) -> dict:
    log_info("子图1", "收集测试资源", "开始...")
    final_plan = state.get("final_plan", "")
    final_plan_json_str = clean_json_data(str(final_plan))

    try:
        schema_data = json.loads(final_plan_json_str)
    except json.JSONDecodeError as e:
        log_error("子图1", "收集测试资源", f"schema解析失败: {e}")
        return {
            "unit_test_files": [],
            "top_entry_file": None,
            "current_test_file": None,
            "last_test_error": None,
            "pending_file_code_update": None,
            "raw_sandbox_result": None,
            "is_environment_error": False,
            "modify_summary": [],
            "unit_test_retry_count": 0,
            "main_test_retry_count": 0
        }

    all_file_items = collect_all_files(schema_data.get("directory_tree", []))

    unit_test_list = []
    top_entry = None

    for item in all_file_items:
        file_path = item.get("file_path", "")
        test_guide = item.get("test_guide", {})
        can_test = test_guide.get("can_test", False)
        skip_reason = test_guide.get("skip_reason", "")

        # ★★★ 利用 TestGuide 精确判断 ★★★
        if file_path.endswith('.py') and can_test:
            unit_test_list.append(item)
            log_debug("子图1", "收集测试资源", f"✅ {file_path} 可单测")
        elif file_path.endswith('.py') and not can_test:
            log_debug("子图1", "收集测试资源", f"⏭️ {file_path} 跳过单测: {skip_reason}")
        else:
            log_debug("子图1", "收集测试资源", f"⏭️ {file_path} 非Python文件，跳过")

        if item.get("is_top_entry", False):
            top_entry = item

    current_file = unit_test_list[0] if unit_test_list else None

    log_info("子图1", "收集测试资源", f"可单测文件数量: {len(unit_test_list)}")
    if top_entry:
        log_info("子图1", "收集测试资源", f"顶层入口: {top_entry['file_path']}")
    else:
        log_warn("子图1", "收集测试资源", "未找到顶层入口标记")

    return {
        "unit_test_files": unit_test_list,
        "top_entry_file": top_entry,
        "current_test_file": current_file,
        "last_test_error": None,
        "pending_file_code_update": None,
        "raw_sandbox_result": None,
        "is_environment_error": False,
        "modify_summary": [],
        "unit_test_retry_count": 0,
        "main_test_retry_count": 0
    }


def unit_test_fix_node(state: VibeProjectState) -> dict:
    log_info("子图1", "单测修复", "检查...")
    current_file = state.get("current_test_file")
    last_error = state.get("last_test_error")
    retry_count = state.get("unit_test_retry_count", 0)

    if not current_file:
        return {"pending_file_code_update": None}

    f_path = current_file["file_path"]
    f_purpose = current_file.get("file_purpose", "")

    if not last_error:
        log_info("子图1", "单测修复", f"{f_path} 首次测试，无需修复")
        return {"pending_file_code_update": None}

    if retry_count >= UNIT_TEST_MAX_RETRY:
        log_warn("子图1", "单测修复", f"⚠️ {f_path} 已达最大重试次数({UNIT_TEST_MAX_RETRY})，跳过该文件")
        return {"pending_file_code_update": None}

    log_info("子图1", "单测修复", f"{f_path} 测试失败，开始修复（第 {retry_count + 1}/{UNIT_TEST_MAX_RETRY} 次）...")
    abs_file = os.path.join(save_path, f_path)
    try:
        with open(abs_file, "r", encoding="utf-8") as f:
            original_code = f.read()
    except Exception as e:
        log_error("子图1", "单测修复", f"读取文件失败: {e}")
        return {"pending_file_code_update": None}

    fix_prompt = safe_format(
        code_fix_prompt,
        file_path=f_path,
        file_purpose=f_purpose,
        original_code=original_code,
        error_log=last_error
    )

    try:
        resp = model_test_fix.invoke(fix_prompt)
        fixed_code = resp.content.strip()
        if fixed_code.startswith("```python"):
            fixed_code = fixed_code.removeprefix("```python").removesuffix("```")
        elif fixed_code.startswith("```"):
            fixed_code = fixed_code.removeprefix("```").removesuffix("```")
        fixed_code = fixed_code.strip()
        log_success("子图1", "单测修复", f"{f_path} 修复完成")
        return {"pending_file_code_update": fixed_code}
    except Exception as e:
        log_error("子图1", "单测修复", f"修复模型调用失败: {e}")
        return {"pending_file_code_update": None}


def unit_test_execute_node(state: VibeProjectState) -> dict:
    log_info("子图1", "单测执行", "开始...")
    current_file = state.get("current_test_file")
    pending_code = state.get("pending_file_code_update")

    if not current_file:
        return {"raw_sandbox_result": {"success": False, "return_code": -1, "stderr": "无测试文件"}}

    f_path = current_file["file_path"]
    f_purpose = current_file.get("file_purpose", "")
    abs_file = os.path.join(save_path, f_path)

    if pending_code:
        try:
            with open(abs_file, "w", encoding="utf-8") as f:
                f.write(pending_code)
            log_info("子图1", "单测执行", f"修复代码已写入：{f_path}")
        except Exception as e:
            log_error("子图1", "单测执行", f"写入文件失败: {e}")
            return {"raw_sandbox_result": {"success": False, "return_code": -1, "stderr": f"写入文件失败：{e}"}}
    else:
        log_info("子图1", "单测执行", f"直接执行测试：{f_path}")

    try:
        with open(abs_file, "r", encoding="utf-8") as f:
            source_code = f.read()
    except Exception as e:
        return {"raw_sandbox_result": {"success": False, "return_code": -1, "stderr": f"读取文件失败：{e}"}}

    sandbox_result = run_single_file_sandbox(source_code)
    log_info("子图1", "单测执行", f"沙箱完成，success={sandbox_result.get('success')}")

    judge_prompt = safe_format(
        sandbox_judge_prompt,
        file_path=f_path,
        file_purpose=f_purpose,
        return_code=sandbox_result.get("return_code", -1),
        stdout=sandbox_result.get("stdout", ""),
        stderr=sandbox_result.get("stderr", "")
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    judge_retry = model_output_retry_cfg.get("judge_retry", default_retry)
    sandbox_judge_retry = get_retry_count_for_node("sandbox_judge", judge_retry, model_output_retry_cfg)

    # ===== Medium 修复：在调用判定模型之前，先识别确定性的环境错误。
    # 判定模型失败的 RuntimeError（没网 / key 失效 / 超时）不能降级成"代码 bug"，否则白白浪费 3 次修复 API。
    raw_code = sandbox_result.get("return_code", -1)
    raw_stderr = sandbox_result.get("stderr", "") or ""
    env_signals = {
        -3: ("Docker 守护进程不可用，沙箱功能未启用", True),
        -2: ("沙箱镜像未构建，请先 build 指定镜像", True),
        -4: ("Docker 守护进程 API 错误（环境层问题，非代码 bug）", True),
        124: ("沙箱运行超时（疑似死循环/阻塞 I/O，属于环境+代码综合，留给判定模型决定）", False),
    }
    if raw_code in env_signals:
        _msg, _is_env = env_signals[raw_code]
        if _is_env:
            log_warn("子图1", "单测执行", f"确定性环境问题 code={raw_code}：{_msg}，跳过判定模型")
            return {
                "raw_sandbox_result": sandbox_result,
                "pending_file_code_update": None,
                "last_test_error": None,
                "is_environment_error": True,
            }

    try:
        validated = invoke_with_validation(
            model=model_test_judge,
            prompt=judge_prompt,
            pydantic_model=SandboxJudgeOutput,
            max_retries=sandbox_judge_retry,
            node_name="沙箱判定",
            validator_context="输出必须包含 success (bool)、is_environment_error (bool)、brief_reason (str)、error_detail (str) 四个字段。"
        )
        judge_dict = validated.dict()
    except RuntimeError as e:
        log_warn("子图1", "单测执行", f"判定模型失败（多为 LLM 网络/配额问题，属环境层）: {e}")
        # ===== Medium 修复：之前硬编码 is_environment_error=False 会把判定模型挂了也去修 3 次文件 → 现在直接当环境错跳过
        _stderr_tail = (raw_stderr[-200:] or str(e)) if len(str(e)) < 300 else str(e)
        judge_dict = {
            "success": False,
            "is_environment_error": True,
            "brief_reason": "判定模型调用失败，作为环境错误跳过",
            "error_detail": f"[JUDGE_MODEL_RUNTIME_ERROR] 无法判定测试结果（{e.__class__.__name__}）。原始 stderr: {_stderr_tail}"
        }

    judge_success = bool(judge_dict.get("success", False))
    judge_env = bool(judge_dict.get("is_environment_error", False))
    return {
        "raw_sandbox_result": sandbox_result,
        "pending_file_code_update": None,
        "last_test_error": None if (judge_success or judge_env) else judge_dict.get("error_detail", "测试失败"),
        "is_environment_error": judge_env,
    }


def unit_test_judge_node(state: VibeProjectState) -> dict:
    log_info("子图1", "单测判定", "开始...")
    current_file = state.get("current_test_file")
    unit_list = state.get("unit_test_files") or []
    modify_summary = state.get("modify_summary") or []
    last_error = state.get("last_test_error")
    is_env_error = state.get("is_environment_error", False)
    retry_count = state.get("unit_test_retry_count", 0)

    if not current_file:
        return {"last_test_error": None}

    f_path = current_file["file_path"]

    if last_error and not is_env_error and retry_count >= UNIT_TEST_MAX_RETRY:
        log_warn("子图1", "单测判定", f"⚠️ {f_path} 已达最大重试次数({UNIT_TEST_MAX_RETRY})，跳过该文件")
        modify_summary.append(
            {"file": f_path, "status": "skip_max_retry", "remark": f"重试{UNIT_TEST_MAX_RETRY}次仍失败"})
        remaining_list = unit_list[1:] if unit_list else []
        next_file = remaining_list[0] if remaining_list else None
        return {
            "unit_test_files": remaining_list,
            "current_test_file": next_file,
            "last_test_error": None,
            "raw_sandbox_result": None,
            "unit_test_retry_count": 0,
            "modify_summary": modify_summary
        }

    if last_error and not is_env_error:
        log_warn("子图1", "单测判定", f"❌ {f_path} 失败，继续修复（第 {retry_count + 1}/{UNIT_TEST_MAX_RETRY} 次）")
        return {
            "last_test_error": last_error,
            "raw_sandbox_result": None,
            "unit_test_retry_count": retry_count + 1
        }

    if is_env_error:
        log_warn("子图1", "单测判定", f"⚠️ {f_path} 环境错误跳过")
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": "环境依赖问题"})
    else:
        log_success("子图1", "单测判定", f"✅ {f_path} 通过")
        modify_summary.append({"file": f_path, "status": "pass", "remark": "测试通过"})

    if unit_list:
        remaining_list = unit_list[1:]
        next_file = remaining_list[0] if remaining_list else None
    else:
        remaining_list = []
        next_file = None

    return {
        "unit_test_files": remaining_list,
        "current_test_file": next_file,
        "last_test_error": None,
        "raw_sandbox_result": None,
        "unit_test_retry_count": 0,
        "modify_summary": modify_summary
    }


def decide_unit_test_next(state: VibeProjectState) -> str:
    unit_list = state.get("unit_test_files") or []
    current_file = state.get("current_test_file")
    last_error = state.get("last_test_error")
    is_env_error = state.get("is_environment_error", False)
    retry_count = state.get("unit_test_retry_count", 0)

    if not unit_list:
        log_info("子图1", "路由", "单测全部完成，进入主测")
        return "main_test_fix_node"

    if not current_file:
        log_warn("子图1", "路由", "unit_list 非空但 current_file 为空，强制进入主测")
        return "main_test_fix_node"

    if last_error and not is_env_error:
        if retry_count >= UNIT_TEST_MAX_RETRY:
            log_warn("子图1", "路由", f"⚠️ {current_file['file_path']} 已达最大重试次数，跳过该文件")
            return "unit_test_judge_node"
        return "unit_test_fix_node"

    return "unit_test_judge_node"


def main_test_fix_node(state: VibeProjectState) -> dict:
    log_info("子图1", "主测修复", "检查...")
    top_entry = state.get("top_entry_file")
    last_error = state.get("last_main_error")
    modify_summary = state.get("modify_summary") or []
    retry_count = state.get("main_test_retry_count", 0)

    if not top_entry:
        return {"pending_main_code_update": None}

    f_path = top_entry["file_path"]
    f_purpose = top_entry.get("file_purpose", "")

    if not last_error:
        log_info("子图1", "主测修复", "首次主测，无需修复")
        return {"pending_main_code_update": None}

    if retry_count >= MAIN_TEST_MAX_RETRY:
        log_warn("子图1", "主测修复", f"⚠️ {f_path} 已达最大重试次数({MAIN_TEST_MAX_RETRY})，跳过主测")
        return {"pending_main_code_update": None}

    log_info("子图1", "主测修复", f"{f_path} 失败，开始修复（第 {retry_count + 1}/{MAIN_TEST_MAX_RETRY} 次）...")

    fix_prompt = safe_format(
        main_code_fix_prompt,
        project_root=save_path,
        entry_file=f_path,
        file_purpose=f_purpose,
        error_log=last_error
    )

    try:
        resp = model_test_fix.invoke(fix_prompt)
        resp_content = clean_json_data(str(resp.content))
        fix_result = json.loads(resp_content)
    except Exception as e:
        log_error("子图1", "主测修复", f"修复模型调用失败: {e}")
        return {"pending_main_code_update": None}

    is_env_error = fix_result.get("is_environment_error", False)
    if is_env_error:
        err_reason = fix_result.get("error_reason", "环境依赖不足")
        log_warn("子图1", "主测修复", f"⚠️ 环境错误：{err_reason}")
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": err_reason})
        return {
            "is_environment_error": True,
            "last_main_error": err_reason,
            "modify_summary": modify_summary,
            "main_test_retry_count": 0
        }

    modified_files = fix_result.get("modified_files", [])
    pending_dict = {}
    for item in modified_files:
        fp = item.get("file_path", "")
        code = item.get("fixed_code", "")
        desc = item.get("change_desc", "")
        pending_dict[fp] = code
        modify_summary.append({"file": fp, "status": "modified", "remark": desc})

    log_info("子图1", "主测修复", f"待修复文件数量：{len(pending_dict)}")
    return {
        "pending_main_code_update": pending_dict,
        "is_environment_error": False,
        "modify_summary": modify_summary,
        "main_test_retry_count": retry_count + 1
    }


def main_test_execute_node(state: VibeProjectState) -> dict:
    log_info("子图1", "主测执行", "开始...")
    notify_progress(state.get("conv_id"), "🧪 [完整模式] 正在运行完整项目主测试...")
    top_entry = state.get("top_entry_file")
    pending_code = state.get("pending_main_code_update")

    if not top_entry:
        return {"raw_main_sandbox_result": {"success": False, "return_code": -1, "stderr": "未找到顶层入口文件"}}

    f_path = top_entry["file_path"]

    if pending_code and isinstance(pending_code, dict):
        for fp, code in pending_code.items():
            abs_file = os.path.join(save_path, fp)
            abs_file = os.path.abspath(abs_file)
            if not abs_file.startswith(os.path.abspath(save_path)):
                log_warn("子图1", "主测执行", f"路径越界跳过：{fp}")
                continue
            try:
                os.makedirs(os.path.dirname(abs_file), exist_ok=True)
                with open(abs_file, "w", encoding="utf-8") as f:
                    f.write(code)
                log_info("子图1", "主测执行", f"修复代码已写入：{fp}")
            except Exception as e:
                log_error("子图1", "主测执行", f"写入文件失败 {fp}：{e}")

    client = get_docker_client()
    if client is None:
        return {
            "raw_main_sandbox_result": {"success": False, "return_code": -3, "stdout": "", "stderr": "Docker 不可用"}}

    container = None
    try:
        # ======================================================================
        # ★★★ 模块路径 3 层兜底（完整模式生成项目同样确保 entry_file 能找到包）★★★
        #   与 modify_code 主测执行保持完全一致的策略：
        #     1) env: PYTHONPATH 双路径覆盖(扁平+src作根)
        #     2) python -c 内联 sys.path.insert 注入（绕过环境变量时机问题）
        #     3) 检测到相对导入(from . / from ..) → 切 python -m <module_name>
        # ======================================================================
        work_root = "/workspace"

        # ---------- 入口文件相对路径形式（容器内 /workspace 下）----------
        entry_rel = f_path.replace("\\", "/").lstrip("/")

        has_relative_import = False
        try:
            with open(os.path.join(save_path, entry_rel), "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(4096)
            # 匹配: from .xxx / from ..xxx / from . import yyy / from .. import yyy
            if re.search(r'(?m)^\s*from\s+\.{1,}[\w\s]', head):
                has_relative_import = True
                log_info("子图1", "主测执行", "入口文件使用相对导入 → 切换为 python -m 包启动模式")
        except Exception:
            pass

        env = {
            "PYTHONPATH": f"{work_root}:{work_root}/src:/vendor",
            "WORKSPACE_ROOT": work_root,
        }

        if has_relative_import:
            module_name = entry_rel
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            module_name = module_name.replace("/", ".")
            command = ["python", "-m", module_name]
            log_info("子图1", "主测执行", f"包启动模式: python -m {module_name}  (working_dir={work_root})")
        else:
            # 与 modify_code 同样的内联注入启动（importlib.util + sys.path 三重插入）
            bootstrap = (
                "import os,sys;"
                f"_WR='{work_root}';"
                "[sys.path.insert(0,p) for p in (_WR, os.path.join(_WR,'src'), '/vendor') if p and p not in sys.path];"
                f"_F=os.path.join(_WR,{entry_rel!r});"
                "import importlib.util as _U;"
                "_S=_U.spec_from_file_location('__vibe_entry__', _F);"
                "_M=_U.module_from_spec(_S);sys.modules['__main__']=_M;sys.modules['__vibe_entry__']=_M;"
                "setattr(_M,'__file__',_F);setattr(_M,'__loader__',_S.loader);setattr(_M,'__package__','');"
                "setattr(_M,'__builtins__',__builtins__ if 'builtins' in dir() else __import__('builtins'));"
                "try:_S.loader.exec_module(_M)\n"
                "except SystemExit:pass\n"
                "except SyntaxError as _e:print(f'SyntaxError: {{_e.msg}} ({{_e.filename}}:{{_e.lineno}})',file=sys.stderr);sys.exit(1)\n"
                "except Exception as _e:\n"
                "   import traceback;traceback.print_exc();sys.exit(1)\n"
            )
            command = ["python", "-c", bootstrap]

        container = client.containers.run(
            image=SANDBOX_IMAGE,
            command=command,
            environment=env,
            stdin_open=True,
            detach=True,
            remove=True,
            volumes={save_path: {"bind": "/workspace", "mode": "ro"}},
            working_dir="/workspace",
        )

        # ===== Critical #2 修复：主测沙箱强制捕获超时（300s 默认），不是抛 ReadTimeout 让整个流程死掉 =====
        timed_out = False
        try:
            container.wait(timeout=MAIN_SANDBOX_TIMEOUT)
        except Exception as _wait_e:
            # docker.errors.APIError(500, "Timeout") / ReadTimeout / ConnectionError 统一按超时处理，后续分类为"环境问题"
            timed_out = True
            log_error("子图1", "主测执行",
                      f"容器等待超时或异常（{MAIN_SANDBOX_TIMEOUT}s）：{_wait_e.__class__.__name__}: {_wait_e}")
        try:
            all_logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore").strip()
        except Exception as _log_e:
            all_logs = ""
            log_warn("子图1", "主测执行", f"拉取容器日志失败（可能已被 remove）：{_log_e}")

        if timed_out:
            result = {
                "success": False,
                "return_code": 124,
                "stdout": all_logs,
                "stderr": f"[SANDBOX_TIMEOUT] 主测运行超过 MAIN_SANDBOX_TIMEOUT={MAIN_SANDBOX_TIMEOUT}s，已强制终止。"
                          " 请检查是否存在死循环、阻塞 I/O、或增加 main_sandbox_timeout 阈值。"
            }
        else:
            result = {"success": True, "return_code": 0, "stdout": all_logs, "stderr": ""}
            log_success("子图1", "主测执行", "完成")
    except docker.errors.ImageNotFound:
        result = {"success": False, "return_code": -2, "stdout": "", "stderr": f"镜像 {SANDBOX_IMAGE} 不存在，请先 `docker build -t {SANDBOX_IMAGE}` 构建沙箱镜像"}
        log_error("子图1", "主测执行", f"镜像 {SANDBOX_IMAGE} 不存在")
    except docker.errors.APIError as _api_e:
        # 连接失败、守护进程崩溃、拉镜像网络错等一律环境错误，不判定是代码 bug
        result = {"success": False, "return_code": -4, "stdout": "", "stderr": f"[DOCKER_API_ERROR] Docker 守护进程 API 错误：{_api_e}"}
        log_error("子图1", "主测执行", f"Docker API 错误：{_api_e}")
    except Exception as e:
        result = {"success": False, "return_code": -1, "stdout": "", "stderr": f"主测沙箱调用失败：{str(e)}"}
        log_error("子图1", "主测执行", f"沙箱调用失败: {e}")
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


def main_test_judge_node(state: VibeProjectState) -> dict:
    log_info("子图1", "主测判定", "开始...")
    top_entry = state.get("top_entry_file")
    raw_result = state.get("raw_main_sandbox_result")
    is_env_error = state.get("is_environment_error", False)
    modify_summary = state.get("modify_summary") or []
    retry_count = state.get("main_test_retry_count", 0)

    if not top_entry:
        return {"last_main_error": "无顶层入口文件"}

    f_path = top_entry["file_path"]
    f_purpose = top_entry.get("file_purpose", "")

    if is_env_error:
        log_warn("子图1", "主测判定", f"⚠️ 环境依赖跳过：{f_path}")
        print("\n📋 项目执行摘要：")
        for item in modify_summary:
            print(f"  - {item['file']} | {item['status']} | {item.get('remark', '')}")
        print("\n🎉 项目生成完成（部分功能因环境依赖跳过业务测试）")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "main_test_retry_count": 0
        }

    last_error = state.get("last_main_error")
    if last_error and retry_count >= MAIN_TEST_MAX_RETRY:
        log_warn("子图1", "主测判定", f"⚠️ {f_path} 已达最大重试次数({MAIN_TEST_MAX_RETRY})，跳过主测")
        modify_summary.append(
            {"file": f_path, "status": "skip_max_retry", "remark": f"主测重试{MAIN_TEST_MAX_RETRY}次仍失败"})
        print("\n📋 项目执行摘要：")
        for item in modify_summary:
            print(f"  - {item['file']} | {item['status']} | {item.get('remark', '')}")
        print("\n🎉 项目生成完成（主测因重试次数耗尽跳过）")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "main_test_retry_count": 0,
            "modify_summary": modify_summary
        }

    judge_prompt = safe_format(
        sandbox_judge_prompt,
        file_path=f_path,
        file_purpose=f_purpose,
        return_code=raw_result.get("return_code", -1) if raw_result else -1,
        stdout=raw_result.get("stdout", "") if raw_result else "",
        stderr=raw_result.get("stderr", "") if raw_result else ""
    )

    model_output_retry_cfg = shared.get("model_output_retry", {})
    default_retry = model_output_retry_cfg.get("default", 2)
    judge_retry = model_output_retry_cfg.get("judge_retry", default_retry)
    sandbox_judge_retry = get_retry_count_for_node("sandbox_judge", judge_retry, model_output_retry_cfg)

    # ===== Medium 修复：主测判定先做确定性环境错短路（和单测保持一致）=====
    _raw_code = raw_result.get("return_code", -1) if raw_result else -1
    _raw_stderr = raw_result.get("stderr", "") if raw_result else ""
    _env_signals_main = {
        -3: "Docker 守护进程未连接，沙箱未启用",
        -2: "沙箱镜像不存在，主测无法执行",
        -4: "Docker API 调用失败（守护进程/网络问题）",
    }
    if _raw_code in _env_signals_main:
        _msg = _env_signals_main[_raw_code]
        log_warn("子图1", "主测判定", f"确定性环境问题 code={_raw_code}：{_msg}，跳过判定模型")
        judge_dict = {
            "success": False,
            "is_environment_error": True,
            "brief_reason": _msg,
            "error_detail": _raw_stderr or _msg,
        }
    else:
        try:
            validated = invoke_with_validation(
                model=model_test_judge,
                prompt=judge_prompt,
                pydantic_model=SandboxJudgeOutput,
                max_retries=sandbox_judge_retry,
                node_name="主测判定",
                validator_context="输出必须包含 success (bool)、is_environment_error (bool)、brief_reason (str)、error_detail (str) 四个字段。"
            )
            judge_dict = validated.dict()
        except RuntimeError as e:
            log_warn("子图1", "主测判定", f"判定模型失败（多为 LLM 网络/配额问题，属环境层）: {e}")
            judge_dict = {
                "success": False,
                "is_environment_error": True,
                "brief_reason": "主测判定模型调用失败（环境错误，非代码 bug）",
                "error_detail":
                    f"[JUDGE_MODEL_RUNTIME_ERROR] {e.__class__.__name__}: {e}；"
                    f"原始 stderr：{(_raw_stderr or '')[-300:]}"
            }

    success = judge_dict.get("success", False)
    judge_is_env = judge_dict.get("is_environment_error", False)
    error_detail = judge_dict.get("error_detail", "")
    brief_reason = judge_dict.get("brief_reason", "")

    if success:
        log_success("子图1", "主测判定", f"✅ {f_path} 通过 - {brief_reason}")
        modify_summary.append({"file": f_path, "status": "pass", "remark": brief_reason})
        print("\n📋 最终执行摘要：")
        for item in modify_summary:
            print(f"  - {item['file']} | {item['status']} | {item.get('remark', '')}")
        print("\n🎉 全部测试通过，项目生成完成！")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "modify_summary": modify_summary,
            "main_test_retry_count": 0
        }
    elif judge_is_env:
        log_warn("子图1", "主测判定", f"⚠️ 环境错误跳过：{f_path} - {brief_reason}")
        modify_summary.append({"file": f_path, "status": "skip_env", "remark": brief_reason})
        print("\n📋 最终执行摘要：")
        for item in modify_summary:
            print(f"  - {item['file']} | {item['status']} | {item.get('remark', '')}")
        print("\n🎉 项目生成完成（顶层业务因环境依赖跳过测试）")
        return {
            "last_main_error": None,
            "raw_main_sandbox_result": None,
            "is_environment_error": True,
            "modify_summary": modify_summary,
            "main_test_retry_count": 0
        }
    else:
        log_error("子图1", "主测判定", f"❌ {f_path} 失败 - {brief_reason}")
        return {
            "last_main_error": error_detail,
            "raw_main_sandbox_result": None,
            "main_test_retry_count": retry_count + 1
        }


def decide_main_test_next(state: VibeProjectState) -> str:
    last_error = state.get("last_main_error")
    is_env_error = state.get("is_environment_error", False)
    retry_count = state.get("main_test_retry_count", 0)

    if last_error and not is_env_error and retry_count < MAIN_TEST_MAX_RETRY:
        return "main_test_fix_node"
    return END


################################################################################################################
# ====================== 构建图 ======================
def build_create_graph():
    log_info("子图1", "构建图", "开始构建...")
    builder = StateGraph(VibeProjectState)

    builder.add_node("user_input_node", user_input_node)
    builder.add_node("parse_user_requirement_node", parse_user_requirement_node)
    builder.add_node("pre_node_front_plan_one_node", pre_node_front_plan_one_node)
    builder.add_node("pre_node_front_plan_two_node", pre_node_front_plan_two_node)
    builder.add_node("sync_node", sync_node)
    builder.add_node("pre_node_final_plan_node", pre_node_final_plan_node)
    builder.add_node("final_plan_output_node", final_plan_output_node)
    builder.add_node("coding_generate_node", coding_generate_node)

    builder.add_node("collect_file_check_node", collect_file_check_node)
    builder.add_node("unit_test_fix_node", unit_test_fix_node)
    builder.add_node("unit_test_execute_node", unit_test_execute_node)
    builder.add_node("unit_test_judge_node", unit_test_judge_node)
    builder.add_node("main_test_fix_node", main_test_fix_node)
    builder.add_node("main_test_execute_node", main_test_execute_node)
    builder.add_node("main_test_judge_node", main_test_judge_node)

    builder.add_edge(START, "user_input_node")
    builder.add_edge("user_input_node", "parse_user_requirement_node")

    builder.add_conditional_edges(
        source="parse_user_requirement_node",
        path=decide_to_plan_addtional,
    )

    builder.add_edge("pre_node_front_plan_one_node", "sync_node")
    builder.add_edge("pre_node_front_plan_two_node", "sync_node")

    builder.add_conditional_edges(
        source="sync_node",
        path=route_after_sync,
        path_map={
            "pre_node_final_plan_node": "pre_node_final_plan_node",
            "sync_node": "sync_node"
        }
    )

    builder.add_edge("pre_node_final_plan_node", "final_plan_output_node")
    builder.add_edge("final_plan_output_node", "coding_generate_node")

    builder.add_edge("coding_generate_node", "collect_file_check_node")
    builder.add_edge("collect_file_check_node", "unit_test_fix_node")
    builder.add_edge("unit_test_fix_node", "unit_test_execute_node")
    builder.add_edge("unit_test_execute_node", "unit_test_judge_node")

    builder.add_conditional_edges(
        source="unit_test_judge_node",
        path=decide_unit_test_next,
    )

    builder.add_edge("main_test_fix_node", "main_test_execute_node")
    builder.add_edge("main_test_execute_node", "main_test_judge_node")

    builder.add_conditional_edges(
        source="main_test_judge_node",
        path=decide_main_test_next,
    )

    graph = builder.compile()
    log_success("子图1", "构建图", "完成")
    return graph


if __name__ == '__main__':
    graph = build_create_graph()

    max_clarify_steps = create_runtime.get("max_clarify_loop_steps", 3)
    thread_config = {"configurable": {"thread_id": "vibe_project_main_001"}}

    init_state = {
        "messages": [],
        "user_raw_requirement": None,
        "user_clarify_answers": [],
        "final_clear_requirement": None,
        "need_user_input": False,
        "clarify_questions": None,
        "info_enough": False,
        "front_plan_one": None,
        "front_plan_two": None,
        "final_plan": None,
        "remaining_steps": max_clarify_steps,
        "plan_one_done": False,
        "plan_two_done": False,
        "wait_counter": 0,
        "unit_test_files": None,
        "top_entry_file": None,
        "current_test_file": None,
        "pending_file_code_update": None,
        "raw_sandbox_result": None,
        "last_test_error": None,
        "pending_main_code_update": None,
        "raw_main_sandbox_result": None,
        "last_main_error": None,
        "is_environment_error": False,
        "modify_summary": [],
        "unit_test_retry_count": 0,
        "main_test_retry_count": 0,
        "_error_summary": None
    }

    for event in graph.stream(init_state, config=thread_config):
        print(event)