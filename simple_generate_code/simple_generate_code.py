import os
import json
import re
import sys
from typing import TypedDict, Annotated, Optional, Dict, List
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI

# ★★★ Provider 工厂（供应商全配置化驱动 / 新增供应商 0 Python 代码） ★★★
_PROV_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
if _PROV_DIR not in sys.path:
    sys.path.insert(0, _PROV_DIR)
try:
    from my_vibecoding.utils.provider_factory import resolve_model_credentials
    from my_vibecoding.utils.prompt_utils import safe_format
except ImportError:
    try:
        from provider_factory import resolve_model_credentials  # type: ignore
        from prompt_utils import safe_format  # type: ignore
    except Exception:
        def resolve_model_credentials(model_provider, model_name, shared_cfg=None):  # type: ignore
            if model_provider == "ali":
                return os.getenv("ALI_API_KEY", ""), model_name
            if model_provider == "deepseek":
                return os.getenv("DEEPSEEK_API_KEY", ""), model_name
            return "", model_name

        def safe_format(template, **kwargs):  # type: ignore
            return template

try:
    from my_vibecoding.utils.progress_notifier import notify_progress
except ImportError:
    try:
        from progress_notifier import notify_progress  # type: ignore
    except Exception:
        def notify_progress(conv_id, text, min_interval=8.0):  # type: ignore
            pass

# ========== 极简模式合理性硬门控（防止"极简单文件=419个shell脚本"这种离谱输出，对齐 main_graph1） ==========
SIMPLE_MAX_FILES = 3           # 极简单文件：最多 3 个文件（main / 工具 / 依赖说明），多了就不是 simple
SIMPLE_ALLOWED_SUFFIXES = (    # simple 只允许可直接运行的单文件相关格式（Python/Shell/配置/小前端）
    ".py", ".sh", ".bat", ".ps1", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".txt", ".md", ".requirements", ".csv", ".env.example",
    ".html", ".js", ".css"
)

load_dotenv()
# ========== 修复：基于 __file__ 的绝对路径，避免依赖 cwd 导致配置文件找不到 ==========
_config_script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.abspath(os.path.join(_config_script_dir, "../config_file/my_config.yaml"))

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败，请检查路径: {config_path}, 错误: {e}") from e

# Provider 工厂读取 shared.llm_providers（新增供应商在这里配置即可，不改 Python）
shared = config.get("shared", {})


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


# ========== 极简版配置 ==========
simple_config = config.get("main_runtime", {}).get("simple_mode", {})
SIMPLE_ENABLED = simple_config.get("enabled", True)

# ========== 项目保存路径 ==========
try:
    absolute_path = config["create_project_path"]
except KeyError:
    absolute_path = config.get("absolute_save_path", os.path.abspath(__file__))
script_dir = os.path.dirname(absolute_path)
base_save_path = os.path.abspath(os.path.join(script_dir, "./new_project/simple"))

# ========== 代码生成模型 ==========
def get_openai_model(model_provider: str, model_name: str):
    """Provider 工厂兼容封装（保留原函数名与签名，调用方零改动即可升级）。"""
    api_key, m_name = resolve_model_credentials(model_provider, model_name, shared)
    if not api_key:
        raise RuntimeError(f"模型 {model_provider} 的API密钥环境变量未配置！")
    return api_key, m_name

simple_gen_cfg = config.get("simple_generate_model")
if not simple_gen_cfg:
    log_warn("极简版", "配置", "未找到 simple_generate_model，回退使用 code_generate_model")
    simple_gen_cfg = config.get("code_generate_model", {})

api_key_simple, model_name_simple = get_openai_model(
    simple_gen_cfg.get("model_provider", ""),
    simple_gen_cfg.get("model_name", "")
)

model_simple_generate = ChatOpenAI(
    model=model_name_simple,
    api_key=api_key_simple,
    base_url=simple_gen_cfg.get("base_url", ""),
    temperature=simple_gen_cfg.get("temperature", 0.1),
    timeout=simple_gen_cfg.get("timeout", 30),
    max_retries=simple_gen_cfg.get("max_retries", 2)
)

log_info("极简版", "模型", f"使用模型: {model_name_simple}")


################################################################################################################
# ====================== 工具函数 ======================
def extract_project_name(code_text: str) -> str:
    lines = code_text.split('\n')
    for line in lines[:10]:
        match = re.search(r'^#\s*PROJECT:\s*([^\n]+)', line, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip()
            name = re.sub(r'[^\w\u4e00-\u9fff]', '_', raw_name)
            return name
    return "simple_project"


def parse_single_file_code(code_text: str) -> tuple[Dict[str, str], str]:
    files = {}
    lines = code_text.split('\n')
    project_name = extract_project_name(code_text)

    file_marker_pattern = r'^#\s*FILE:\s*([^\n]+)$'
    current_file = None
    current_content = []
    found_marker = False

    for line in lines:
        match = re.match(file_marker_pattern, line)
        if match:
            if current_file is not None:
                files[current_file] = '\n'.join(current_content).strip()
                current_content = []
            current_file = match.group(1).strip()
            if '.' not in current_file:
                current_file += '.py'
            found_marker = True
            continue

        if current_file is not None:
            current_content.append(line)

    if current_file is not None:
        files[current_file] = '\n'.join(current_content).strip()
        return files, project_name

    if not found_marker:
        filename = "main.py"
        for line in lines[:5]:
            match = re.search(r'#\s*(?:file|filename):\s*([^\n]+)', line, re.IGNORECASE)
            if match:
                filename = match.group(1).strip()
                break
        if filename == "main.py":
            for line in lines:
                class_match = re.match(r'^class\s+(\w+)', line)
                if class_match:
                    class_name = class_match.group(1)
                    snake = re.sub(r'(?<!^)(?=[A-Z])', '_', class_name).lower()
                    filename = f"{snake}.py"
                    break
                func_match = re.match(r'^def\s+(\w+)', line)
                if func_match:
                    filename = f"{func_match.group(1)}.py"
                    break
        if '.' not in filename:
            filename += '.py'
        files[filename] = code_text

    return files, project_name


################################################################################################################
# ====================== 状态定义 ======================
class SimpleProjectState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_requirements: str | None
    final_plan: str | None
    modify_summary: list[dict] | None
    current_mode: str | None
    generated_files: Dict[str, str] | None

    # 主图作为编译子图节点传入（运行时父图 state dict 含该字段），用于进度实时入库
    conv_id: str | None


################################################################################################################
# ====================== 提示词 ======================
simple_generate_prompt = """
你是一个Python程序员。根据用户需求，生成一个可直接运行的Python文件或最多 3 个小文件。

【用户需求】
{USER_REQUIREMENT}

【极端重要约束 — 违反直接作废】
1. 【文件数量上限】最多产出 **3 个文件**（= 极简单文件模式！超过 3 个属于滥用此模式，你会被判定失败！）
2. 【语言/格式】除非用户明确指定 shell / bat / bash / 前端，否则 99% 情况只要一个 `main.py` 搞定！
3. 禁止生成上百个重复文件（如 scripts/analyze_xxx_step.sh × 200 个），那是 complete 模式的工作，简单需求直接用 for 循环+列表参数即可
4. 如果用户的需求其实超过极简单文件（含多模块/数据库/前后端/权限），你也要遵守 3 个文件上限，做一个简化版本，不要擅自扩容

【输出要求】
1. 在代码开头用 `# PROJECT: 项目名称` 指定文件夹名（例如：`# PROJECT: calculator`）
2. 每个文件用 `# FILE: 路径/文件名.py` 标记
3. 如果是配置文件，用 `# FILE: requirements.txt` 标记，不要加 .py
4. 只输出代码，不要解释
5. 所有函数必须包含文档字符串
6. 末尾包含 if __name__ == "__main__": 测试块
7. 使用模拟数据，禁止硬编码密钥

只输出代码（包含 `# PROJECT:` 和 `# FILE:` 标记）：
"""


################################################################################################################
# ====================== 节点 ======================
def simple_generate_node(state: SimpleProjectState) -> dict:
    log_info("极简版", "生成", "开始生成代码...")
    conv_id = state.get("conv_id")
    user_req = state.get("user_requirements", "")

    notify_progress(conv_id, "⚡ [极简模式] 正在调用模型生成代码...")

    prompt = safe_format(simple_generate_prompt, USER_REQUIREMENT=user_req)

    try:
        log_info("极简版", "生成", "调用模型...")
        resp = model_simple_generate.invoke(prompt)
        code_text = resp.content.strip()

        notify_progress(conv_id, "⚡ [极简模式] 模型返回，解析文件...")

        if code_text.startswith("```python"):
            code_text = code_text.removeprefix("```python").removesuffix("```")
        elif code_text.startswith("```"):
            code_text = code_text.removeprefix("```").removesuffix("```")
        code_text = code_text.strip()

        files, project_name = parse_single_file_code(code_text)
        project_folder = project_name.replace(' ', '_')
        save_path = os.path.join(base_save_path, project_folder)
        os.makedirs(save_path, exist_ok=True)

        # ========== 硬门控1：文件数量（极简单文件最多 3 个）==========
        file_count = len(files)
        if file_count > SIMPLE_MAX_FILES:
            _bad_detail = f"检测到极简模式输出 {file_count} 个文件（超过上限 {SIMPLE_MAX_FILES} 个），这是 complete/demo 的工作量"
            log_error("极简版", "门控", _bad_detail)
            notify_progress(conv_id, f"⚠️ 模型在极简模式下输出了 {file_count} 个文件，升级到演示模式重新生成...")
            return {
                "current_mode": "demo",
                "final_plan": f"⚠️ 自动升级：模型在极简模式下输出了 {file_count} 个文件，重新走【演示模式】生成（请稍候几秒会自动重新进入生成）",
                "messages": [AIMessage(content=(
                    f"⚠️ 简单需求被模型理解成了 {file_count} 个文件（已超过极简模式上限 {SIMPLE_MAX_FILES} 个）。\n"
                    f"已自动升级到【演示模式】重新生成，请稍候 ~"
                ))],
                "modify_summary": [{"file": "upgrade", "status": "auto_upgrade", "remark": _bad_detail}],
            }

        # ========== 硬门控2：后缀黑名单（simple 绝不能出现 .sql/.db/.vue/.jsx/.tsx 这类完整工程文件） ==========
        disallowed = []
        for fp in files.keys():
            low = fp.lower()
            suffix = "." + low.rsplit(".", 1)[-1] if "." in low else ""
            if suffix not in SIMPLE_ALLOWED_SUFFIXES and not low.endswith(".example"):
                disallowed.append(fp)
        if disallowed:
            log_error("极简版", "门控", f"输出非 simple 允许的后缀: {disallowed[:8]}")
            notify_progress(conv_id, "⚠️ 输出包含不适合 simple 的文件类型，升级到演示模式...")
            first = ", ".join(disallowed[:5])
            return {
                "current_mode": "demo",
                "final_plan": f"⚠️ 自动升级：输出包含不适合 simple 的文件（{first}），已转【演示模式】重新生成",
                "messages": [AIMessage(content=(
                    f"⚠️ 输出包含不适合极简模式的文件类型：{first}。\n"
                    f"已自动升级到【演示模式】重新生成，请稍候 ~"
                ))],
                "modify_summary": [{"file": "upgrade", "status": "auto_upgrade",
                                    "remark": f"disallowed suffixes: {disallowed}"}],
            }

        log_info("极简版", "生成", f"项目: {project_folder}, 文件: {list(files.keys())}")
        notify_progress(conv_id, f"⚡ [极简模式] 解析到 {len(files)} 个文件，写入磁盘...")

        generated_files = {}
        modify_summary = []
        for file_path, code in files.items():
            abs_file = os.path.join(save_path, file_path)
            abs_file = os.path.abspath(abs_file)
            if not abs_file.startswith(os.path.abspath(save_path)):
                log_warn("极简版", "生成", f"路径越界，已跳过: {file_path}")
                continue

            os.makedirs(os.path.dirname(abs_file), exist_ok=True)
            with open(abs_file, "w", encoding="utf-8") as f:
                f.write(code)
            generated_files[file_path] = code
            log_success("极简版", "生成", f"已生成: {file_path}")

            try:
                compile(code, file_path, "exec")
                modify_summary.append({"file": file_path, "status": "generated", "remark": "语法检查通过"})
            except SyntaxError as e:
                log_error("极简版", "语法检查", f"{file_path}: {e}")
                modify_summary.append({"file": file_path, "status": "generated_with_errors", "remark": str(e)})
            except Exception:
                # 非 Python 文件（.md/.json/.sh 等）跳过 compile 语义
                modify_summary.append({"file": file_path, "status": "generated", "remark": "非Python文件，跳过语法检查"})

        notify_progress(conv_id, f"✅ [极简模式] 完成，已生成 {len(generated_files)} 个文件到 {project_folder}/")

        result_msg = f"⚡ 极简模式：已生成 {len(files)} 个文件到 {project_folder}/"
        if any(item["status"] == "generated_with_errors" for item in modify_summary):
            result_msg += "（⚠️ 部分文件存在语法错误）"

        return {
            "current_mode": "simple",
            "final_plan": result_msg,
            "generated_files": generated_files,
            "modify_summary": modify_summary,
            "messages": [AIMessage(content=result_msg)]
        }

    except Exception as e:
        log_error("极简版", "生成", f"失败: {e}")
        return {
            "current_mode": "simple",
            "final_plan": f"极简版生成失败: {str(e)}",
            "modify_summary": [{"file": "unknown", "status": "failed", "remark": str(e)}],
            "messages": [AIMessage(content=f"❌ 极简版生成失败: {str(e)}")]
        }


################################################################################################################
# ====================== 构建图 ======================
def build_simple_graph():
    log_info("极简版", "构建图", "开始构建子图...")
    builder = StateGraph(SimpleProjectState)

    builder.add_node("simple_generate_node", simple_generate_node)

    builder.add_edge(START, "simple_generate_node")
    builder.add_edge("simple_generate_node", END)

    graph = builder.compile()
    log_success("极简版", "构建图", "子图构建完成")
    return graph


if __name__ == "__main__":
    graph = build_simple_graph()
    init_state = {
        "messages": [],
        "user_requirements": "写一个计算器函数",
        "final_plan": None,
        "modify_summary": None,
        "current_mode": None,
        "generated_files": None
    }
    result = graph.invoke(init_state)
    print(result.get("final_plan"))