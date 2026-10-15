import os
import json
import re
import operator
import sys
from typing import TypedDict, Annotated, Optional, Dict, List
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

# ★★★ Provider 工厂（供应商全配置化驱动 / 新增供应商 0 Python 代码） ★★★
_PROV_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
if _PROV_DIR not in sys.path:
    sys.path.insert(0, _PROV_DIR)
try:
    from my_vibecoding.utils.provider_factory import resolve_model_credentials
    from my_vibecoding.utils.prompt_utils import safe_format
    from my_vibecoding.utils.progress_notifier import notify_progress
except ImportError:
    try:
        from provider_factory import resolve_model_credentials  # type: ignore
        from prompt_utils import safe_format  # type: ignore
        from progress_notifier import notify_progress  # type: ignore
    except Exception:
        def resolve_model_credentials(model_provider, model_name, shared_cfg=None):  # type: ignore
            if model_provider == "ali":
                return os.getenv("ALI_API_KEY", ""), model_name
            if model_provider == "deepseek":
                return os.getenv("DEEPSEEK_API_KEY", ""), model_name
            return "", model_name

        def safe_format(template, **kwargs):  # type: ignore
            return template

        def notify_progress(conv_id, text, min_interval=8.0):  # type: ignore
            pass

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


# ========== Demo模式配置 ==========
demo_config = config.get("main_runtime", {}).get("demo_mode", {})
DEMO_ENABLED = demo_config.get("enabled", True)
DEMO_MAX_RETRY = demo_config.get("max_retry", 2)
DEMO_MAX_FILES = demo_config.get("max_files", 3)
DEMO_ENABLE_SANDBOX = demo_config.get("enable_sandbox", True)
DEMO_CONTINUE_ON_FAILURE = demo_config.get("continue_on_failure", True)
DEMO_MAX_CLARIFY_ROUND = demo_config.get("max_clarify_round", 3)

# ========== 取消信号 ==========
demo_cancel_signals = config.get("mode_judge", {}).get("cancel_signals", [
    "不想写了", "算了", "取消", "别生成", "不用了", "暂停", "放弃", "算了不做了"
])

# ========== 项目保存路径（独立子目录） ==========
try:
    absolute_path = config["create_project_path"]
except KeyError:
    absolute_path = config.get("absolute_save_path", os.path.abspath(__file__))
script_dir = os.path.dirname(absolute_path)
base_save_path = os.path.abspath(os.path.join(script_dir, "./new_project/demo"))

# ========== 模型配置（Demo模式专用） ==========
def get_openai_model(model_provider: str, model_name: str):
    """Provider 工厂兼容封装（保留原函数名与签名，调用方零改动即可升级）。"""
    api_key, m_name = resolve_model_credentials(model_provider, model_name, shared)
    if not api_key:
        raise RuntimeError(f"模型 {model_provider} 的API密钥环境变量未配置！")
    return api_key, m_name

# 优先使用 demo_generate_model，否则回退到 code_generate_model
demo_gen_cfg = config.get("demo_generate_model")
if not demo_gen_cfg:
    log_warn("Demo版", "配置", "未找到 demo_generate_model，回退使用 code_generate_model")
    demo_gen_cfg = config.get("code_generate_model", {})

api_key_demo, model_name_demo = get_openai_model(
    demo_gen_cfg.get("model_provider", ""),
    demo_gen_cfg.get("model_name", "")
)

model_demo_generate = ChatOpenAI(
    model=model_name_demo,
    api_key=api_key_demo,
    base_url=demo_gen_cfg.get("base_url", ""),
    temperature=demo_gen_cfg.get("temperature", 0.2),
    timeout=demo_gen_cfg.get("timeout", 60),
    max_retries=demo_gen_cfg.get("max_retries", 2)
)

log_info("Demo版", "模型", f"使用模型: {model_name_demo} (provider: {demo_gen_cfg.get('model_provider')})")

# ========== 沙箱函数（纯语法检查） ==========
def run_single_file_sandbox(source_code: str, deps: Optional[List[str]] = None) -> dict:
    try:
        compile(source_code, "<string>", "exec")
        return {"success": True, "return_code": 0, "stdout": "语法校验通过", "stderr": ""}
    except SyntaxError as e:
        return {"success": False, "return_code": 1, "stdout": "", "stderr": f"语法错误：{str(e)}"}
    except Exception as e:
        return {"success": False, "return_code": 1, "stdout": "", "stderr": str(e)}


################################################################################################################
# ====================== 工具函数 ======================
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


def extract_project_name(code_text: str) -> str:
    lines = code_text.split('\n')
    for line in lines[:10]:
        match = re.search(r'^#\s*PROJECT:\s*([^\n]+)', line, re.IGNORECASE)
        if match:
            raw_name = match.group(1).strip()
            name = re.sub(r'[^\w\u4e00-\u9fff]', '_', raw_name)
            return name
    return "demo_project"


def parse_multifile_code(code_text: str) -> tuple[Dict[str, str], str]:
    files = {}
    pattern = r'#\s*FILE:\s*([^\n]+)\s*\n'
    parts = re.split(pattern, code_text)

    project_name = extract_project_name(code_text)

    if len(parts) <= 1:
        filename = "main.py"
        lines = code_text.split('\n')
        for line in lines[:5]:
            if '# file:' in line.lower() or '# filename:' in line.lower():
                match = re.search(r'#\s*(?:file|filename):\s*([^\n]+)', line, re.IGNORECASE)
                if match:
                    filename = match.group(1).strip()
                    break
        if '.' not in filename:
            filename += '.py'
        files[filename] = code_text
        return files, project_name

    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        if '.' not in filename:
            filename += '.py'
        code = parts[i + 1].strip()
        if code:
            files[filename] = code

    return files, project_name


def get_error_feedback(sandbox_results: dict) -> str:
    feedback = []
    for file_path, result in sandbox_results.items():
        if not result.get("success", False):
            error = result.get("stderr", "未知错误")
            feedback.append(f"文件 {file_path} 错误: {error}")
    return "\n".join(feedback) if feedback else ""


################################################################################################################
# ====================== 状态定义 ======================
class DemoProjectState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_requirements: str | None
    user_clarify_answers: Annotated[list[str], operator.add]
    final_clear_requirement: str | None
    need_user_input: bool
    clarify_questions: list[str] | None
    is_info_enough: bool
    clarify_round: int
    final_plan: str | None
    modify_summary: list[dict] | None
    current_mode: str | None

    demo_error_feedback: str | None
    demo_retry_count: int
    generated_files: Dict[str, str] | None
    sandbox_results: dict | None

    confirmed_info: Dict[str, str] | None

    # 主图作为编译子图节点传入（运行时父图 state dict 含该字段），用于进度实时入库
    conv_id: str | None


################################################################################################################
# ====================== 提示词 ======================
demo_clarify_prompt = """
你是项目需求分析专家。判断用户需求是否充足。

【核心规则】
如果用户已经说了具体的项目类型（爬虫、API、网站、工具、管理系统等），立即判定为信息充足，直接生成代码，不要再追问。

【不要追问的细节（用默认值）】
- 目标网站 → example.com
- 爬取字段 → 标题、内容
- 输出格式 → JSON
- 数据库 → SQLite
- 登录/分页/动态加载 → 简单项目不考虑

【什么时候才需要追问？】
只有当用户完全没说"要做什么"时，才问一句"你想做什么类型的项目？"
例如用户只说"帮我写一个项目"、"生成一个代码"。

【用户输入】
{USER_RAW_INPUT}

【已确认信息】
{CONFIRMED_INFO}

【历史补充回答】
{USER_CLARIFY_ANSWERS}

【输出格式】
{"is_enough": true/false, "final_requirement": "需求描述", "questions": []}
"""

demo_generate_prompt = """
你是Python高级开发工程师。根据用户需求，生成一个可直接运行的Python项目。

【用户需求】
{USER_REQUIREMENT}

【输出格式】
1. 在代码开头用 `# PROJECT: 项目名称` 指定项目文件夹名称
2. 每个文件用 `# FILE: 路径/文件名.py` 标记
3. 如果是配置文件（如 requirements.txt），用 `# FILE: requirements.txt` 标记

【质量约束】
1. 代码必须语法正确，可直接运行
2. 所有函数/类必须包含文档字符串
3. 禁止硬编码API密钥、Token、密码等敏感信息
4. 使用模拟数据代替真实数据
5. 只生成实际需要的文件，不要生成空文件

【错误反馈】
{ERROR_FEEDBACK}

只输出代码，不要输出任何解释。
"""


################################################################################################################
# ====================== 节点 ======================
def demo_clarify_node(state: DemoProjectState) -> dict:
    user_req = state.get("user_requirements", "")
    answer_list = state.get("user_clarify_answers") or []
    clarify_round = state.get("clarify_round", 0)
    confirmed = state.get("confirmed_info") or {}

    log_info("Demo版", "需求澄清", f"第 {clarify_round + 1} 轮...")

    # 检测取消信号
    if any(signal in user_req for signal in demo_cancel_signals):
        log_success("Demo版", "需求澄清", "用户取消生成")
        return {
            "need_user_input": False,
            "clarify_questions": None,
            "final_clear_requirement": None,
            "is_info_enough": False,
            "messages": [AIMessage(content="好的，已取消生成。")]
        }

    # ★★★ 主图已完成澄清 → 不再重复追问 ★★★
    # 背景：主图 understand/ask_clarify 已让用户选择模式并补充项目类型，合并进 user_requirements
    # 时带有 [用户选择]/[用户补充] 标记。此时再进 demo 澄清追问答一轮，用户会感觉"每次都要追问"。
    if "[用户补充]" in user_req or "[用户选择]" in user_req:
        log_success("Demo版", "需求澄清", "主图已完成澄清 → 直接放行")
        notify_progress(state.get("conv_id"), "✅ 需求已确认，开始生成代码...")
        return {
            "need_user_input": False,
            "clarify_questions": None,
            "final_clear_requirement": user_req,
            "is_info_enough": True,
            "messages": [AIMessage(content="✅ 需求已明确，开始生成代码...")]
        }

    # 硬检测：如果有项目类型词，直接放行
    project_types = [
        "爬虫", "网站", "API", "工具", "脚本", "管理系统", "博客", "论坛", "商城", "自动化", "数据分析",
        "游戏", "小游戏", "贪吃蛇", "俄罗斯方块", "聊天机器人", "机器人", "记账", "笔记", "待办",
        "下载器", "下载工具", "转换器", "转换工具", "可视化", "图表", "桌面应用", "终端", "CLI",
        "定时任务", "文件处理", "图片处理", "批量处理", "音乐", "视频", "天气", "翻译", "压缩", "加密",
    ]
    for pt in project_types:
        if pt in user_req:
            log_success("Demo版", "需求澄清", f"检测到项目类型词 '{pt}'，直接放行")
            notify_progress(state.get("conv_id"), f"✅ 需求已明确（{pt}），开始生成代码...")
            return {
                "need_user_input": False,
                "clarify_questions": None,
                "final_clear_requirement": user_req,
                "is_info_enough": True,
                "messages": [AIMessage(content="✅ 需求已明确，开始生成代码...")]
            }

    confirmed_text = json.dumps(confirmed, ensure_ascii=False, indent=2) if confirmed else "无"

    if clarify_round >= DEMO_MAX_CLARIFY_ROUND:
        log_warn("Demo版", "需求澄清", f"已达到最大澄清轮次({DEMO_MAX_CLARIFY_ROUND})，强制生成")
        notify_progress(state.get("conv_id"), "✅ 需求已确认，开始生成代码...")
        return {
            "need_user_input": False,
            "clarify_questions": None,
            "final_clear_requirement": user_req or "基于当前信息生成代码",
            "is_info_enough": True,
            "messages": [AIMessage(content="⚠️ 已达到最大追问次数，将基于已有信息生成代码。")]
        }

    user_clarify_text = "\n".join(answer_list)

    prompt = safe_format(
        demo_clarify_prompt,
        USER_RAW_INPUT=user_req,
        CONFIRMED_INFO=confirmed_text,
        USER_CLARIFY_ANSWERS=user_clarify_text
    )

    try:
        resp = model_demo_generate.invoke(prompt)
        cleaned = clean_json_data(resp.content)
        result = json.loads(cleaned)

        is_enough = result.get("is_enough", True)
        final_req = result.get("final_requirement", user_req)
        questions = result.get("questions", [])

    except Exception as e:
        log_error("Demo版", "需求澄清", f"模型调用失败: {e}，强制生成")
        notify_progress(state.get("conv_id"), "✅ 需求已确认，开始生成代码...")
        return {
            "need_user_input": False,
            "clarify_questions": None,
            "final_clear_requirement": user_req,
            "is_info_enough": True,
            "messages": [AIMessage(content="⚠️ 需求分析出错，将直接生成代码。")]
        }

    if not is_enough and questions:
        question_text = "\n".join([f"- {q}" for q in questions])
        log_info("Demo版", "需求澄清", f"大模型生成追问: {questions}")

        user_answer = interrupt({
            "question": f"请补充以下信息：\n{question_text}"
        })

        user_ans = str(user_answer)
        if any(signal in user_ans for signal in demo_cancel_signals):
            log_success("Demo版", "需求澄清", "用户取消生成")
            return {
                "need_user_input": False,
                "clarify_questions": None,
                "final_clear_requirement": None,
                "is_info_enough": False,
                "messages": [AIMessage(content="好的，已取消生成。")]
            }

        new_confirmed = confirmed.copy()
        if len(user_ans) > 3:
            if not new_confirmed.get("latest_answer"):
                new_confirmed["latest_answer"] = user_ans
            else:
                new_confirmed["latest_answer"] += f"\n{user_ans}"
            if any(kw in user_ans for kw in ["读者", "管理员", "用户", "角色"]):
                new_confirmed["roles"] = user_ans
            if any(kw in user_ans for kw in ["借阅", "归还", "管理", "统计"]):
                new_confirmed["features"] = user_ans
            if any(kw in user_ans for kw in ["前后端", "数据库", "API"]):
                new_confirmed["tech_stack"] = user_ans
            if len(user_ans) > 20:
                new_confirmed["general"] = user_ans

        new_answers = answer_list + [str(user_answer)]
        new_round = clarify_round + 1

        return {
            "need_user_input": False,
            "clarify_questions": None,
            "final_clear_requirement": None,
            "is_info_enough": False,
            "user_clarify_answers": new_answers,
            "clarify_round": new_round,
            "confirmed_info": new_confirmed,
            "messages": [HumanMessage(content=str(user_answer))]
        }

    log_success("Demo版", "需求澄清", "信息充足")
    notify_progress(state.get("conv_id"), "✅ 需求已确认，开始生成代码...")
    return {
        "need_user_input": False,
        "clarify_questions": None,
        "final_clear_requirement": final_req,
        "is_info_enough": True,
        "confirmed_info": confirmed,
        "messages": [AIMessage(content="✅ 需求已明确，开始生成代码...")]
    }


def demo_generate_node(state: DemoProjectState) -> dict:
    user_req = state.get("final_clear_requirement") or state.get("user_requirements", "")
    retry_count = state.get("demo_retry_count", 0)
    max_retry = DEMO_MAX_RETRY
    error_feedback = state.get("demo_error_feedback", "")

    # 检测取消信号
    if any(signal in user_req for signal in demo_cancel_signals):
        log_success("Demo版", "生成", "用户取消生成")
        return {
            "current_mode": "demo",
            "final_plan": "已取消生成",
            "messages": [AIMessage(content="好的，已取消生成。")]
        }

    if retry_count == 0:
        log_info("Demo版", "生成", "开始生成代码...")
        notify_progress(state.get("conv_id"), "⚙️ [演示模式] 正在生成代码，约 1~2 分钟，请稍候...")
    else:
        log_info("Demo版", "生成", f"第 {retry_count}/{max_retry} 次重试...")
        notify_progress(state.get("conv_id"), f"🔁 [演示模式] 第 {retry_count}/{max_retry} 次修复重试中...")

    prompt = safe_format(
        demo_generate_prompt,
        USER_REQUIREMENT=user_req,
        ERROR_FEEDBACK=error_feedback
    )

    complex_indicators = ["完整", "全部", "前后端", "管理系统", "平台", "所有功能", "图书管理系统"]
    need_upgrade = any(kw in user_req for kw in complex_indicators)

    try:
        resp = model_demo_generate.invoke(prompt)
        code_text = resp.content.strip()

        if code_text.startswith("```python"):
            code_text = code_text.removeprefix("```python").removesuffix("```")
        elif code_text.startswith("```"):
            code_text = code_text.removeprefix("```").removesuffix("```")
        code_text = code_text.strip()

        files, project_name = parse_multifile_code(code_text)

        project_folder_name = project_name.replace(' ', '_')
        save_path = os.path.join(base_save_path, project_folder_name)
        os.makedirs(save_path, exist_ok=True)

        log_info("Demo版", "生成", f"项目名: {project_folder_name}, 解析到 {len(files)} 个文件: {list(files.keys())}")

        if DEMO_ENABLE_SANDBOX:
            log_info("Demo版", "沙箱测试", f"测试 {len(files)} 个文件...")
            notify_progress(state.get("conv_id"), f"🧪 [演示模式] 代码生成完成（{len(files)} 个文件），正在沙箱测试...")

            sandbox_results = {}
            all_pass = True

            for file_path, code in files.items():
                if not file_path.endswith('.py'):
                    sandbox_results[file_path] = {"success": True, "stderr": "非Python文件，跳过测试"}
                    continue

                try:
                    compile(code, file_path, 'exec')
                    success = True
                    stderr = ""
                    stdout = "语法检查通过"
                except SyntaxError as e:
                    success = False
                    stderr = f"语法错误: {str(e)} (行 {e.lineno})"
                    stdout = ""
                except Exception as e:
                    success = False
                    stderr = str(e)
                    stdout = ""

                sandbox_results[file_path] = {
                    "success": success,
                    "stderr": stderr,
                    "stdout": stdout,
                    "return_code": 0 if success else 1
                }

                if not success:
                    all_pass = False
                    log_error("Demo版", "沙箱测试", f"❌ {file_path} 失败")
                    log_error("Demo版", "沙箱测试", f"  错误详情: {stderr[:200]}")
                else:
                    log_success("Demo版", "沙箱测试", f"✅ {file_path} 通过")

            if all_pass:
                for file_path, code in files.items():
                    abs_file = os.path.join(save_path, file_path)
                    os.makedirs(os.path.dirname(abs_file), exist_ok=True)
                    with open(abs_file, "w", encoding="utf-8") as f:
                        f.write(code)
                log_success("Demo版", "生成", f"✅ 沙箱测试通过，已写入 {len(files)} 个文件到 {project_folder_name}/")

                return {
                    "current_mode": "demo",
                    "final_plan": f"✅ Demo模式：已生成 {len(files)} 个文件到 {project_folder_name}/，沙箱测试通过",
                    "generated_files": files,
                    "sandbox_results": sandbox_results,
                    "modify_summary": [{"file": fp, "status": "generated", "remark": "沙箱测试通过"} for fp in files.keys()],
                    "demo_error_feedback": None,
                    "demo_retry_count": 0,
                    "messages": [AIMessage(content=f"✅ Demo模式生成完成！已生成 {len(files)} 个文件到 {project_folder_name}/，所有文件沙箱测试通过。")]
                }

            if retry_count < max_retry:
                error_feedback = "\n".join([f"文件 {fp} 错误: {res['stderr']}" for fp, res in sandbox_results.items() if not res['success']])
                log_warn("Demo版", "生成", f"沙箱测试失败，第 {retry_count + 1}/{max_retry} 次重试")
                notify_progress(state.get("conv_id"), f"⚠️ 沙箱测试发现问题，正在进行第 {retry_count + 1}/{max_retry} 次修复...")

                return {
                    "current_mode": "demo",
                    "demo_error_feedback": error_feedback,
                    "demo_retry_count": retry_count + 1,
                    "sandbox_results": sandbox_results,
                    "messages": [AIMessage(content=f"⚠️ Demo模式：代码生成完成但沙箱测试失败，正在进行第 {retry_count + 1}/{max_retry} 次重试修复...")]
                }

            if DEMO_CONTINUE_ON_FAILURE:
                for file_path, code in files.items():
                    abs_file = os.path.join(save_path, file_path)
                    os.makedirs(os.path.dirname(abs_file), exist_ok=True)
                    with open(abs_file, "w", encoding="utf-8") as f:
                        f.write(code)

                log_warn("Demo版", "生成", f"达到最大重试次数，已写入 {len(files)} 个文件到 {project_folder_name}/（存在沙箱错误）")

                return {
                    "current_mode": "demo",
                    "final_plan": f"⚠️ Demo模式：已生成 {len(files)} 个文件到 {project_folder_name}/，但沙箱测试存在错误（已重试{max_retry}次）",
                    "generated_files": files,
                    "sandbox_results": sandbox_results,
                    "modify_summary": [
                        {"file": fp, "status": "generated_with_errors", "remark": res.get("stderr", "未知错误")}
                        for fp, res in sandbox_results.items() if not res.get("success", False)
                    ],
                    "demo_error_feedback": None,
                    "demo_retry_count": 0,
                    "messages": [AIMessage(content=f"⚠️ Demo模式生成完成！已生成 {len(files)} 个文件到 {project_folder_name}/，但沙箱测试存在错误（已重试{max_retry}次）。请检查生成的代码。")]
                }
            else:
                log_error("Demo版", "生成", f"沙箱测试失败，已重试 {max_retry} 次")
                return {
                    "current_mode": "demo",
                    "final_plan": f"❌ Demo模式生成失败：沙箱测试未通过（已重试{max_retry}次）",
                    "modify_summary": [
                        {"file": fp, "status": "sandbox_failed", "remark": res.get("stderr", "未知错误")}
                        for fp, res in sandbox_results.items() if not res.get("success", False)
                    ],
                    "demo_error_feedback": None,
                    "demo_retry_count": 0,
                    "messages": [AIMessage(content=f"❌ Demo模式生成失败：沙箱测试未通过（已重试{max_retry}次）。")]
                }

        else:
            for file_path, code in files.items():
                abs_file = os.path.join(save_path, file_path)
                os.makedirs(os.path.dirname(abs_file), exist_ok=True)
                with open(abs_file, "w", encoding="utf-8") as f:
                    f.write(code)

            log_success("Demo版", "生成", f"已写入 {len(files)} 个文件到 {project_folder_name}/（沙箱测试已禁用）")
            return {
                "current_mode": "demo",
                "final_plan": f"✅ Demo模式：已生成 {len(files)} 个文件到 {project_folder_name}/（沙箱测试已禁用）",
                "generated_files": files,
                "modify_summary": [{"file": fp, "status": "generated", "remark": "沙箱测试已禁用"} for fp in files.keys()],
                "demo_error_feedback": None,
                "demo_retry_count": 0,
                "messages": [AIMessage(content=f"✅ Demo模式生成完成！已生成 {len(files)} 个文件到 {project_folder_name}/。")]
            }

    except Exception as e:
        error_str = str(e)
        if "timed out" in error_str.lower() or "timeout" in error_str.lower():
            if need_upgrade:
                log_warn("Demo版", "生成", "生成超时，自动升级到完整工程模式")
                return {
                    "current_mode": "complete",
                    "final_plan": "⚠️ 需求复杂，已自动升级到完整工程模式重新生成",
                    "modify_summary": [{"file": "upgrade", "status": "auto_upgrade", "remark": "Demo超时升级到Complete"}],
                    "messages": [AIMessage(content="⚠️ 您的需求较为复杂，已自动升级到完整工程模式重新生成...")]
                }
            else:
                if retry_count < max_retry:
                    return {
                        "demo_error_feedback": "生成超时，重试中...",
                        "demo_retry_count": retry_count + 1,
                        "messages": [AIMessage(content=f"⚠️ 生成超时，正在进行第 {retry_count + 1}/{max_retry} 次重试...")]
                    }
                else:
                    log_error("Demo版", "生成", "超时且重试次数用尽")
                    return {
                        "current_mode": "demo",
                        "final_plan": "❌ Demo模式生成失败：请求超时",
                        "modify_summary": [{"file": "unknown", "status": "failed", "remark": "超时"}],
                        "demo_error_feedback": None,
                        "demo_retry_count": 0,
                        "messages": [AIMessage(content="❌ Demo模式生成失败：请求超时。")]
                    }
        else:
            log_error("Demo版", "生成", f"失败: {e}")
            return {
                "current_mode": "demo",
                "final_plan": f"❌ Demo模式生成失败: {str(e)}",
                "modify_summary": [{"file": "unknown", "status": "failed", "remark": str(e)}],
                "demo_error_feedback": None,
                "demo_retry_count": 0,
                "messages": [AIMessage(content=f"❌ Demo模式生成失败: {str(e)}")]
            }


def route_after_clarify(state: DemoProjectState) -> str:
    if state.get("is_info_enough", False):
        log_info("Demo版", "路由", "信息充足，进入生成")
        return "demo_generate_node"
    else:
        log_info("Demo版", "路由", "信息不足，继续澄清")
        return "demo_clarify_node"


def route_after_generate(state: DemoProjectState) -> str:
    retry_count = state.get("demo_retry_count", 0)
    if retry_count > 0:
        return "demo_generate_node"
    return END


################################################################################################################
# ====================== 构建图 ======================
def build_demo_graph():
    log_info("Demo版", "构建图", "开始构建子图...")
    builder = StateGraph(DemoProjectState)

    builder.add_node("demo_clarify_node", demo_clarify_node)
    builder.add_node("demo_generate_node", demo_generate_node)

    builder.add_edge(START, "demo_clarify_node")

    builder.add_conditional_edges(
        source="demo_clarify_node",
        path=route_after_clarify
    )

    builder.add_conditional_edges(
        source="demo_generate_node",
        path=route_after_generate
    )

    graph = builder.compile()
    log_success("Demo版", "构建图", "子图构建完成")
    return graph


if __name__ == "__main__":
    graph = build_demo_graph()

    init_state = {
        "messages": [],
        "user_requirements": "帮我生成一个项目",
        "user_clarify_answers": [],
        "final_clear_requirement": None,
        "need_user_input": False,
        "clarify_questions": None,
        "is_info_enough": False,
        "clarify_round": 0,
        "final_plan": None,
        "modify_summary": None,
        "current_mode": None,
        "demo_error_feedback": None,
        "demo_retry_count": 0,
        "generated_files": None,
        "sandbox_results": None,
        "confirmed_info": {}
    }

    result = graph.invoke(init_state)
    print(result.get("final_plan"))