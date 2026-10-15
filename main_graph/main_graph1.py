import json
import os
import re
import sqlite3
from typing import TypedDict, Annotated, Optional, List, Dict, Any
import operator
import uuid
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.types import Send, Command, interrupt

from pydantic import BaseModel, Field, ValidationError
from typing import Type, Callable, List

from my_vibecoding.project_generate_code.project_generate_code import build_create_graph
from my_vibecoding.simple_generate_code.simple_generate_code import build_simple_graph
from my_vibecoding.demo_generate_code.demo_generate_code import build_demo_graph
from my_vibecoding.modify_code.modify_code import build_change_graph

load_dotenv()

config_path = "../config_file/my_config.yaml"

absolute_path = os.path.abspath(__file__)
script_dir = os.path.dirname(absolute_path)
DB_FILE = os.path.abspath(os.path.join(script_dir, "../agent_data.db"))

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败: {e}") from e


################################################################################################################
# ====================== 日志系统 ======================
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


LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_SUCCESS = 2
LOG_LEVEL_WARN = 3
LOG_LEVEL_ERROR = 4
CURRENT_LOG_LEVEL = LOG_LEVEL_INFO
_indent_level = 0


def log_info(module: str, node: str, msg: str):
    if CURRENT_LOG_LEVEL <= LOG_LEVEL_INFO:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{LogColor.CYAN}[{timestamp}] [{module}] [{node}] {msg}{LogColor.END}")


def log_success(module: str, node: str, msg: str):
    if CURRENT_LOG_LEVEL <= LOG_LEVEL_SUCCESS:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{LogColor.GREEN}[{timestamp}] [{module}] [{node}] ✅ {msg}{LogColor.END}")


def log_warn(module: str, node: str, msg: str):
    if CURRENT_LOG_LEVEL <= LOG_LEVEL_WARN:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{LogColor.YELLOW}[{timestamp}] [{module}] [{node}] ⚠️ {msg}{LogColor.END}")


def log_error(module: str, node: str, msg: str):
    if CURRENT_LOG_LEVEL <= LOG_LEVEL_ERROR:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{LogColor.RED}[{timestamp}] [{module}] [{node}] ❌ {msg}{LogColor.END}")


def log_separator(module: str, title: str = ""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    if title:
        print(f"{LogColor.BOLD}{LogColor.MAGENTA}[{timestamp}] [===== {module} - {title} =====]{LogColor.END}")
    else:
        print(f"{LogColor.MAGENTA}[{timestamp}] [===== {module} =====]{LogColor.END}")


def log_subgraph_start(module: str, mode: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    mode_names = {"simple": "⚡ 极简", "demo": "🧪 Demo", "complete": "🏗️ 完整", "modify": "🔧 修改"}
    print(
        f"{LogColor.BOLD}{LogColor.MAGENTA}[{timestamp}] [===== {module} - {mode_names.get(mode, mode)} =====]{LogColor.END}")


def log_subgraph_end(module: str, status: str = "完成"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = "✅" if status == "完成" else "❌"
    print(
        f"{LogColor.BOLD}{LogColor.MAGENTA}[{timestamp}] [===== {module} - {status_icon} {status} =====]{LogColor.END}")


################################################################################################################
# ====================== 配置读取 ======================
main_runtime = config.get("main_runtime", {})

# 压缩配置
compression_config = main_runtime.get("compression", {})
COMPRESSION_ENABLED = compression_config.get("enabled", True)
COMPRESSION_MAX_MESSAGES = compression_config.get("max_messages", 12)
COMPRESSION_MAX_TOKENS = compression_config.get("max_tokens", 8000)
COMPRESSION_KEEP_RECENT = compression_config.get("keep_recent_messages", 4)
COMPRESSION_SUMMARY_PROMPT = compression_config.get("summary_prompt", """
压缩以下对话为一段摘要，不超过200字：
{conversation_text}
""")

demo_config = main_runtime.get("demo_mode", {})
DEMO_ENABLED = demo_config.get("enabled", True)
DEMO_MAX_RETRY = demo_config.get("max_retry", 2)
DEMO_MAX_FILES = demo_config.get("max_files", 10)
DEMO_ENABLE_SANDBOX = demo_config.get("enable_sandbox", True)

simple_config = main_runtime.get("simple_mode", {})
SIMPLE_ENABLED = simple_config.get("enabled", True)
SIMPLE_OUTPUT_FILENAME = simple_config.get("output_filename", "simple_project.py")

# ★★★ 只读取数值型配置，没有任何关键词列表 ★★★
mode_judge_config = config.get("mode_judge", {})
MODE_THRESHOLDS = mode_judge_config.get("thresholds", {"complete_min": 10, "demo_min": 5, "simple_min": 0})
MODE_FUZZY_ZONE = mode_judge_config.get("fuzzy_zone", {"low": 5, "high": 9})


def get_openai_model(model_provider: str, model_name: str):
    if model_provider == "ali":
        api_key = os.getenv("ALI_API_KEY")
    elif model_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None
    if not api_key:
        raise RuntimeError(f"模型 {model_provider} 的API密钥未配置！")
    return api_key, model_name


api_key_final_confirm_intention, model_name_final_confirm_intention = get_openai_model(
    config["final_confirm_intention_model"]["model_provider"],
    config["final_confirm_intention_model"]["model_name"])

api_key_chat_model, model_name_chat_model = get_openai_model(
    config["chat_model"]["model_provider"],
    config["chat_model"]["model_name"])

final_confirm_intention_model = ChatOpenAI(
    model=model_name_final_confirm_intention,
    api_key=api_key_final_confirm_intention,
    base_url=config["final_confirm_intention_model"]["base_url"],
    temperature=0.1,
    timeout=60,
    max_retries=2
)

chat_model = ChatOpenAI(
    model=model_name_chat_model,
    api_key=api_key_chat_model,
    base_url=config["chat_model"]["base_url"],
    temperature=0.7,
    timeout=60,
    max_retries=2
)

memory_extract_model = chat_model


def clean_json_data(json_str: str) -> str:
    content = json_str.strip()
    if content.startswith("```json"):
        content = content.removeprefix("```json").removesuffix("```")
    elif content.startswith("```"):
        content = content.removeprefix("```").removesuffix("```")
    content = content.strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if match:
        content = match.group(0)
    return content.strip()


################################################################################################################
# ====================== 数据库初始化 ======================
def init_business_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation (
        conv_id TEXT PRIMARY KEY,
        user_id TEXT,
        title TEXT,
        conv_type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        conv_id TEXT,
        memory_type TEXT,
        content TEXT,
        meta_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user(user_id)
    )
    """)

    cur.execute("PRAGMA table_info(conversation_memory)")
    columns = [col[1] for col in cur.fetchall()]
    if "conv_id" not in columns:
        cur.execute("ALTER TABLE conversation_memory ADD COLUMN conv_id TEXT")
        log_info("数据库", "迁移", "conversation_memory 表已添加 conv_id 列")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_memory_conv_id ON conversation_memory(conv_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_memory_user_id ON conversation_memory(user_id)")

    conn.commit()
    conn.close()


init_business_db()


################################################################################################################
# ====================== 消息压缩 ======================
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_words = len(re.findall(r'[a-zA-Z]+', text))
    return int(chinese_chars * 2 + english_words * 1.3 + len(text.split()) * 0.5)


def get_messages_token_count(messages: list) -> int:
    total = 0
    for msg in messages:
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        total += estimate_tokens(content)
    return total


def should_compress_messages(messages: list) -> bool:
    if not COMPRESSION_ENABLED:
        return False
    if len(messages) > COMPRESSION_MAX_MESSAGES:
        log_info("压缩", "检查", f"消息数量 {len(messages)} > {COMPRESSION_MAX_MESSAGES}")
        return True
    if get_messages_token_count(messages) > COMPRESSION_MAX_TOKENS:
        return True
    return False


def compress_messages(messages: list) -> list:
    if not should_compress_messages(messages):
        return messages

    keep_recent = COMPRESSION_KEEP_RECENT
    recent_messages = messages[-keep_recent:]
    old_messages = messages[:-keep_recent]

    conversation_text = ""
    for msg in old_messages:
        role = "用户" if isinstance(msg, HumanMessage) else ("助手" if isinstance(msg, AIMessage) else "系统")
        content = str(msg.content) if hasattr(msg, 'content') else str(msg)
        if isinstance(msg, AIMessage) and content.startswith("【历史对话摘要】"):
            conversation_text += f"[已摘要] {content}\n"
        else:
            conversation_text += f"{role}: {content}\n"

    compress_prompt = COMPRESSION_SUMMARY_PROMPT.format(conversation_text=conversation_text)

    try:
        resp = memory_extract_model.invoke(compress_prompt)
        summary = resp.content.strip()
        summary_msg = AIMessage(content=f"【历史对话摘要】{summary}")
        compressed = [summary_msg] + recent_messages
        log_info("压缩", "完成", f"{len(messages)} → {len(compressed)} 条")
        return compressed
    except Exception as e:
        log_error("压缩", "失败", str(e))
        return recent_messages


################################################################################################################
# ====================== 状态定义 ======================
class MainGraph(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_requirements: str | None

    is_chat: bool | None
    is_create_project: bool | None
    is_modify_project: bool | None

    current_mode: str | None
    subgraph_status: str | None

    user_raw_requirement: str | None
    user_change_requirement: str | None

    remaining_steps: int | None
    max_loop: int | None

    user_clarify_answers: Annotated[list[str], operator.add]
    final_clear_requirement: str | None
    need_user_input: bool
    clarify_questions: list[str] | None
    info_enough: bool
    front_plan_one: str | None
    front_plan_two: str | None
    final_plan: str | None

    user_id: Optional[str]
    conv_id: Optional[str]
    long_term_memory: Optional[str]

    demo_error_feedback: str | None
    demo_retry_count: int
    generated_files: dict[str, str] | None
    sandbox_results: dict | None

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

    inferred_can_unit_test: list[str] | None
    inferred_top_entry: str | None
    top_entry_confidence: str | None
    test_retry_count: int
    main_retry_count: int
    test_phase: str | None
    need_re_evaluate: bool | None
    candidate_top_entries: list[str] | None

    clarify_attempts: int
    is_project_clarified: bool
    clarify_round: int

    understood_requirement: str | None
    requirement_summary: str | None
    missing_info: list[str] | None
    is_requirement_complete: bool
    understanding_round: int
    force_mode: str | None

    business_context: Optional[Dict] | None


################################################################################################################
# ====================== 编译子图 ======================
log_info("主图", "初始化", "编译子图...")
create_project_subgraph = build_create_graph()
simple_subgraph = build_simple_graph()
demo_subgraph = build_demo_graph()
change_project_subgraph = build_change_graph()
log_success("主图", "初始化", "所有子图编译完成")


################################################################################################################
# ====================== 节点 ======================
def load_long_memory_node(state: MainGraph) -> dict:
    user_id = state.get("user_id")
    conv_id = state.get("conv_id") or state.get("thread_id") or "default"

    if not user_id:
        return {"long_term_memory": None}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT memory_type, content FROM conversation_memory
        WHERE user_id = ? AND (conv_id = ? OR conv_id IS NULL)
        ORDER BY created_at ASC
    """, (user_id, conv_id))
    rows = cur.fetchall()
    conn.close()

    if rows:
        memory_text = "\n".join([f"[{t}] {c}" for t, c in rows])
        log_success("主图", "加载记忆", f"加载了 {len(rows)} 条记忆 (对话: {conv_id})")
        return {"long_term_memory": memory_text}

    log_info("主图", "加载记忆", f"无记忆 (对话: {conv_id})")
    return {"long_term_memory": None}


def compress_messages_node(state: MainGraph) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {}
    compressed = compress_messages(messages)
    if len(compressed) != len(messages):
        return {"messages": compressed}
    return {}


def user_input_node(state: MainGraph) -> dict:
    messages = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    log_info("主图", "用户输入", f"{user_input[:50]}...")
    return {"user_requirements": user_input}


def analyze_user_intention_node(state: MainGraph) -> dict:
    """
    意图识别 - 完全由大模型判断
    ★★★ 明确区分"询问功能"和"要求生成" ★★★
    """
    log_info("主图", "意图识别", "分析用户意图...")
    user_requirements = state["user_requirements"]

    prompt = f"""
你是意图分类专家。请根据用户输入判断其真实意图。

【三种意图】
- is_chat: 用户正在闲聊、问候、**咨询问题、了解功能、询问能力**、表达情绪等，没有要求生成或修改代码
- is_create_project: 用户明确要求生成新代码、新建项目（如"帮我写一个爬虫"）
- is_modify_project: 用户明确要求修改现有代码（如"修复这个bug"）

【判断原则】
- **"你能做什么"、"有哪些功能"、"你有什么能力" → 属于 is_chat**（这是咨询，不是命令）
- **"帮我写"、"生成"、"创建一个" → 属于 is_create_project**（这是命令）
- **"修改"、"修复"、"改一下" → 属于 is_modify_project**（这是命令）
- 不要根据关键词硬匹配，而是理解整体语义。

【用户输入】
{user_requirements}

【输出格式】只输出JSON：
{{"is_chat": boolean, "is_create_project": boolean, "is_modify_project": boolean}}
"""

    try:
        response = final_confirm_intention_model.invoke(prompt)
        cleaned = clean_json_data(response.content)
        result = json.loads(cleaned)
    except Exception as e:
        log_error("主图", "意图识别", f"失败: {e}，降级为闲聊")
        return {"is_chat": True, "is_create_project": False, "is_modify_project": False}

    if not any([result.get("is_chat"), result.get("is_create_project"), result.get("is_modify_project")]):
        log_warn("主图", "意图识别", "三字段全false，降级为闲聊")
        return {"is_chat": True, "is_create_project": False, "is_modify_project": False}

    mode = "chat" if result["is_chat"] else ("pending_create" if result["is_create_project"] else "modify")
    log_success("主图", "意图识别",
                f"→ {'闲聊' if result['is_chat'] else ('新建' if result['is_create_project'] else '修改')}")

    return {
        "is_chat": result["is_chat"],
        "is_create_project": result["is_create_project"],
        "is_modify_project": result["is_modify_project"],
        "user_raw_requirement": user_requirements if result["is_create_project"] else None,
        "user_change_requirement": user_requirements if result["is_modify_project"] else None,
        "current_mode": mode
    }


def understand_and_judge_mode_node(state: MainGraph) -> dict:
    """
    模式判定与需求理解 - 完全由大模型决策
    """
    user_req = state.get("user_requirements", "")
    messages = state.get("messages", [])
    understanding_round = state.get("understanding_round", 0)

    log_info("主图", "需求理解", f"第 {understanding_round + 1} 轮分析")

    history_text = ""
    for msg in messages[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            history_text += f"助手: {msg.content}\n"

    complete_min = MODE_THRESHOLDS.get("complete_min", 10)
    demo_min = MODE_THRESHOLDS.get("demo_min", 5)

    prompt = f"""
你是需求分析和模式判定专家。根据用户输入，判断需求是否充足，并决定走哪个模式。

【五种模式】
- chat: 闲聊，不生成代码
- simple: 极简单文件（少于{demo_min}个文件）
- demo: 多文件演示（{demo_min}到{complete_min-1}个文件）
- complete: 完整工程（{complete_min}个文件以上）
- modify: 修改已有代码

【判断规则】
1. 如果用户在闲聊、问候、咨询问题 → mode = chat
2. 如果用户明确要求修改代码 → mode = modify
3. 如果用户要求生成代码：
   - 有明确的项目类型和功能描述 → is_complete = true，根据需求复杂度选择 simple/demo/complete
   - 只有项目意图但没有具体内容 → is_complete = false，追问细节
4. 用户说"开始吧"、"就这样"等确认词 → is_complete = true，默认走 demo

【用户输入（核心依据）】
{user_req}

【历史对话（仅供参考）】
{history_text}

【输出格式】
{{"mode": "chat|simple|demo|complete|modify", "is_complete": true/false, "understood": "需求总结", "questions": []}}
"""

    try:
        resp = final_confirm_intention_model.invoke(prompt)
        cleaned = clean_json_data(resp.content)
        result = json.loads(cleaned)

        mode = result.get("mode", "chat")
        is_complete = result.get("is_complete", False)
        understood = result.get("understood", user_req)
        questions = result.get("questions", [])

        if mode == "chat":
            log_success("主图", "需求理解", "→ 💬 闲聊")
            return {
                "is_chat": True,
                "is_create_project": False,
                "is_modify_project": False,
                "current_mode": "chat"
            }

        if mode == "modify":
            log_success("主图", "需求理解", "→ 🔧 修改")
            return {
                "is_chat": False,
                "is_create_project": False,
                "is_modify_project": True,
                "current_mode": "modify",
                "user_change_requirement": user_req,
                "is_requirement_complete": True
            }

        if is_complete:
            log_success("主图", "需求理解", f"→ 🏗️ {mode}")
            return {
                "is_chat": False,
                "is_create_project": True,
                "is_modify_project": False,
                "is_requirement_complete": True,
                "current_mode": mode,
                "understood_requirement": understood,
                "requirement_summary": understood[:200],
                "need_user_input": False,
                "understanding_round": understanding_round + 1,
                "missing_info": [],
                "force_mode": None,
                "business_context": {}
            }

        # 信息不足 → 追问
        log_info("主图", "需求理解", "信息不足，进入追问")
        if not questions:
            questions = ["请告诉我你想做什么类型的项目？"]
        return {
            "is_chat": False,
            "is_create_project": False,
            "is_modify_project": False,
            "is_requirement_complete": False,
            "need_user_input": True,
            "clarify_questions": questions,
            "understood_requirement": understood,
            "understanding_round": understanding_round + 1,
            "missing_info": [],
            "force_mode": None,
            "business_context": {}
        }

    except Exception as e:
        log_error("主图", "需求理解", f"失败: {e}")
        return {
            "is_chat": True,
            "is_create_project": False,
            "is_modify_project": False,
            "current_mode": "chat"
        }


def ask_clarify_questions_node(state: MainGraph) -> dict:
    """
    向用户追问缺失信息
    ★★★ 展示三条路线供选择 ★★★
    ★★★ 取消时切换到聊天模式，避免循环 ★★★
    """
    questions = state.get("clarify_questions", [])
    user_req = state.get("user_requirements", "")

    routes_text = """
🟢 极简模式：单文件脚本，50-200行代码
   适合：简单工具、数据处理脚本、快速原型

🟡 演示模式：多文件项目，有基本的功能模块划分
   适合：爬虫、API服务、小工具

🔴 完整模式：大型项目，含数据库、前后端、权限管理
   适合：管理系统、企业级应用
"""

    question_text = f"""
请告诉我你想做什么类型的项目？

📌 你可以选择以下三条路线：

{routes_text}

💡 也可以直接描述你想做什么，比如：
• 帮我写一个爬虫
• 生成一个API服务
• 做一个数据分析工具

（如果你觉得信息够了，直接说"开始吧"即可。）
"""

    log_info("主图", "追问", f"展示三条路线")
    user_answer = interrupt({"question": question_text})

    if not user_answer or len(str(user_answer).strip()) == 0:
        understanding_round = state.get("understanding_round", 0)
        return {"understanding_round": understanding_round + 1}

    user_answer_str = str(user_answer).strip()

    prompt = f"""
你是意图判断专家。用户回答了追问，请判断用户意图。

【用户原始需求】
{user_req}

【用户回答】
{user_answer_str}

【三条路线参考】
- 极简模式 / simple / 路线1
- 演示模式 / demo / 路线2
- 完整模式 / complete / 路线3

【判断规则】
1. 用户表达了"开始"、"确认"、"可以"、"好的" → intent = "start"
2. 用户表达了"取消"、"不做了"、"算了" → intent = "cancel"
3. 用户选择了某条路线 → intent = "route_selected"，route = 对应的路线名
4. 用户补充了具体的项目描述 → intent = "continue"，提取 project_type
5. 其他 → intent = "continue"

【输出格式】
{{"intent": "start|cancel|route_selected|continue", "route": "simple|demo|complete|null", "project_type": "项目类型或null", "summary": "简短说明"}}
"""

    try:
        resp = final_confirm_intention_model.invoke(prompt)
        cleaned = clean_json_data(resp.content)
        result = json.loads(cleaned)

        intent = result.get("intent", "continue")
        route = result.get("route")
        project_type = result.get("project_type")
        summary = result.get("summary", "")

        log_info("主图", "追问", f"大模型判断意图: {intent}, 路线: {route}, 项目类型: {project_type}")

        # ★★★ 取消 → 切换到聊天模式，彻底退出循环 ★★★
        if intent == "cancel":
            log_success("主图", "追问", "用户取消生成，切换到聊天模式")
            return {
                "is_chat": True,
                "is_create_project": False,
                "is_modify_project": False,
                "current_mode": "chat",
                "is_requirement_complete": False,
                "need_user_input": False,
                "clarify_questions": None,
                "messages": [AIMessage(content="好的，已取消生成。有什么需要随时找我。")]
            }

        if intent == "route_selected" and route:
            route_names = {"simple": "极简模式", "demo": "演示模式", "complete": "完整模式"}
            log_success("主图", "追问", f"用户选择了{route_names.get(route, route)}")
            return {
                "is_requirement_complete": True,
                "need_user_input": False,
                "clarify_questions": None,
                "current_mode": route,
                "understood_requirement": summary or f"用户选择了{route_names.get(route, route)}",
                "user_requirements": f"{user_req}\n[用户选择] {route_names.get(route, route)}，{user_answer_str}"
            }

        if intent == "start":
            log_success("主图", "追问", "用户确认开始，放行")
            return {
                "is_requirement_complete": True,
                "need_user_input": False,
                "clarify_questions": None,
                "current_mode": "demo",
                "understood_requirement": summary or "用户确认开始生成",
                "user_requirements": f"{user_req}\n[用户确认] {user_answer_str}"
            }

        if project_type:
            log_success("主图", "追问", f"用户补充了项目类型: {project_type}，放行")
            return {
                "is_requirement_complete": True,
                "need_user_input": False,
                "clarify_questions": None,
                "current_mode": "demo",
                "understood_requirement": summary or f"用户需要{project_type}项目",
                "user_requirements": f"{user_req}\n[用户补充] {user_answer_str}"
            }

        if len(user_answer_str) > 2:
            log_success("主图", "追问", "用户提供了内容，放行")
            return {
                "is_requirement_complete": True,
                "need_user_input": False,
                "clarify_questions": None,
                "current_mode": "demo",
                "understood_requirement": summary or user_answer_str,
                "user_requirements": f"{user_req}\n[用户补充] {user_answer_str}"
            }

        return {
            "is_requirement_complete": False,
            "need_user_input": True,
            "clarify_questions": ["请告诉我你想做什么类型的项目？"],
            "understanding_round": state.get("understanding_round", 0) + 1,
            "missing_info": ["项目类型"]
        }

    except Exception as e:
        log_error("主图", "追问", f"大模型判断失败: {e}")
        return {
            "is_requirement_complete": False,
            "need_user_input": True,
            "clarify_questions": ["请告诉我你想做什么类型的项目？"],
            "understanding_round": state.get("understanding_round", 0) + 1,
            "missing_info": ["项目类型"]
        }


def decide_project_mode_node(state: MainGraph) -> dict:
    """
    模式决策节点 - 只做透传，不做任何判断
    """
    mode = state.get("current_mode", "demo")
    force_mode = state.get("force_mode", None)

    if force_mode:
        log_success("主图", "模式决策", f"用户强制指定: {force_mode}")
        return {"current_mode": force_mode}

    log_success("主图", "模式决策", f"最终模式: {mode}")
    return {"current_mode": mode}


def chat_node(state: MainGraph) -> dict:
    """
    聊天节点 - 根据用户意图智能展示功能列表
    ★★★ 完全由大模型判断用户意图 ★★★
    """
    log_info("主图", "聊天", "生成回复...")
    user_requirements = state["user_requirements"]

    messages = state.get("messages", [])
    compressed = compress_messages(messages)

    chat_history = ""
    for msg in compressed[-6:]:
        if isinstance(msg, HumanMessage):
            chat_history += f"用户: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            chat_history += f"助手: {msg.content}\n"

    # ★★★ 全部模式描述 ★★★
    all_modes_text = """
💬 对话模式：日常聊天、咨询问题、技术讨论
   适合：闲聊、问问题、寻求帮助

⚡ 极简模式：单文件脚本，50-200行代码，无数据库无前端
   适合：简单工具、数据处理脚本、快速原型

🧪 演示模式：多文件项目，有基本的功能模块划分
   适合：爬虫、API服务、小工具

🏗️ 完整模式：大型项目，含数据库、前后端、权限管理
   适合：管理系统、企业级应用

🔧 修改模式：修改、修复、优化已有代码
   适合：修复bug、添加功能、代码重构
"""

    # ★★★ 三条项目路线 ★★★
    routes_text = """
🟢 极简模式：单文件脚本，50-200行代码
   适合：简单工具、数据处理脚本、快速原型

🟡 演示模式：多文件项目，有基本的功能模块划分
   适合：爬虫、API服务、小工具

🔴 完整模式：大型项目，含数据库、前后端、权限管理
   适合：管理系统、企业级应用
"""

    prompt = f"""
你是亲和友善的AI助手。根据用户输入，判断用户想了解什么，并给出对应的回复。

【用户当前输入】
{user_requirements}

【历史对话】
{chat_history}

【可展示的内容】

1. 全部功能（当用户想了解你的能力时展示）：
{all_modes_text}

2. 项目生成路线（当用户想创建项目时展示）：
{routes_text}

【判断规则 - 理解用户意图，不匹配特定词语】
1. 如果用户想了解"你能做什么"、"有哪些功能"、"你有什么能力" → 展示【全部功能】
2. 如果用户想"生成项目"、"创建项目"、"写一个项目" → 展示【项目生成路线】
3. 如果用户是普通聊天 → 正常回复
4. 如果用户提出了具体需求（如"帮我写一个爬虫"）→ 正常回复，不要展示列表

【回复要求】
- 简洁自然，不要长篇大论
- 如果是展示功能列表，用清晰的格式
- 如果是普通聊天，直接回答问题
"""

    try:
        response = chat_model.invoke(prompt)
        reply = response.content
    except Exception as e:
        log_error("主图", "聊天", f"失败: {e}")
        reply = "抱歉，我暂时无法回复。"

    log_success("主图", "聊天", f"回复: {reply[:50]}...")
    return {
        "current_mode": "chat",
        "messages": compressed + [AIMessage(content=reply)]
    }


def save_long_memory_node(state: MainGraph) -> dict:
    user_id = state.get("user_id")
    conv_id = state.get("conv_id") or state.get("thread_id") or "default"
    messages = state.get("messages", [])

    if not user_id or len(messages) == 0:
        return {}

    msg_text = "\n".join([f"{m.type}:{m.content}" for m in messages[-8:]])
    prompt = f"""
从对话中提取长期记忆，只提取三类：
1. preference: 技术偏好
2. history_project: 做过的项目
3. general: 其他重要信息

输出JSON数组：[{{"memory_type": "...", "content": "..."}}]

对话：{msg_text}
"""

    try:
        resp = memory_extract_model.invoke(prompt)
        cleaned = clean_json_data(resp.content)
        items = json.loads(cleaned)
    except Exception as e:
        log_error("主图", "保存记忆", f"失败: {e}")
        return {}

    if not items:
        log_info("主图", "保存记忆", "无有价值记忆")
        return {}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for item in items:
        if item.get("memory_type") and item.get("content"):
            cur.execute(
                "INSERT INTO conversation_memory(user_id, conv_id, memory_type, content, meta_json) VALUES (?, ?, ?, ?, ?)",
                (user_id, conv_id, item["memory_type"], item["content"], json.dumps({}))
            )
    conn.commit()
    conn.close()
    log_success("主图", "保存记忆", f"保存了 {len(items)} 条记忆 (对话: {conv_id})")
    return {}


def clear_conv_memory_node(state: MainGraph) -> dict:
    conv_id = state.get("conv_id") or state.get("thread_id") or "default"
    user_id = state.get("user_id")

    if not user_id:
        return {"messages": [AIMessage(content="未找到用户信息，无法清除记忆。")]}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM conversation_memory WHERE user_id = ? AND conv_id = ?", (user_id, conv_id))
    deleted = cur.rowcount
    conn.commit()
    conn.close()

    log_success("主图", "清除记忆", f"删除了 {deleted} 条记忆 (对话: {conv_id})")
    return {
        "long_term_memory": None,
        "messages": [AIMessage(content=f"已清除当前对话的 {deleted} 条记忆，你可以重新开始了。")]
    }


def fill_subgraph_config_node(state: MainGraph) -> dict:
    create_runtime = config.get("create_runtime", {})
    return {
        "remaining_steps": create_runtime.get("max_clarify_loop_steps", 3),
        "max_loop": config.get("change_max_loop", 5)
    }


def reset_create_sub_state_node(state: MainGraph) -> dict:
    log_info("主图", "重置状态", "重置新建项目子图")
    return {
        "current_mode": "complete",
        "user_clarify_answers": [],
        "final_clear_requirement": None,
        "need_user_input": False,
        "clarify_questions": None,
        "info_enough": False,
        "front_plan_one": None,
        "front_plan_two": None,
        "final_plan": None,
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
        "clarify_attempts": 0,
        "is_project_clarified": False,
        "clarify_round": 0,
        "is_requirement_complete": False,
        "understanding_round": 0,
        "understood_requirement": None,
        "requirement_summary": None,
        "missing_info": [],
        "force_mode": None,
        "business_context": None
    }


def after_chat_node(state: MainGraph) -> str:
    return "save_long_memory_node"


################################################################################################################
# ====================== 路由 ======================
def branch_node(state: MainGraph):
    if state.get("is_chat"):
        return "chat_node"
    elif state.get("is_create_project"):
        return "reset_create_sub_state_node"
    elif state.get("is_modify_project"):
        return "change_subgraph_node"
    else:
        log_warn("主图", "分支", "意图识别异常，降级为聊天")
        return "chat_node"


def route_after_understanding(state: MainGraph) -> str:
    """
    理解后的路由
    ★★★ 优先判断 current_mode，再判断 is_complete ★★★
    """
    current_mode = state.get("current_mode")
    is_complete = state.get("is_requirement_complete", False)
    understanding_round = state.get("understanding_round", 0)

    # ★★★ 如果已经是聊天模式，直接走聊天 ★★★
    if current_mode == "chat":
        log_info("主图", "路由", "当前为聊天模式，直接走聊天节点")
        return "chat_node"

    if is_complete:
        return "decide_project_mode_node"

    if understanding_round >= 3:
        log_warn("主图", "路由", "已达最大轮次，强制放行")
        return "decide_project_mode_node"

    return "ask_clarify_questions_node"


def route_after_mode_decision(state: MainGraph) -> str:
    mode = state.get("current_mode")
    if mode == "simple":
        return "simple_subgraph_node"
    elif mode == "demo":
        return "demo_subgraph_node"
    elif mode == "complete":
        return "create_subgraph_node"
    elif mode == "modify":
        return "change_subgraph_node"
    else:
        log_warn("主图", "路由", f"未知模式 {mode}，兜底走 demo")
        return "demo_subgraph_node"


def route_after_subgraph(state: MainGraph) -> str:
    return "save_long_memory_node"


################################################################################################################
# ====================== 构建图 ======================
builder = StateGraph(MainGraph)

builder.add_node("load_long_memory_node", load_long_memory_node)
builder.add_node("save_long_memory_node", save_long_memory_node)
builder.add_node("user_input_node", user_input_node)
builder.add_node("compress_messages_node", compress_messages_node)
builder.add_node("analyze_user_intention_node", analyze_user_intention_node)
builder.add_node("fill_subgraph_config_node", fill_subgraph_config_node)
builder.add_node("reset_create_sub_state_node", reset_create_sub_state_node)
builder.add_node("chat_node", chat_node)

builder.add_node("understand_and_judge_mode_node", understand_and_judge_mode_node)
builder.add_node("ask_clarify_questions_node", ask_clarify_questions_node)
builder.add_node("decide_project_mode_node", decide_project_mode_node)

builder.add_node("create_subgraph_node", create_project_subgraph)
builder.add_node("simple_subgraph_node", simple_subgraph)
builder.add_node("demo_subgraph_node", demo_subgraph)
builder.add_node("change_subgraph_node", change_project_subgraph)

builder.add_node("clear_conv_memory_node", clear_conv_memory_node)

builder.add_edge(START, "load_long_memory_node")
builder.add_edge("load_long_memory_node", "user_input_node")
builder.add_edge("user_input_node", "compress_messages_node")
builder.add_edge("compress_messages_node", "analyze_user_intention_node")
builder.add_edge("analyze_user_intention_node", "fill_subgraph_config_node")

builder.add_conditional_edges(
    source="fill_subgraph_config_node",
    path=branch_node
)

builder.add_edge("reset_create_sub_state_node", "understand_and_judge_mode_node")

builder.add_conditional_edges(
    source="understand_and_judge_mode_node",
    path=route_after_understanding,
    path_map={
        "decide_project_mode_node": "decide_project_mode_node",
        "ask_clarify_questions_node": "ask_clarify_questions_node",
        "chat_node": "chat_node",
    }
)

builder.add_edge("ask_clarify_questions_node", "understand_and_judge_mode_node")

builder.add_conditional_edges(
    source="decide_project_mode_node",
    path=route_after_mode_decision
)

builder.add_edge("create_subgraph_node", "save_long_memory_node")
builder.add_edge("simple_subgraph_node", "save_long_memory_node")
builder.add_edge("demo_subgraph_node", "save_long_memory_node")
builder.add_edge("change_subgraph_node", "save_long_memory_node")

builder.add_conditional_edges(
    source="chat_node",
    path=after_chat_node
)

builder.add_edge("save_long_memory_node", END)

sqlite_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)


class GraphInterruptException(Exception):
    def __init__(self, interrupt_value):
        self.interrupt_value = interrupt_value
        super().__init__(f"GraphInterrupt:{interrupt_value}")


def run_with_interrupt_support(initial_state: dict, config: dict):
    thread_config = config
    state = initial_state.copy()

    log_separator("主图", "开始执行")

    for event in graph.stream(state, config=thread_config):
        if "__interrupt__" in event:
            interrupt_data = event["__interrupt__"]
            if isinstance(interrupt_data, list) and len(interrupt_data) > 0:
                question = interrupt_data[0].value if hasattr(interrupt_data[0], 'value') else interrupt_data[0]
            else:
                question = interrupt_data
            log_warn("主图", "中断", f"等待用户输入: {question}")
            raise GraphInterruptException(question)
        else:
            for key, value in event.items():
                if key != "__interrupt__":
                    state[key] = value

    log_separator("主图", "执行完成")
    return state


if __name__ == "__main__":
    user_id = "user_001"
    run_config = {"configurable": {"thread_id": "main_thread_001", "user_id": user_id}}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO user(user_id,username) VALUES (?,?)", (user_id, "default_user"))
    conn.commit()
    conn.close()

    log_separator("主图", "启动 VibeCoding AI")

    while True:
        user_input = input("\n👤 请输入（输入 'exit' 退出）: ").strip()
        if user_input.lower() in ['exit', 'quit', '退出']:
            log_info("主图", "退出", "再见！")
            break
        if not user_input:
            continue

        init_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_requirements": user_input,
            "user_id": user_id,
            "conv_id": "main_thread_001",
            "is_chat": None,
            "is_create_project": None,
            "is_modify_project": None,
            "current_mode": None,
            "subgraph_status": None,
            "user_raw_requirement": None,
            "user_change_requirement": None,
            "remaining_steps": None,
            "max_loop": None,
            "user_clarify_answers": [],
            "final_clear_requirement": None,
            "need_user_input": False,
            "clarify_questions": None,
            "info_enough": False,
            "front_plan_one": None,
            "front_plan_two": None,
            "final_plan": None,
            "long_term_memory": None,
            "demo_error_feedback": None,
            "demo_retry_count": 0,
            "generated_files": None,
            "sandbox_results": None,
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
            "inferred_can_unit_test": None,
            "inferred_top_entry": None,
            "top_entry_confidence": None,
            "test_retry_count": 0,
            "main_retry_count": 0,
            "test_phase": None,
            "need_re_evaluate": False,
            "candidate_top_entries": None,
            "clarify_attempts": 0,
            "is_project_clarified": False,
            "clarify_round": 0,
            "is_requirement_complete": False,
            "understanding_round": 0,
            "understood_requirement": None,
            "requirement_summary": None,
            "missing_info": [],
            "force_mode": None,
            "business_context": None
        }

        try:
            result_state = run_with_interrupt_support(init_state, run_config)

            mode = result_state.get("current_mode", "unknown")
            mode_names = {
                "chat": "💬 对话", "simple": "⚡ 极简", "demo": "🧪 Demo",
                "complete": "🏗️ 完整", "modify": "🔧 修改"
            }

            print("\n" + "=" * 60)
            print("📋 执行摘要")
            print("=" * 60)
            print(f"\n📌 模式: {mode_names.get(mode, mode)}")

            modify_summary = result_state.get("modify_summary") or []
            if modify_summary:
                print("\n📝 修改摘要:")
                for item in modify_summary:
                    print(f"  - {item.get('file', '')} | {item.get('status', '')} | {item.get('remark', '')}")

            top_entry = result_state.get("top_entry_file")
            if top_entry:
                print(f"\n📌 顶层入口: {top_entry.get('file_path', '')}")

            final_plan = result_state.get("final_plan")
            if final_plan:
                print(f"\n📄 最终计划已生成 (长度: {len(final_plan)})")

            print("\n" + "=" * 60)

        except GraphInterruptException as e:
            user_resp = input(f"\n💬 {e.interrupt_value}\n👤 请输入：")
            resume_cmd = Command(resume=user_resp)
            result_state = graph.invoke(resume_cmd, config=run_config)

        except Exception as e:
            log_error("主图", "执行", f"异常: {e}")
            import traceback

            traceback.print_exc()