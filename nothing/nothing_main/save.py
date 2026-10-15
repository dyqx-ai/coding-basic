import json
import os
import re
import sqlite3
from typing import TypedDict, Annotated, Optional
import operator
import uuid
from datetime import datetime

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.types import Send, Command, interrupt

# 子图构建函数
from my_vibecoding.project_generate_code.final_gernerate_code import build_create_graph
from my_vibecoding.modify_code.real_check_and_change_code import build_change_graph


load_dotenv()

config_path = "../../config_file/my_config.yaml"

absolute_path = os.path.abspath(__file__)
script_dir = os.path.dirname(absolute_path)
save_path = os.path.abspath(os.path.join(script_dir, "../new_project"))
# sqlite数据库文件路径
DB_FILE = os.path.abspath(os.path.join(script_dir, "../../agent_data.db"))

# ========== 配置读取异常捕获 ==========
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败，请检查路径: {config_path}, 错误: {e}") from e


def get_openai_model(model_provider: str, model_name: str):
    if model_provider == "ali":
        api_key = os.getenv("ALI_API_KEY")
    elif model_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None
    if not api_key:
        raise RuntimeError(f"模型 {model_provider} 的API密钥环境变量未配置！")
    return api_key, model_name


# 意图识别模型
api_key_final_confirm_intention, model_name_final_confirm_intention = get_openai_model(
    config["final_confirm_intention_model"]["model_provider"],
    config["final_confirm_intention_model"]["model_name"])

# 闲聊对话模型
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

# 记忆摘要抽取模型（复用闲聊模型）
memory_extract_model = chat_model

################################################################################################################
# 业务数据库初始化
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
        memory_type TEXT,
        content TEXT,
        meta_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES user(user_id)
    )
    """)
    conn.commit()
    conn.close()

init_business_db()

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
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    return content.strip()

################################################################################################################
# 主图状态定义
class MainGraph(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_requirements: str | None

    is_chat: bool | None
    is_create_project: bool | None
    is_modify_project: bool | None

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
    long_term_memory: Optional[str]

################################################################################################################
#编译子图实例
create_project_subgraph = build_create_graph()
change_project_subgraph = build_change_graph()

################################################################################################################
#节点：加载长期记忆
def load_long_memory_node(state: MainGraph) -> dict:
    user_id = state.get("user_id")
    if not user_id:
        return {"long_term_memory": None}

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT memory_type, content FROM conversation_memory
        WHERE user_id = ? ORDER BY created_at ASC
    """, (user_id,))
    rows = cur.fetchall()
    conn.close()

    memory_segments = []
    for mem_type, content in rows:
        memory_segments.append(f"[{mem_type}] {content}")
    memory_text = "\n".join(memory_segments) if memory_segments else None
    return {"long_term_memory": memory_text}

################################################################################################################
#节点：保存长期记忆摘要
save_memory_prompt = """
你是记忆摘要提取器，从本轮完整对话中提取有价值的长期记忆。
只提取下面三类信息，无价值内容直接返回空列表[]：
1.preference：用户技术偏好、讨厌的框架、技术栈习惯、个人喜好
2.history_project：用户做过的项目简介、项目目标
3.general：其他重要约定、用户明确强调的信息

输出格式严格JSON数组：
[
  {"memory_type":"preference","content":"摘要内容"},
  {"memory_type":"history_project","content":"摘要内容"}
]
禁止输出思考、解释、markdown，只输出纯净JSON。

本轮对话历史：
"""

def save_long_memory_node(state: MainGraph) -> dict:
    user_id = state.get("user_id")
    messages = state.get("messages", [])
    if not user_id or len(messages) == 0:
        return {}

    msg_text = "\n".join([f"{m.type}:{m.content}" for m in messages[-8:]])
    # 字符串拼接，彻底放弃format，杜绝大括号解析
    prompt = save_memory_prompt + msg_text
    try:
        resp = memory_extract_model.invoke(prompt)
        clean_txt = clean_json_data(resp.content)
        memory_items = json.loads(clean_txt)
    except Exception as e:
        print(f"[记忆抽取失败] {e}")
        return {}

    if not isinstance(memory_items, list):
        return {}
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    for item in memory_items:
        m_type = item.get("memory_type")
        m_content = item.get("content")
        if m_type and m_content:
            cur.execute("""
                INSERT INTO conversation_memory(user_id, memory_type, content, meta_json)
                VALUES (?, ?, ?, ?)
            """, (user_id, m_type, m_content, json.dumps({})))
    conn.commit()
    conn.close()
    return {}

################################################################################################################
#节点1：用户输入
def user_input_node(state: MainGraph) -> dict:
    messages = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_input = msg.content
            break
    return {
        "user_requirements": user_input
    }

################################################################################################################
#节点2：分析用户意图提示词【无占位符纯模板，杜绝format解析】
analyze_intention_prompt = """
你是严格的用户意图分类器，读取用户输入，完成意图判定。
三个布尔字段互相互斥，**同一时刻只能有一个字段为true，剩下两个必须是false，绝不允许多true、绝不允许全false**。

字段释义：
is_chat：true代表用户是闲聊，无工程代码相关需求，只是问候、吐槽、聊天、无关提问，不执行工程任务。
is_create_project：true代表用户需要新建工程，从零开发项目、搭建全新代码项目，没有存量现有工程修改需求。
is_modify_project：true代表用户需要修改已有工程，对本地现存代码做修改、修复bug、新增功能、重构、调整逻辑等存量迭代。

参考样例：
样例1
用户输入：帮我写一个简易的todo后端服务
输出：{"is_chat":false,"is_create_project":true,"is_modify_project":false}

样例2
用户输入：今天天气好烦，好累
输出：{"is_chat":true,"is_create_project":false,"is_modify_project":false}

样例3
用户输入：我本地这个项目有bug，帮我修复一下
输出：{"is_chat":false,"is_create_project":false,"is_modify_project":true}

硬性强制规则：
1.三者只能一个true，另外两个全部false。
2.只输出纯净JSON字符串，禁止任何解释、思考、markdown、```代码块、多余文字。

输出JSON Schema：
{
  "is_chat": boolean,
  "is_create_project": boolean,
  "is_modify_project": boolean
}

用户输入：
"""
################################################################################################################
#聊天提示词【无占位符纯模板】
chat_prompt = """
你是陪伴对话助手，不要透露你是AI、大模型、程序、代码助手这类身份。
你的主要能力：可以帮用户从零生成完整代码工程，也可以读取本地现有工程做代码修改迭代。
用户闲聊时自然通俗回复，语气亲和，简短自然，不要长篇大论。
如果用户询问你能干什么，如实告知上面的两项核心能力。

【用户长期记忆】
"""
chat_prompt_part2 = """
历史上下文：
"""
chat_prompt_part3 = """
用户当前提问：
"""
################################################################################################################
#节点2：分析用户意图
def analyze_user_intention_node(state: MainGraph) -> dict:
    user_requirements = state["user_requirements"]
    # 字符串拼接，完全不使用format，彻底根除大括号KeyError
    final_analyze_intention_prompt = analyze_intention_prompt + user_requirements
    try:
        response = final_confirm_intention_model.invoke(final_analyze_intention_prompt)
        response_clean = clean_json_data(response.content)
        response_json = json.loads(response_clean)
    except Exception as e:
        print(f"[意图识别失败] {repr(e)}，降级为闲聊模式")
        return {"is_chat": True, "is_create_project": False, "is_modify_project": False}

    raw_req = None
    change_req = None
    if response_json["is_create_project"]:
        raw_req = user_requirements
    if response_json["is_modify_project"]:
        change_req = user_requirements

    return {
        "is_chat": response_json["is_chat"],
        "is_create_project": response_json["is_create_project"],
        "is_modify_project": response_json["is_modify_project"],
        "user_raw_requirement": raw_req,
        "user_change_requirement": change_req
    }

################################################################################################################
#填充子图配置参数
def fill_subgraph_config_node(state: MainGraph) -> dict:
    return {
        "remaining_steps": config.get("max_clarify_loop_steps", 3),
        "max_loop": config.get("max_loop", 3)
    }

#重置新建项目子图状态
def reset_create_sub_state_node(state: MainGraph) -> dict:
    return {
        "user_clarify_answers": [],
        "final_clear_requirement": None,
        "need_user_input": False,
        "clarify_questions": None,
        "info_enough": False,
        "front_plan_one": None,
        "front_plan_two": None,
        "final_plan": None
    }

################################################################################################################
#节点3：聊天节点
def chat_node(state: MainGraph) -> dict:
    user_requirements = state["user_requirements"]
    long_memory = state.get("long_term_memory") or "无"
    chat_history_text = ""
    if len(state["messages"]) >= 2:
        history_msgs = state["messages"][:-1]
        chat_history_text = "\n".join([f"{m.type}:{m.content}" for m in history_msgs])

    # 分段拼接，完全不使用format
    final_chat_prompt = (
        chat_prompt
        + long_memory
        + chat_prompt_part2
        + chat_history_text
        + chat_prompt_part3
        + user_requirements
    )
    try:
        response = chat_model.invoke(final_chat_prompt)
    except Exception as e:
        print(f"[闲聊模型调用异常]{repr(e)}")
        reply_text = "抱歉，我暂时无法回复。"
    else:
        reply_text = response.content

    print(f"\n助手:{reply_text}\n")
    return {"messages": [AIMessage(content=reply_text)]}

################################################################################################################
#分支节点1：派发子图任务，已移除Send，全部走标准子图嵌入，依靠state字段透传
def branch_node(state: MainGraph):
    if state["is_chat"]:
        return "chat_node"
    elif state["is_create_project"]:
        return "reset_create_sub_state_node"
    elif state["is_modify_project"]:
        return "change_subgraph_node"
    else:
        raise ValueError("用户意图分类错误")

################################################################################################################
def after_chat_node(state: MainGraph) -> str:
    return "save_long_memory_node"

################################################################################################################

builder = StateGraph(MainGraph)

builder.add_node("load_long_memory_node", load_long_memory_node)
builder.add_node("save_long_memory_node", save_long_memory_node)
builder.add_node("user_input_node", user_input_node)
builder.add_node("analyze_user_intention_node", analyze_user_intention_node)
builder.add_node("fill_subgraph_config_node", fill_subgraph_config_node)
builder.add_node("reset_create_sub_state_node", reset_create_sub_state_node)
builder.add_node("chat_node", chat_node)

builder.add_node("create_subgraph_node", create_project_subgraph)
builder.add_node("change_subgraph_node", change_project_subgraph)

# 流转
builder.add_edge(START, "load_long_memory_node")
builder.add_edge("load_long_memory_node", "user_input_node")
builder.add_edge("user_input_node", "analyze_user_intention_node")
builder.add_edge("analyze_user_intention_node", "fill_subgraph_config_node")

builder.add_conditional_edges(
    source="fill_subgraph_config_node",
    path=branch_node
)

# 新建项目路径
builder.add_edge("reset_create_sub_state_node", "create_subgraph_node")
builder.add_edge("create_subgraph_node", "save_long_memory_node")

# 修改项目路径（标准子图嵌入，不再Send）
builder.add_edge("change_subgraph_node", "save_long_memory_node")

# 闲聊路径
builder.add_conditional_edges(
    source="chat_node",
    path=after_chat_node
)

builder.add_edge("save_long_memory_node", END)

# ========== 编译：绑定SqliteSaver持久化checkpointer，interrupt/resume必须依赖 ==========
sqlite_conn = sqlite3.connect(DB_FILE, check_same_thread=False)
checkpointer = SqliteSaver(sqlite_conn)
checkpointer.setup()
graph = builder.compile(checkpointer=checkpointer)

# ---------------- 命令行测试入口 ----------------
if __name__ == "__main__":
    run_config = {
        "configurable": {
            "thread_id": "test_001",
            "user_id": "user_001"
        }
    }
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO user(user_id,username) VALUES (?,?)", ("user_001", "default_user"))
    conn.commit()
    conn.close()

    print("=== 测试闲聊 ===")
    result = graph.invoke(
        {"messages": [("user", "你好，你是谁？")], "user_id": "user_001"},
        config=run_config
    )
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
