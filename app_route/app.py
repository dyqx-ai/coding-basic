import os
import sys
import sqlite3
import uuid
from typing import Optional, Dict, Any

# ========== 修复：确保无论从哪个工作目录启动 app.py，都能正确导入 my_vibecoding 包 ==========
# app.py 位于 my_vibecoding/app_route/app.py，父父目录就是包的上一级
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT_PARENT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", ".."))
if _PACKAGE_ROOT_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT_PARENT)
# 如果 my_vibecoding 目录本身是工作区根目录（目录名不等于 my_vibecoding），需要再补偿一级
_MY_PACKAGE_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if os.path.basename(_MY_PACKAGE_DIR) == "my_vibecoding" and _MY_PACKAGE_DIR not in sys.path:
    # 这种情况不需要把包目录加入，但要保证包父目录在
    if _PACKAGE_ROOT_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_ROOT_PARENT)

from flask import Flask, request, jsonify, render_template
from langgraph.types import Command

# ========== langgraph相关导入 ==========
from my_vibecoding.main_graph.main_graph import graph, DB_FILE as MAIN_DB_FILE

app = Flask(__name__)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DB_PATH = os.path.join(BASE_DIR, "agent_data.db")
BUSINESS_DB_PATH = os.path.join(BASE_DIR, "business.db")

# ★★★ 统一使用 main_graph.py 中的数据库路径 ★★★
MAIN_DB_PATH = MAIN_DB_FILE

os.makedirs(BASE_DIR, exist_ok=True)


# ===================== 业务数据库初始化 =====================
def init_business_db():
    conn = sqlite3.connect(BUSINESS_DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        title TEXT,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conv_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(conv_id) REFERENCES conversations(id)
    )
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS long_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        conv_id TEXT,
        memory_text TEXT NOT NULL,
        create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()


init_business_db()


# ===================== 业务数据库工具函数 =====================
def create_conversation(user_id: str, conv_title: str = "新对话") -> str:
    conv_id = str(uuid.uuid4())
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO conversations(id, user_id, title) VALUES (?, ?, ?)",
                (conv_id, user_id, conv_title))
    conn.commit()
    conn.close()
    return conv_id


def save_message(conv_id: str, role: str, content: str):
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO messages(conv_id, role, content) VALUES (?, ?, ?)",
                (conv_id, role, content))
    conn.commit()
    conn.close()


def get_conv_list(user_id: str):
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT * FROM conversations WHERE user_id=? ORDER BY create_time DESC",
                       (user_id,)).fetchall()
    res = [dict(r) for r in rows]
    conn.close()
    return res


def get_msg_history(conv_id: str):
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    rows = cur.execute("SELECT role,content,create_time FROM messages WHERE conv_id=? ORDER BY create_time ASC",
                       (conv_id,)).fetchall()
    res = [dict(r) for r in rows]
    conn.close()
    return res


# ===================== 辅助函数 =====================
def get_interrupt_info(config: dict) -> dict:
    state = graph.get_state(config)
    result = {
        "is_paused": False,
        "clarify_questions": None,
        "interrupt_payload": None,
        "need_user_input": False
    }

    if state.tasks and len(state.tasks) > 0:
        task = state.tasks[0]
        if hasattr(task, 'interrupts') and task.interrupts and len(task.interrupts) > 0:
            result["is_paused"] = True
            interrupt_value = task.interrupts[0].value
            result["interrupt_payload"] = interrupt_value

            if isinstance(interrupt_value, dict) and "question" in interrupt_value:
                result["clarify_questions"] = [interrupt_value["question"]]
                result["need_user_input"] = True
            elif isinstance(interrupt_value, str):
                result["clarify_questions"] = [interrupt_value]
                result["need_user_input"] = True

    if not result["clarify_questions"]:
        clarify = state.values.get("clarify_questions")
        if clarify:
            result["clarify_questions"] = clarify if isinstance(clarify, list) else [clarify]
            result["need_user_input"] = True

    return result


# ===================== Flask API接口 =====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/conv/new", methods=["POST"])
def api_new_conv():
    data = request.get_json()
    user_id = data.get("user_id", "default_user")
    title = data.get("title", "新对话")
    cid = create_conversation(user_id, title)
    return jsonify({"code": 0, "conv_id": cid})


@app.route("/api/conv/list", methods=["GET"])
def api_conv_list():
    user_id = request.args.get("user_id", "default_user")
    lst = get_conv_list(user_id)
    return jsonify({"code": 0, "data": lst})


@app.route("/api/conv/history", methods=["GET"])
def api_history():
    conv_id = request.args.get("conv_id")
    if not conv_id:
        return jsonify({"code": -1, "msg": "缺少conv_id"}), 400
    history = get_msg_history(conv_id)
    return jsonify({"code": 0, "data": history})


@app.route("/api/conv/rename", methods=["POST"])
def api_rename_conv():
    """修改对话名称"""
    data = request.get_json()
    conv_id = data.get("conv_id")
    new_title = data.get("title", "").strip()

    if not conv_id:
        return jsonify({"code": -1, "msg": "缺少conv_id"}), 400
    if not new_title:
        return jsonify({"code": -1, "msg": "标题不能为空"}), 400

    conn = sqlite3.connect(BUSINESS_DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conv_id))
    if cur.rowcount == 0:
        conn.close()
        return jsonify({"code": -1, "msg": "对话不存在"}), 404
    conn.commit()
    conn.close()

    print(f"[API] 重命名: 对话 {conv_id} → {new_title}")
    return jsonify({"code": 0, "msg": "重命名成功", "new_title": new_title})


def _do_invoke_and_respond(graph_input, config: dict, conv_id: str):
    """
    统一执行图并响应（api_chat 与 api_agent_resume 共用）：
    - 图挂起（interrupt）→ 追问消息入库 + 返回 agent_wait_input（否则前端历史断档，看起来像"没反应"）
    - 图跑完 → AI 回复入库 + 返回 code=0
    """
    result = graph.invoke(graph_input, config=config)

    interrupt_info = get_interrupt_info(config)
    if interrupt_info.get("is_paused"):
        # ★★★ 修复：追问消息也必须保存到对话历史，用户刷新/回看才能看到机器人问过什么 ★★★
        qs = interrupt_info.get("clarify_questions")
        if qs:
            q_text = "\n".join(qs if isinstance(qs, list) else [str(qs)])
            save_message(conv_id, "assistant", q_text)
        return jsonify({
            "code": 1,
            "msg": "agent_wait_input",
            "conv_id": conv_id,
            "mode": result.get("current_mode", "unknown"),
            "data": {
                "clarify_questions": qs,
                "interrupt_payload": interrupt_info.get("interrupt_payload")
            }
        })

    messages = result.get("messages", [])
    ai_msg = ""
    if messages and len(messages) > 0:
        last_msg = messages[-1]
        ai_msg = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)

    current_mode = result.get("current_mode", "unknown")

    if ai_msg:
        save_message(conv_id, "assistant", ai_msg)

    # 防御：子图完成后 state 里若还残留追问字段，同样按等待输入返回（保持旧版语义）
    clarify_qs = result.get("clarify_questions")
    if clarify_qs:
        return jsonify({
            "code": 1,
            "msg": "agent_wait_input",
            "conv_id": conv_id,
            "mode": current_mode,
            "data": {
                "clarify_questions": clarify_qs if isinstance(clarify_qs, list) else [clarify_qs],
                "interrupt_payload": None
            }
        })

    return jsonify({
        "code": 0,
        "conv_id": conv_id,
        "reply": ai_msg,
        "mode": current_mode
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    req_data = request.get_json()
    user_id = req_data.get("user_id", "default_user")
    conv_id = req_data.get("conv_id")
    user_input = req_data.get("content", "").strip()

    if not user_input:
        return jsonify({"code": -1, "msg": "消息不能为空"}), 400

    if not conv_id:
        conv_id = create_conversation(user_id, user_input[:16])

    save_message(conv_id, "user", user_input)

    config = {"configurable": {"thread_id": conv_id, "user_id": user_id}}

    # ★★★ 关键修复：若该对话还挂着未回答的追问（interrupt 挂起），新消息必须作为回答走 Command(resume=...)；
    # 否则普通 invoke 只更新状态、没有任何节点处理这条消息（LangGraph 实锤行为），
    # 表现就是用户连发几条消息都"没反应"，只能放弃。★★★
    if get_interrupt_info(config).get("is_paused"):
        try:
            return _do_invoke_and_respond(Command(resume=user_input), config, conv_id)
        except Exception as e:
            print(f"[api_chat resume 异常] {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                "code": -1,
                "msg": f"恢复执行异常: {str(e)}",
                "conv_id": conv_id
            }), 500

    try:
        return _do_invoke_and_respond(
            {"messages": [("user", user_input)], "user_id": user_id, "conv_id": conv_id},
            config,
            conv_id
        )

    except Exception as e:
        print(f"[api_chat 异常] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "code": -1,
            "msg": f"执行异常: {str(e)}",
            "conv_id": conv_id
        }), 500


@app.route("/api/agent/state", methods=["GET"])
def api_get_agent_state():
    conv_id = request.args.get("conv_id")
    if not conv_id:
        return jsonify({"code": -1, "msg": "缺少conv_id"}), 400

    config = {"configurable": {"thread_id": conv_id}}
    interrupt_info = get_interrupt_info(config)

    state = graph.get_state(config)
    current_mode = state.values.get("current_mode", "unknown")

    return jsonify({
        "code": 0,
        "data": {
            "is_paused": interrupt_info.get("is_paused", False),
            "clarify_questions": interrupt_info.get("clarify_questions"),
            "interrupt_payload": interrupt_info.get("interrupt_payload"),
            "need_user_input": interrupt_info.get("need_user_input", False),
            "mode": current_mode
        }
    })


@app.route("/api/agent/resume", methods=["POST"])
def api_agent_resume():
    req_data = request.get_json()
    conv_id = req_data.get("conv_id")
    user_answer = req_data.get("content", "").strip()
    user_id = req_data.get("user_id", "default_user")

    if not conv_id or not user_answer:
        return jsonify({"code": -1, "msg": "参数缺失conv_id/content"}), 400

    save_message(conv_id, "user", user_answer)

    config = {"configurable": {"thread_id": conv_id, "user_id": user_id}}

    try:
        return _do_invoke_and_respond(Command(resume=user_answer), config, conv_id)

    except Exception as e:
        print(f"[api_agent_resume 异常] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "code": -1,
            "msg": f"恢复执行异常: {str(e)}",
            "conv_id": conv_id
        }), 500


@app.route("/api/conv/clear", methods=["POST"])
def api_clear_conv():
    """删除对话：只删除该对话及其关联记忆，不影响其他对话"""
    data = request.get_json()
    conv_id = data.get("conv_id")
    user_id = data.get("user_id", "default_user")

    if not conv_id:
        return jsonify({"code": -1, "msg": "缺少conv_id"}), 400

    # 1. 删除该对话的消息
    conn = sqlite3.connect(BUSINESS_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE conv_id=?", (conv_id,))
    cur.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()

    # 2. ★★★ 只删除该对话的记忆，不影响其他对话 ★★★
    try:
        main_conn = sqlite3.connect(MAIN_DB_PATH)
        main_cur = main_conn.cursor()
        # 先检查表是否存在
        main_cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_memory'")
        if main_cur.fetchone():
            main_cur.execute("DELETE FROM conversation_memory WHERE conv_id = ?", (conv_id,))
            deleted = main_cur.rowcount
            main_conn.commit()
            main_conn.close()
            print(f"[API] 清除记忆: 对话 {conv_id} 的 {deleted} 条记忆已删除")
        else:
            main_conn.close()
            print(f"[API] 清除记忆: conversation_memory 表不存在，跳过")
    except Exception as e:
        print(f"[API] 清除记忆失败: {e}")

    return jsonify({"code": 0, "msg": "会话已删除（含关联记忆）"})


if __name__ == '__main__':
    print("=" * 60)
    print("🚀 VibeCoding AI Web 服务启动")
    print(f"   - 业务数据库: {BUSINESS_DB_PATH}")
    print(f"   - Checkpoint数据库: {CHECKPOINT_DB_PATH}")
    print(f"   - 主图数据库: {MAIN_DB_PATH}")
    print(f"   - 访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)