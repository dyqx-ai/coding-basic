# -*- coding: utf-8 -*-
"""
进度通知器：子图长任务（代码生成/沙箱测试）执行期间，实时把进度消息写入业务库。

背景：web 端 /api/chat 与 /api/agent/resume 是同步执行模型，图跑完才返回响应。
Demo/Complete 模式生成代码 + 沙箱测试动辄 1~5 分钟，用户在此期间看不到任何反馈，
只能刷新页面从历史里找结果。此模块让中间进度实时入库（business.db messages 表），
配合前端轮询 /api/conv/history 即可增量展示"正在生成 → 正在测试 → 完成"全过程。

设计约束：
- 通知失败绝不影响主流程（静默吞异常）
- 同一 (conv_id, text) 在 min_interval 秒内防重（节点重试/重复进入时不会刷屏）
"""
import os
import sqlite3
import threading
import time

_BUSINESS_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app_route", "business.db"
)

_LAST_NOTIFY: dict = {}
_LOCK = threading.Lock()


def notify_progress(conv_id, text: str, min_interval: float = 8.0):
    """向业务库写入一条 assistant 进度消息。conv_id 为空或写入失败时静默跳过。"""
    if not conv_id or not text:
        return
    key = (str(conv_id), text)
    now = time.time()
    with _LOCK:
        if _LAST_NOTIFY.get(key, 0.0) + min_interval > now:
            return
        _LAST_NOTIFY[key] = now
    try:
        conn = sqlite3.connect(_BUSINESS_DB_PATH, timeout=5)
        try:
            conn.execute(
                "INSERT INTO messages(conv_id, role, content) VALUES (?, 'assistant', ?)",
                (str(conv_id), text),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # 进度通知属于锦上添花，任何失败都不能打断生成主流程
        pass
