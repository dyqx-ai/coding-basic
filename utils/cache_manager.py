"""
代码摘要缓存管理器
职责：为生成和修改模式提供代码摘要的哈希缓存，避免重复 LLM 调用
"""
import sqlite3
import hashlib
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

# 缓存数据库路径（放在 my_vibecoding 根目录下，与 utils 平级）
CACHE_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "code_cache.db")


class CodeSummaryCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化缓存表（如果不存在）"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_cache (
                cache_key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 增加索引加速查询
        cur.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON code_cache(file_path)")
        conn.commit()
        conn.close()
        print(f"[Cache] 缓存数据库已就绪: {self.db_path}")

    def _make_key(self, file_path: str, code: str) -> str:
        """
        生成缓存键：基于 文件路径 + 源码 的 MD5
        这样不同路径下相同内容的文件不会互相污染
        """
        raw = f"{file_path}||{code}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    def get(self, file_path: str, code: str) -> Optional[Dict[str, Any]]:
        """
        查询缓存
        返回：摘要字典（如 {'rel_path': '...', 'file_desc': '...', ...}）
        未命中返回 None
        """
        key = self._make_key(file_path, code)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT summary_json FROM code_cache WHERE cache_key = ?", (key,))
        row = cur.fetchone()
        conn.close()

        if row:
            try:
                summary = json.loads(row[0])
                print(f"[Cache] ✅ 命中: {file_path}")
                return summary
            except json.JSONDecodeError:
                print(f"[Cache] ⚠️ 缓存数据损坏，忽略: {file_path}")
                return None
        return None

    def set(self, file_path: str, code: str, summary_data: Dict[str, Any]) -> None:
        """
        写入缓存（如果已存在则覆盖）
        summary_data: 任何可 JSON 序列化的字典
        """
        key = self._make_key(file_path, code)
        summary_json = json.dumps(summary_data, ensure_ascii=False)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO code_cache (cache_key, file_path, summary_json) VALUES (?, ?, ?)",
            (key, file_path, summary_json)
        )
        conn.commit()
        conn.close()
        print(f"[Cache] 💾 写入: {file_path}")

    def exists(self, file_path: str, code: str) -> bool:
        """快速判断是否存在缓存（比 get 更轻量）"""
        key = self._make_key(file_path, code)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM code_cache WHERE cache_key = ?", (key,))
        found = cur.fetchone() is not None
        conn.close()
        return found

    def clear(self):
        """清空所有缓存（调试用）"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM code_cache")
        conn.commit()
        conn.close()
        print("[Cache] 🗑️ 所有缓存已清空")


# 全局单例（方便导入使用）
_default_cache = None

def get_cache() -> CodeSummaryCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = CodeSummaryCache()
    return _default_cache


# 便捷函数（供外部直接调用）
def get_cached_summary(file_path: str, code: str) -> Optional[Dict[str, Any]]:
    return get_cache().get(file_path, code)

def set_cached_summary(file_path: str, code: str, summary_data: Dict[str, Any]) -> None:
    get_cache().set(file_path, code, summary_data)