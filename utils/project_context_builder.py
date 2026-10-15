"""
项目上下文构建器
职责：为生成阶段构建"三层理解"（全貌 + 角色 + 上下文）
"""
import json
from typing import List, Dict, Any, Optional


class ProjectContextBuilder:
    """构建项目上下文，供 single_file_generate_prompt 使用"""

    @staticmethod
    def build_context(schema_data: dict) -> dict:
        """
        从 Schema 构建完整的项目上下文
        """
        business_context = schema_data.get("business_context", {})
        generation_order = schema_data.get("generation_order", [])
        directory_tree = schema_data.get("directory_tree", [])

        all_files = ProjectContextBuilder._collect_all_files(directory_tree)

        # 按层分组
        layers = {"models": [], "services": [], "api": [], "utils": [], "main": [], "other": []}
        for file_item in all_files:
            file_path = file_item.get("file_path", "")
            if file_path.startswith("backend/models/") or file_path.startswith("models/"):
                layers["models"].append(file_path)
            elif file_path.startswith("backend/services/") or file_path.startswith("services/"):
                layers["services"].append(file_path)
            elif file_path.startswith("backend/api/") or file_path.startswith("api/"):
                layers["api"].append(file_path)
            elif file_path.startswith("backend/utils/") or file_path.startswith("utils/"):
                layers["utils"].append(file_path)
            elif file_path.endswith("app.py") or file_path.endswith("main.py"):
                layers["main"].append(file_path)
            else:
                layers["other"].append(file_path)

        # 提取流程信息
        core_flows = business_context.get("core_flows", [])
        flow_names = [f.get("name", "") for f in core_flows if f.get("name")]

        return {
            "project_name": schema_data.get("project_name", "未命名项目"),
            "project_desc": schema_data.get("project_desc", ""),
            "overview": business_context.get("overview", ""),
            "user_roles": business_context.get("user_roles", []),
            "core_flows": flow_names,
            "layers": layers,
            "generation_order": [item.get("file", "") if isinstance(item, dict) else item for item in generation_order],
            "total_files": len(all_files),
            "data_models": schema_data.get("data_models", {}),
            "error_registry": schema_data.get("error_registry", {}),
            "layer_rules": schema_data.get("layer_rules", {}),
        }

    @staticmethod
    def build_ready_files_summary(
            ready_files: List[dict],
            all_files: List[dict]
    ) -> str:
        """
        从已生成文件列表中提取摘要（供提示词使用）
        ready_files: 已生成并通过校验的文件列表
        all_files: 所有文件的 Schema 定义（用于补充文件职责信息）
        """
        if not ready_files:
            return "（还没有已生成的文件，你是第一个）"

        summary_lines = []
        summary_lines.append("✅ 已生成并验证的文件：")

        # 构建文件路径 → file_item 映射
        file_map = {f.get("file_path", ""): f for f in all_files}

        for item in ready_files:
            file_path = item.get("file_path", "")
            file_item = file_map.get(file_path, {})
            purpose = file_item.get("file_purpose", "")
            business_role = file_item.get("business_role", {})
            in_flows = business_role.get("in_flows", [])

            # 获取真实的导出接口（从 AST 提取）
            exports = item.get("actual_exports", [])
            export_str = ", ".join(exports[:5]) if exports else "无公开接口"
            if len(exports) > 5:
                export_str += f" 等 {len(exports)} 个"

            lines = [f"  - {file_path}"]
            if purpose:
                lines.append(f"    职责：{purpose}")
            if in_flows:
                lines.append(f"    参与流程：{', '.join(in_flows)}")
            lines.append(f"    导出接口：{export_str}")
            summary_lines.append("\n".join(lines))

        return "\n".join(summary_lines)

    @staticmethod
    def build_upcoming_files_summary(
            generation_order: List[str],
            current_index: int,
            all_files: List[dict]
    ) -> str:
        """
        从 generation_order 中提取即将生成的文件列表
        """
        if not generation_order or current_index >= len(generation_order) - 1:
            return "（没有后续文件，这是最后一个）"

        file_map = {f.get("file_path", ""): f for f in all_files}
        upcoming = []

        # 只显示接下来 3-5 个文件
        next_indices = range(current_index + 1, min(current_index + 6, len(generation_order)))
        for idx in next_indices:
            file_path = generation_order[idx]
            file_item = file_map.get(file_path, {})
            purpose = file_item.get("file_purpose", "")
            depend_on = file_item.get("depend_on_files", [])

            lines = [f"  {idx + 1}. {file_path}"]
            if purpose:
                lines.append(f"     职责：{purpose}")
            if depend_on:
                lines.append(f"     依赖：{', '.join(depend_on[:3])}")
            upcoming.append("\n".join(lines))

        if not upcoming:
            return "（没有后续文件）"

        summary_lines = ["📋 即将生成的文件："]
        summary_lines.extend(upcoming)
        total_remaining = len(generation_order) - current_index - 1
        if total_remaining > 5:
            summary_lines.append(f"  ... 还有 {total_remaining - 5} 个文件")

        return "\n".join(summary_lines)

    @staticmethod
    def build_file_context(
            file_path: str,
            file_item: dict,
            context: dict,
            ready_files: List[dict],
            ready_summary: str,
            upcoming_summary: str
    ) -> str:
        """
        构建单个文件的完整上下文（三层理解）
        """
        business_role = file_item.get("business_role", {})
        export_apis = file_item.get("export_apis", [])
        call_other_apis = file_item.get("call_other_apis", [])
        depend_on_files = file_item.get("depend_on_files", [])

        # 第一层：项目全貌
        level1 = f"""
============================================================
📋 项目全貌（你正在参与的项目）
============================================================
项目名称：{context.get('project_name', '未命名')}
项目概述：{context.get('overview', '')}

核心业务流程：
{chr(10).join(['  - ' + f for f in context.get('core_flows', [])])}

用户角色：
{chr(10).join(['  - ' + r for r in context.get('user_roles', [])])}

文件总数：{context.get('total_files', 0)} 个
数据模型：{', '.join(context.get('data_models', {}).keys()) if context.get('data_models') else '无'}
============================================================
"""

        # 第二层：当前文件角色定位
        in_flows = business_role.get("in_flows", [])
        flow_position = business_role.get("flow_position", "")
        business_rules = business_role.get("business_rules", [])
        business_data = business_role.get("business_data", {})

        level2 = f"""
============================================================
📌 当前文件角色定位
============================================================
文件路径：{file_path}
文件职责：{file_item.get('file_purpose', '')}

参与的业务流程：{', '.join(in_flows) if in_flows else '无'}
在流程中的位置：{flow_position if flow_position else '未指定'}

业务规则：
{chr(10).join(['  - ' + r for r in business_rules]) if business_rules else '  无'}

业务数据：
  输入：{business_data.get('input', '未指定')}
  输出：{business_data.get('output', '未指定')}
  副作用：{', '.join(business_data.get('side_effects', [])) if business_data.get('side_effects') else '无'}

依赖文件：{', '.join(depend_on_files) if depend_on_files else '无'}
导出接口：{', '.join([a.get('name', '') for a in export_apis]) if export_apis else '无'}
调用接口：{', '.join([c.get('api_name', '') for c in call_other_apis]) if call_other_apis else '无'}
============================================================
"""

        # 第三层：上下文感知
        level3 = f"""
============================================================
🔗 上下文感知（你与项目的关系）
============================================================
{ready_summary}

{upcoming_summary}

总进度：{len(ready_files)}/{context.get('total_files', 0)} 个文件已完成
============================================================
"""

        return f"{level1}\n{level2}\n{level3}"

    @staticmethod
    def format_context_for_prompt(context: dict) -> str:
        """
        把上下文格式化为提示词可插入的文本块（简化版，用于快速展示）
        """
        lines = []
        lines.append("【项目全貌】")
        lines.append(f"项目：{context.get('project_name', '')}")
        lines.append(f"概述：{context.get('overview', '')}")
        lines.append(f"流程：{', '.join(context.get('core_flows', []))}")
        lines.append(f"角色：{', '.join(context.get('user_roles', []))}")
        return "\n".join(lines)

    @staticmethod
    def _collect_all_files(tree_list) -> List[dict]:
        """递归收集所有文件"""
        result = []
        for dir_item in tree_list:
            files = dir_item.get("files", [])
            result.extend(files)
            sub_dirs = dir_item.get("sub_dirs", [])
            if sub_dirs:
                result.extend(ProjectContextBuilder._collect_all_files(sub_dirs))
        return result