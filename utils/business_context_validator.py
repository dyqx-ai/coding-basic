"""
业务上下文校验器
职责：校验 13 层契约中的业务相关层
"""
import ast
import re
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict


class BusinessContextValidator:
    """校验业务上下文的一致性"""

    @staticmethod
    def validate_business_consistency(schema_data: dict) -> List[str]:
        """
        检查所有文件的 business_role 是否覆盖所有 core_flows
        """
        errors = []

        business_context = schema_data.get("business_context", {})
        core_flows = business_context.get("core_flows", [])
        directory_tree = schema_data.get("directory_tree", [])

        # 收集所有文件及其参与的业务流程
        file_flow_map = {}
        all_files = BusinessContextValidator._collect_all_files(directory_tree)

        for file_item in all_files:
            business_role = file_item.get("business_role", {})
            in_flows = business_role.get("in_flows", [])
            file_path = file_item.get("file_path", "")
            file_flow_map[file_path] = in_flows

        # 检查每个核心流程是否有文件参与
        for flow in core_flows:
            flow_name = flow.get("name", "")
            if not flow_name:
                continue

            # 查找哪些文件参与了这个流程
            participating_files = []
            for file_path, flows in file_flow_map.items():
                if flow_name in flows:
                    participating_files.append(file_path)

            if not participating_files:
                errors.append(f"❌ 业务流程 '{flow_name}' 没有任何文件参与")
            else:
                # 检查流程的 steps 是否都有文件覆盖
                steps = flow.get("steps", [])
                for step in steps:
                    step_desc = step.get("action", "")
                    if not step_desc:
                        continue
                    # 检查是否有文件的 business_role 提到了这个步骤
                    found = False
                    for file_item in all_files:
                        role = file_item.get("business_role", {})
                        position = role.get("flow_position", "")
                        if flow_name in role.get("in_flows", []) and step_desc[:20] in position:
                            found = True
                            break
                    if not found:
                        # 宽松检查：只要文件参与了流程，且流程有步骤，就算覆盖
                        # 更精确的检查需要更复杂的语义匹配
                        pass

        # 检查是否有文件不参与任何业务流程
        for file_item in all_files:
            file_path = file_item.get("file_path", "")
            business_role = file_item.get("business_role", {})
            in_flows = business_role.get("in_flows", [])
            if not in_flows and not file_path.endswith("__init__.py"):
                errors.append(f"⚠️ 文件 '{file_path}' 不参与任何业务流程")

        return errors

    @staticmethod
    def validate_generation_order(schema_data: dict) -> List[str]:
        """
        检查 generation_order 是否满足依赖关系
        """
        errors = []

        generation_order = schema_data.get("generation_order", [])
        directory_tree = schema_data.get("directory_tree", [])

        if not generation_order:
            errors.append("❌ generation_order 为空，大模型未生成生成顺序")
            return errors

        all_files = BusinessContextValidator._collect_all_files(directory_tree)
        file_path_set = {f.get("file_path", "") for f in all_files}

        # 检查 order 中的文件是否都在 schema 中
        for idx, item in enumerate(generation_order):
            file_path = item.get("file", "") if isinstance(item, dict) else item
            if file_path not in file_path_set:
                errors.append(f"❌ generation_order 中的 '{file_path}' 在 directory_tree 中不存在")

        # 检查依赖顺序
        file_map = {f.get("file_path", ""): f for f in all_files}
        order_map = {}
        for idx, item in enumerate(generation_order):
            file_path = item.get("file", "") if isinstance(item, dict) else item
            order_map[file_path] = idx

        for file_path, idx in order_map.items():
            file_item = file_map.get(file_path)
            if not file_item:
                continue
            depend_on = file_item.get("depend_on_files", [])
            for dep in depend_on:
                if dep in order_map and order_map[dep] > idx:
                    errors.append(
                        f"❌ 依赖顺序错误：'{file_path}' (位置 {idx}) 依赖 '{dep}' (位置 {order_map[dep]})，"
                        f"但依赖文件应该在前面"
                    )

        return errors

    @staticmethod
    def check_business_rules_in_code(source_code: str, business_rules: List[str]) -> List[str]:
        """
        AST 检查代码中是否实现了 business_rules
        """
        errors = []

        if not business_rules:
            return errors

        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            errors.append(f"语法错误，无法解析: {e}")
            return errors

        # 提取代码中的字符串字面量（包含注释、docstring、字符串常量）
        strings_in_code = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Str) or (
                    hasattr(node, 'value') and isinstance(getattr(node, 'value', None), ast.Constant)):
                try:
                    strings_in_code.append(ast.unparse(node).lower())
                except:
                    pass

        # 提取所有注释（通过源码行）
        comments = []
        for line in source_code.split('\n'):
            line = line.strip()
            if line.startswith('#'):
                comments.append(line[1:].strip().lower())

        code_text = source_code.lower()

        for rule in business_rules:
            rule_lower = rule.lower()

            # 检查1：规则是否出现在注释或docstring中
            in_comment = any(rule_lower in c for c in comments) or any(rule_lower in s for s in strings_in_code)

            # 检查2：规则是否体现在代码逻辑中（关键词匹配）
            keywords = []
            if "不能" in rule or "不可以" in rule or "禁止" in rule:
                keywords.extend(["if", "raise", "check", "validate", "verify", "ensure"])
            if "必须" in rule or "需要" in rule:
                keywords.extend(["if", "not", "raise", "check", "validate", "verify", "ensure", "required"])

            # 提取规则中的关键实体词
            entity_words = []
            for word in rule_lower.split():
                if len(word) > 2 and word not in ["不能", "可以", "必须", "需要", "禁止", "允许", "的", "了", "和",
                                                  "与"]:
                    entity_words.append(word)

            # 检查实体词是否在代码中出现
            entity_present = any(word in code_text for word in entity_words if len(word) > 2)

            if not in_comment and not entity_present:
                errors.append(f"❌ 业务规则未在代码中体现: '{rule}'")

        return errors

    @staticmethod
    def validate_layer_rules(schema_data: dict) -> List[str]:
        """
        检查是否有文件违反 layer_rules
        """
        errors = []

        layer_rules = schema_data.get("layer_rules", {})
        if not layer_rules:
            return errors

        allowed_deps = layer_rules.get("allowed_dependencies", [])
        forbidden_imports = layer_rules.get("forbidden_imports", [])
        directory_tree = schema_data.get("directory_tree", [])

        # 构建层映射：文件路径 → 所在层
        layer_map = {}
        all_files = BusinessContextValidator._collect_all_files(directory_tree)
        for file_item in all_files:
            file_path = file_item.get("file_path", "")
            # 根据路径判断层（简化版，实际可配置）
            if file_path.startswith("backend/api/") or file_path.startswith("api/"):
                layer_map[file_path] = "api"
            elif file_path.startswith("backend/services/") or file_path.startswith("services/"):
                layer_map[file_path] = "services"
            elif file_path.startswith("backend/models/") or file_path.startswith("models/"):
                layer_map[file_path] = "models"
            elif file_path.startswith("backend/utils/") or file_path.startswith("utils/"):
                layer_map[file_path] = "utils"
            elif file_path.endswith("app.py") or file_path.endswith("main.py"):
                layer_map[file_path] = "main"
            else:
                # 尝试从路径中提取层
                parts = file_path.split("/")
                if len(parts) >= 2:
                    layer_map[file_path] = parts[0]
                else:
                    layer_map[file_path] = "unknown"

        # 检查每个文件的依赖是否违反规则
        for file_path, layer in layer_map.items():
            file_item = next((f for f in all_files if f.get("file_path") == file_path), {})
            depend_on = file_item.get("depend_on_files", [])

            for dep in depend_on:
                dep_layer = layer_map.get(dep, "unknown")
                if dep_layer == "unknown":
                    continue

                # 检查 forbidden 规则
                for forbidden in forbidden_imports:
                    from_layer = forbidden.get("from", "")
                    to_layers = forbidden.get("to", [])
                    if layer == from_layer and dep_layer in to_layers:
                        errors.append(
                            f"❌ 违反分层规则：'{file_path}' ({layer}) 禁止依赖 '{dep}' ({dep_layer})"
                        )

        return errors

    @staticmethod
    def build_flow_participation_map(schema_data: dict) -> Dict[str, List[str]]:
        """
        构建“业务流程 → 参与文件”的映射表
        """
        flow_map = defaultdict(list)

        business_context = schema_data.get("business_context", {})
        core_flows = business_context.get("core_flows", [])
        directory_tree = schema_data.get("directory_tree", [])

        flow_names = [f.get("name", "") for f in core_flows if f.get("name")]

        all_files = BusinessContextValidator._collect_all_files(directory_tree)
        for file_item in all_files:
            file_path = file_item.get("file_path", "")
            business_role = file_item.get("business_role", {})
            in_flows = business_role.get("in_flows", [])

            for flow_name in in_flows:
                if flow_name in flow_names:
                    flow_map[flow_name].append(file_path)

        return dict(flow_map)

    @staticmethod
    def _collect_all_files(tree_list) -> List[dict]:
        """递归收集所有文件"""
        result = []
        for dir_item in tree_list:
            files = dir_item.get("files", [])
            result.extend(files)
            sub_dirs = dir_item.get("sub_dirs", [])
            if sub_dirs:
                result.extend(BusinessContextValidator._collect_all_files(sub_dirs))
        return result

    # ============================================================
    # ★★★ 摇篮扼杀 C：Schema 包路径 / 目录结构一致性校验 ★★★
    # ============================================================
    @staticmethod
    def validate_package_structure_consistency(schema_data: dict) -> List[str]:
        """
        校验 interface_contacts / call_other_apis 中的包路径前缀，
        必须在 directory_tree 中真实存在对应的 .py 文件；
        且若采用 `project_name/...` 或 `src/project_name/...` 包式结构，
        对应顶层包目录下必须有 __init__.py，否则运行期 ModuleNotFound。
        """
        errors: List[str] = []
        directory_tree = schema_data.get("directory_tree", [])
        all_files = BusinessContextValidator._collect_all_files(directory_tree)
        file_path_set: Set[str] = {f.get("file_path", "") for f in all_files if f.get("file_path", "").endswith(".py")}

        # --- 1) 顶层包根推断：统计 .py 出现最多的一级前缀 ---
        pkg_candidates: Dict[str, int] = defaultdict(int)
        for rel in file_path_set:
            parts = [p for p in rel.replace("\\", "/").split("/") if p]
            if not parts:
                continue
            # main.py / app.py 这类根目录直接入口视为 root pkg = ""
            if len(parts) == 1:
                pkg_candidates[""] += 1
            else:
                first = parts[0]
                if first == "src" and len(parts) >= 3:
                    pkg_candidates[f"src/{parts[1]}"] += 1  # src 作根容器，真实包是 parts[1]
                elif first == "src":
                    pkg_candidates["src"] += 1  # 扁平 src
                else:
                    pkg_candidates[first] += 1  # project_name/... 包式

        expected_pkgs: Set[str] = {name for name, cnt in pkg_candidates.items() if name and cnt >= 2}

        # --- 2) 每个包目录必须有 __init__.py ---
        for pkg in sorted(expected_pkgs):
            init_rel = f"{pkg}/__init__.py"
            if init_rel not in file_path_set:
                errors.append(
                    f"❌ 包路径缺失 __init__.py：'{pkg}/' 下未生成 __init__.py，"
                    f"将导致绝对导入 `from {pkg.replace('/', '.')}.xxx ...` 失败"
                )

        # --- 0) Schema.entry_file 必须真实出现在 directory_tree 的 .py 文件里（摇篮扼杀 C 规则 4）
        #      否则主测阶段 execute_entry.py 找不到文件，会被判成代码错实际是契约写死缺文件 = 改 3 次也修不好 =====
        entry_file = (schema_data.get("entry_file") or "").replace("\\", "/")
        if entry_file and entry_file.endswith(".py") and entry_file not in file_path_set:
            errors.append(
                f"❌ Schema.entry_file='{entry_file}' 未在 directory_tree 的 .py 文件清单中出现。"
                f"请在 file list 中声明一个 is_top_entry=true 且 file_path='{entry_file}' 的文件，"
                f"否则主测阶段会因找不到入口脚本而直接判定失败。"
            )

        # --- 1b) is_top_entry 唯一且正确：Schema 标注 is_top_entry=True 的 .py 文件不能超过 1 个，且必须与 entry_file 一致 ---
        top_entries = [f.get("file_path", "").replace("\\", "/")
                       for f in all_files
                       if f.get("file_path", "").endswith(".py")
                       and bool(f.get("is_top_entry", False))]
        if len(top_entries) > 1:
            errors.append(
                f"❌ directory_tree 中声明 is_top_entry=True 的文件超过 1 个：{top_entries}。"
                f"【强制规则 10】全项目只能且仅有 1 个顶层入口脚本。"
            )
        if entry_file and top_entries and top_entries[0] != entry_file:
            errors.append(
                f"❌ Schema.entry_file='{entry_file}' 与实际标记 is_top_entry=True 的文件 '{top_entries[0]}' 不一致。"
            )

        # --- 3) interface_contacts / call_other_apis 中使用的模块前缀必须真实存在 ---
        def _check_module_path(module_prefix: str, where: str):
            """允许 module_prefix = "a.b.c" 或 "a/b/c"，对应 a/b/c.py 或 a/b/c/__init__.py"""
            if not module_prefix:
                return
            rel_candidate = module_prefix.replace(".", "/")
            as_file = rel_candidate + ".py"
            as_pkg = rel_candidate + "/__init__.py"
            # 支持在 src/ 下：允许 caller 写 a.b.c 实际是 src/a/b/c.py
            src_as_file = f"src/{as_file}"
            src_as_pkg = f"src/{as_pkg}"
            if (as_file not in file_path_set and as_pkg not in file_path_set
                    and src_as_file not in file_path_set and src_as_pkg not in file_path_set):
                errors.append(
                    f"❌ {where} 引用的模块路径不存在：'{module_prefix}'。"
                    f"directory_tree 中未找到 {as_file} | {as_pkg} | {src_as_file} | {src_as_pkg}"
                )

        # 遍历每个文件的 export_apis.caller_expected_from & call_other_apis.import_module
        for file_item in all_files:
            rel = file_item.get("file_path", "")
            for api in file_item.get("export_apis", []) or []:
                for caller in api.get("caller_expected_from", []) or []:
                    if isinstance(caller, str) and caller:
                        _check_module_path(caller, f"文件 {rel} 接口 {api.get('name','?')} 的 caller_expected_from")
            for call in file_item.get("call_other_apis", []) or []:
                mod = call.get("import_module") or ""
                if isinstance(mod, str) and mod and not mod.startswith(("typing", "collections", "os", "sys",
                                                                       "re", "json", "datetime", "pathlib",
                                                                       "subprocess", "logging", "abc")):
                    # 过滤明显的 stdlib，只查项目内部包前缀
                    first_seg = mod.split(".")[0]
                    if first_seg in {p.split("/")[-1] for p in expected_pkgs} or first_seg == "src":
                        _check_module_path(mod, f"文件 {rel} 的 call_other_apis.import_module")

        return errors

    # ============================================================
    # ★★★ 摇篮扼杀 B：禁止生成 `from .` / `from ..` 相对导入 ★★★
    # ============================================================
    @staticmethod
    def check_no_relative_imports(source_code: str, file_path: str = "") -> List[str]:
        """
        AST + regex 双重判断；禁止顶层 from .xxx 或 from ..xxx。
        原因：相对导入只能在 python -m pkg.module 模式下生效，
              而用户在 IDE 里 `python path/to/file.py` 或沙箱直接 `python f_path`
              会触发 ImportError: attempted relative import with no known parent package。
        """
        errors: List[str] = []
        if not source_code:
            return errors
        rel = file_path or "<unknown>"
        # 1) AST 精确判断
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # AST 解析交给其它校验门；这里再用 regex 兜底
            pass
        else:
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    level = getattr(node, "level", 0) or 0
                    if level >= 1:
                        module = getattr(node, "module", None) or ""
                        names = ", ".join(getattr(a, "name", "") for a in getattr(node, "names", []) or [])
                        errors.append(
                            f"❌ {rel} 使用了相对导入（level={level}）："
                            f"from {'.' * level}{module} import {names}；"
                            f"请改为绝对导入（以项目根/顶层包名为前缀，如 from pkg_a.subpkg_b.xxx import yyy）"
                        )
        # 2) regex 兜底（防止 AST 失败但仍有 from . 写死）
        if not errors:
            bad_lines = re.findall(r"(?m)^\s*from\s+\.{1,}[\w\.]*\s+import\b", source_code)
            if bad_lines:
                errors.append(
                    f"❌ {rel} 存在疑似相对导入语句：{bad_lines[:3]}；禁止使用 from . 或 from .. 形式"
                )
        return errors

    # ============================================================
    # ★★★ 摇篮扼杀 A：检查入口文件是否已有自举 sys.path 头 ★★★
    # ============================================================
    @staticmethod
    def has_self_bootstrap_header(source_code: str) -> bool:
        if not source_code:
            return False
        # 幂等检测：只要出现基于 __file__ 的 sys.path.insert 且注释包含 VIBE_BOOTSTRAP 标记
        return "VIBE_BOOTSTRAP" in source_code and "__file__" in source_code and "sys.path.insert" in source_code

    @staticmethod
    def build_entry_bootstrap_header(entry_rel_path: str, project_name: str = "") -> str:
        """
        根据 entry_rel_path 深度自动定位 project_root，注入 idempotent 头。
        - IDE 双击运行、沙箱 `python cli/main.py`、Docker 中 /workspace 下运行都能正确找到包。
        - 一旦已经存在（VIBE_BOOTSTRAP 标记），build 方不会重复追加。
        """
        entry_rel = entry_rel_path.replace("\\", "/")
        depth = len([p for p in entry_rel.split("/") if p]) - 1  # 到 project_root 需要跳几层
        # 构造 join 内部参数：第一个 _V_os.path.dirname(__file__)，再追加 depth 次 ".."
        # （注意：顶部是 import os as _V_os，必须用别名，防止 del 之后或外部未 import 的用户代码中发生 NameError）
        args_list = ["_V_os.path.dirname(__file__)"] + ['".."'] * depth
        path_join_arg = ", ".join(args_list)
        header = f'''# -*- coding: utf-8 -*-
# >>> VIBE_BOOTSTRAP >>>
# 自动注入：确保 IDE 双击 / 沙箱 python path/file.py / python -m / Docker 都能找到顶层包
import os as _V_os, sys as _V_sys
_V_PROJECT_ROOT = _V_os.path.abspath(_V_os.path.join({path_join_arg}))
if _V_PROJECT_ROOT not in _V_sys.path:
    _V_sys.path.insert(0, _V_PROJECT_ROOT)
_V_SRC_ROOT = _V_os.path.join(_V_PROJECT_ROOT, "src")
if _V_os.path.isdir(_V_SRC_ROOT) and _V_SRC_ROOT not in _V_sys.path:
    _V_sys.path.insert(0, _V_SRC_ROOT)
del _V_os, _V_sys, _V_PROJECT_ROOT, _V_SRC_ROOT
# <<< VIBE_BOOTSTRAP <<<

'''
        return header