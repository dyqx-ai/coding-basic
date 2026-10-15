import os
import json
import re
import base64

from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()
config_path = "../../config_file/my_config.yaml"

try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败，请检查路径: {config_path}, 错误: {e}") from e

project_path = config["change_project_path"]
absolute_project_path = os.path.abspath(project_path)
# 摘要缓存目录
ai_summary_dir = os.path.join(absolute_project_path, ".ai_file_summary")
summary_md_path = os.path.join(ai_summary_dir, "summary.md")

def get_openai_model(model_provider: str, model_name: str):
    if model_provider == "ali":
        api_key = os.getenv("ALI_API_KEY")
    elif model_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None
    return api_key, model_name


# ========== 修改子图专属模型 ==========
# 1.检测摘要md是否存在 + 文件任务分析（轻量flash，省token）
api_key_analyze_change, model_name_analyze_change = get_openai_model(
    config["analyze_change_model"]["model_provider"],
    config["analyze_change_model"]["model_name"])

# 2.文件摘要生成模型（flash）
api_key_summary, model_name_summary = get_openai_model(
    config["file_summary_model"]["model_provider"],
    config["file_summary_model"]["model_name"])

# 3.执行代码修改，需要更强能力
api_key_code_generate, model_name_code_generate = get_openai_model(
    config["code_change_model"]["model_provider"],
    config["code_change_model"]["model_name"])


# ========== 健壮性：API密钥校验 ==========
def _check_api_key(provider_name, key):
    if not key:
        raise RuntimeError(f"[{provider_name}] API密钥未配置，请检查.env文件")


_check_api_key(config["analyze_change_model"]["model_provider"], api_key_analyze_change)
_check_api_key(config["file_summary_model"]["model_provider"], api_key_summary)
_check_api_key(config["code_change_model"]["model_provider"], api_key_code_generate)


model_analyze_change = ChatOpenAI(
    model=model_name_analyze_change,
    api_key=api_key_analyze_change,
    base_url=config["analyze_change_model"]["base_url"],
    temperature=0.1,
    timeout=60
)

model_file_summary = ChatOpenAI(
    model=model_name_summary,
    api_key=api_key_summary,
    base_url=config["file_summary_model"]["base_url"],
    temperature=0.1,
    timeout=60
)

model_code_change = ChatOpenAI(
    model=model_name_code_generate,
    api_key=api_key_code_generate,
    base_url=config["code_change_model"]["base_url"],
    temperature=0.2,
    timeout=60
)


################################################################################################################
# json数据清洗
def clean_json_data(json_str: str) -> str:
    """
    清洗大模型返回原始字符串，提取纯净JSON字符串
    1. 移除 ```json / ``` 代码标记
    2. 截取最外层第一组 {} / [] JSON
    3. strip 多余空白换行
    """
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


def parse_local_summary_md(file_path: str) -> list[dict]:
    """直接读取磁盘存储的JSON数组，summary.md里面存放纯JSON"""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return []
        data = json.loads(content)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, Exception) as e:
        print(f"[parse_local_summary_md 读取磁盘摘要损坏] {e}")
        return []


################################################################################################################
#图状态定义
class CheckAndChangeCode(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 消息历史，langgraph自动reducer合并消息
    user_change_requirement: str | None  # 用户提出的代码修改原始需求
    change_is_ok: bool  # 是否已经完成修改校验，True代表修改通过可以输出结果
    file_name_list: list[str]  # 需要处理的目标文件名称列表
    change_content_list: list[str]  # 对应每个文件的修改后源码内容列表，和file_name_list一一对应
    project_file_tree: str  # 当前存量项目的目录树文本，供大模型参考项目结构
    file_summary_list: list[dict]  # 文件摘要对象列表，存储各个文件对外接口、函数、依赖信息
    next_file_name_list: list[str]  # 下一轮循环需要读取/处理的文件列表
    user_question: list[str] | None  # 【预留暂未使用】需要向用户发起询问的问题集合，为空代表无需提问
    loop_count: int  # 当前已经执行的循环轮次计数，用于限制最大循环次数
    max_loop: int  # 允许最大循环次数，防止无限循环死跑
    has_local_summary_md: bool  # ✨标记是否存在本地摘要md文件，判断是否可以读取缓存摘要
    is_enough_info: bool # ✅修复：补上缺失字段

    # --- 节点6输出，透传给节点3，平铺状态，不传递原始JSON字符串 ---
    insufficient_reason: str | None  # 信息不足的原因，首次=None
    need_additional_file_list: list[str] | None  # 节点6建议补充读取的文件，首次=None

    # --- 节点4产出：本轮读取到的文件源码 {rel_path:source_code} ---
    target_file_source_map: dict[str, str] | None
    # --- 节点6成功分支产出：修改完成的文件 {rel_path:new_source_code} ---
    changed_file_map: dict[str, str] | None#修改完成的文件 {rel_path:new_source_code}
    processed_file_list: list[str]  # 本轮已经读取处理过的文件黑名单，禁止重复选取，规避死循环

################################################################################################################
#全部提示词（所有非占位符大括号已全部转义，format零报错）
################################################################################################################
#节点2提示词，判断项目是否存在摘要md文件
detect_summary_md_prompt = """
你收到一份项目目录树文本，请扫描全部目录与文件，自主识别目录树内是否存在**项目代码摘要/架构摘要类markdown文档**。
摘要文档定义：用来记录工程各个文件职责、模块说明、函数对外接口、项目架构总结的md文件，不是业务源码、不是普通说明文档。

## 输出强制要求
严格输出 JSON 对象，**禁止输出任何思考、解释、markdown、```代码块、多余换行、多余文字**。

### JSON字段定义
1. exist_summary_file: bool
   - true：目录树里面找到了符合定义的摘要md文档
   - false：目录树没有找到任何摘要md文档
2. summary_file_rel_path: str
   - 如果存在：输出该摘要文件的真实相对路径（从项目根出发）；如果识别到多个，取第一个即可
   - 如果不存在：填写空字符串 `""`

### ✅正确输出示例（严格照搬格式，不要增加额外字符）
{{"exist_summary_file":true,"summary_file_rel_path":".ai_file_summary/summary.md"}}

{{"exist_summary_file":false,"summary_file_rel_path":""}}

【项目目录树】
{PROJECT_TREE}
"""

################################################################################################################
#用在节点3，分析项目修改任务，并输出需要读取的文件列表
analyze_change_task_prompt = """
你是项目修改分析师，结合用户修改需求、项目目录树、已经收集完成的全部文件摘要，还有上一轮执行修改节点反馈的信息不足原因、建议补充读取的文件，做出判断。

【用户修改需求】
{USER_CHANGE_REQ}

【项目完整目录树】
{PROJECT_TREE}

【已经收集的文件摘要信息】
{SUMMARY_CONTEXT}

【上一轮信息不足原因（若无则为空）】
{INSUFFICIENT_REASON}

【上一轮处理过的文件黑名单】
{PROCESSED_FILE_LIST}

【上一轮建议补充读取的文件列表（若无则为空数组）】
{SUGGEST_ADDITIONAL_FILES}

你的任务：
1. 综合全部信息，判断本轮需要读取哪些文件；
2. 需要过滤掉已经拥有完整摘要的文件，不要重复读取；
3. 对上一轮模型建议补充的文件做复核，合理则纳入，不合理直接丢弃。

输出JSON只能包含下面2个key：
need_call_file_list: list[str]；本轮真正需要读取的文件相对路径，禁止重复已经拿到摘要的文件。

样例A：需要继续读取文件
{{
  "need_call_file_list": ["src/config.py","src/db.py"]
}}

样例B：不需要读取文件
{{
  "need_call_file_list": []
}}

硬性规则：
1.只返回纯净JSON字符串，不要任何额外文字、markdown、```标记。
2.不要编造不存在函数、依赖、文件路径。
3.不要把已经获取摘要的文件放入need_call_file_list。
"""

################################################################################################################
#摘要提示词，用在节点5，用于总结文件摘要
file_summary_prompt = """
你是资深Python代码分析专家，本次任务：**仅分析传入的这一个源码文件，输出单个JSON字典对象，不要输出数组，不要一次性处理多个文件，每个文件单独处理**。
输出结果字典后续会直接存入LangGraph状态的file_summary_list列表中，每一个文件对应一份独立摘要字典。

## 输出字段严格定义，一个都不能少
1. rel_path: str
    当前被分析文件的项目相对路径，直接复制输入的{REL_PATH}，不要修改路径字符串。
2. file_desc: str
    文件完整业务职责总结；写清楚这个文件解决什么问题、承担什么模块职责、核心业务逻辑，描述尽量详实。
3. export_apis: list[str]
    只提取本文件对外暴露可供其它文件导入调用的类、函数完整签名；
    只写签名，不要写函数内部实现代码；
    没有对外暴露接口返回空数组[]，禁止返回null。
4. depend_on: list[str]
    本文件依赖的**本项目内部其它文件的相对路径**；
    只填项目内文件，第三方库不要写进来；
    没有内部依赖就返回空数组[]，禁止返回null。

## ✅参考样例（下面为阅读方便做了换行，你的输出必须是压缩单行JSON，不要带换行缩进）
{{
  "rel_path": "service/user.py",
  "file_desc": "用户业务模块，封装用户新增、查询、更新业务逻辑，对外提供用户操作能力，供上层接口调用。",
  "export_apis": [
    "def create_user(username: str, password: str) -> dict",
    "def get_user_info(user_id:int) -> dict"
  ],
  "depend_on": [
    "common/db.py",
    "common/tools.py"
  ]
}}

上面样例仅供看懂结构，**你的输出必须压缩成一整行，不能带换行缩进**。
正确单行输出参考：
{{"rel_path":"service/user.py","file_desc":"用户业务模块，封装用户新增、查询、更新业务逻辑，对外提供用户操作能力。","export_apis":["def create_user(username: str, password: str) -> dict","def get_user_info(user_id:int) -> dict"],"depend_on":["common/db.py","common/tools.py"]}}

## ❌禁止错误格式
1. 禁止外层套 [] 数组，不要返回 [ {{...}} ]
2. 禁止编造源码不存在的函数、类、依赖文件
3. 禁止输出markdown、```json、思考、解释、前置后置文字
4. 字段缺失、字段为null都是错误；无内容必须使用空列表[]

硬性约束：
1. 只输出纯净单行JSON字典，不允许任何额外字符。
2. 所有逻辑全部基于【文件完整源码】，源码没有的内容绝不编造。
3. export_apis、depend_on 无内容必须返回[]，绝对不能返回null。

【文件相对路径】
{REL_PATH}

【文件完整源码】
{SOURCE_CODE}
"""

################################################################################################################
#执行修改提示词，用在节点6，支持两种分支：信息不足 / 可以执行代码修改
execute_change_code_prompt = """
你是资深Python开发工程师。
输入：用户修改需求、项目完整目录树、已收集的文件摘要、当前已经读取的文件源码、旧版已处理文件黑名单。
你有两种判断分支：
分支1：现有信息不足，缺少部分依赖文件，无法安全修改代码。注意补足文件的时候，如果还需要别的文件，把需要的文件名称加入need_additional_file_list进行补足。
分支2：信息足够，可以执行代码修改，输出全部修改后的完整源码。

【用户修改需求】
{USER_CHANGE_REQ}

【项目完整目录树】
{PROJECT_TREE}

【已收集文件摘要】
{SUMMARY_LIST_JSON}

【旧版已处理文件黑名单】
{PROCESSED_FILE_LIST}
说明：这是上一轮结束后的黑名单。
由你来维护更新黑名单：
1. 你需要结合本轮已经读取分析过的文件，把确认后续不再需要二次读取的文件加入到新的processed_file_list；
2. need_additional_file_list里面放你接下来还需要读取补足的文件；
3. need_additional_file_list不能放入黑名单内已经标记完成的文件，除非你有充足理由需要重读并写在insufficient_reason。

【当前读取到的文件源码】
{FILE_SOURCE_DATA}

## JSON输出字段严格定义，必须全部存在
1. is_enough_info: bool
 true：信息充足，可以执行代码修改；false：信息不足，不能修改。
2. insufficient_reason: str
 信息不足时填写详细原因；信息充足填空字符串""。
3. need_additional_file_list: list[str]
 信息不足：填写你判断还需要补充读取的项目文件相对路径；信息充足返回空数组[]。
 ⚠️一般严禁填入旧黑名单里面的路径，特殊重读场景必须在insufficient_reason写明理由。
4. changed_files: object
 key=文件相对路径，value=修改完成后的完整源码；
 信息不足时，为空对象{{}}；
 ⚠️禁止输出diff片段，必须输出完整文件全部源代码。
5. processed_file_list: list[str]
 ⚠️由你输出**更新后的新版本黑名单**：
基于【旧版已处理文件黑名单】，把本轮分析完毕、后续不再需要读取的文件追加到此列表。
不要无脑全盘复制旧列表，要结合本轮业务判断，决定哪些文件正式标记处理完成。

## ✅样例1：信息不足（输出压缩单行）
{{
  "is_enough_info": false,
  "insufficient_reason": "需要读取common/db.py，确认数据库操作函数签名，否则修改会破坏接口契约",
  "need_additional_file_list": ["common/db.py"],
  "changed_files": {{}},
  "processed_file_list": ["service/user.py"]
}}

## ✅样例2：信息充足，可以修改
{{
  "is_enough_info": true,
  "insufficient_reason": "",
  "need_additional_file_list": [],
  "changed_files": {{
    "service/user.py":"import os\\n\\ndef create_user(username: str):\\n    '''创建用户'''\\n    return {{'username':username}}\\n"
  }},
  "processed_file_list": ["service/user.py","common/db.py"]
}}

## ❌禁止输出
1. 禁止外层包裹数组[]；禁止```json、markdown、思考解释文字。
2. 不要输出diff片段，changed_files内必须存放完整源码。
3. 不要私自改动需求没有提及的无关逻辑。
4. 原有对外接口签名不能随意改动，防止跨文件调用报错。

硬性约束：
1. 只输出纯净单行JSON，无任何多余字符。
2. 不编造需求以外的业务逻辑。
3. 如果需要重读旧黑名单内文件，必须在insufficient_reason写明完整理由。
"""
################################################################################################################
#节点7，修改完成后，重新生成文件摘要
re_summary_modified_prompt = """
你是Python代码分析专家，任务：针对已经修改完成的单个文件，重新生成该文件最新摘要字典。
只处理当前传入的这一个文件，输出**单个JSON字典对象，禁止输出数组**。
输出的摘要会替换内存file_summary_list里面该文件的旧摘要条目，其它文件摘要保持不变。

## 输出字段严格定义
1. rel_path: str
    当前文件相对路径，原样复制输入的{REL_PATH}，不允许修改路径。
2. file_desc: str
    更新后的文件业务职责、功能描述，结合本次代码变更更新描述。
3. export_apis: list[str]
    文件对外暴露的类、函数签名；只写签名，不要实现代码；无接口返回[]，禁止null。
4. depend_on: list[str]
    本文件依赖的本项目内部其他文件路径；只写项目内文件，第三方库忽略；无依赖返回[]，禁止null。

## ✅参考样例（换行仅阅读，输出压缩单行JSON）
{{
  "rel_path": "service/user.py",
  "file_desc": "用户业务模块，本次新增批量查询用户接口，封装用户新增、查询、更新业务逻辑，对外提供用户操作能力。",
  "export_apis": ["def create_user(username: str, password: str) -> dict", "def batch_query_user(user_ids:list[int])->list[dict]"],
  "depend_on": ["common/db.py","common/tools.py"]
}}

单行输出参考：
{{"rel_path":"service/user.py","file_desc":"用户业务模块，本次新增批量查询用户接口，封装用户新增、查询、更新业务逻辑，对外提供用户操作能力。","export_apis":["def create_user(username: str, password: str) -> dict","def batch_query_user(user_ids:list[int])->list[dict]"],"depend_on":["common/db.py","common/tools.py"]}}

## ❌禁止输出
1. 禁止外层套[]数组，不要返回列表
2. 禁止```json、markdown、思考、解释、多余文字
3. list字段无内容必须返回[]，严禁返回null
4. 不要处理除此文件以外的其它文件

## 强制约束
1. 完全基于【修改之后完整源码】生成最新摘要，不要沿用旧摘要里面过时信息。
2. 如实提取变更之后的对外接口、内部依赖，不能编造不存在内容。
3. 只输出纯净JSON字典，无任何额外字符。

【文件相对路径】
{REL_PATH}

【修改完成后的完整源码】
{CHANGED_FILE_SOURCE}
"""

################################################################################################################
#从本地项目读取工程结构
def scan_project_tree_os(root_dir: str):
    skip_dirs = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "build", "dist", ".ai_file_summary", ".modify_backup"}
    lines = []
    root = os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        depth = dirpath.replace(root,"").count(os.sep)
        indent = "  " * depth
        dir_name = os.path.basename(dirpath)
        lines.append(f"{indent}- {dir_name}/")
        sub_indent = "  " * (depth + 1)
        for fname in filenames:
            lines.append(f"{sub_indent}- {fname}")
    return "\n".join(lines)

#节点1：从本地项目读取工程结构
def read_local_project_tree_node(state: CheckAndChangeCode) -> dict:
    the_project_tree = scan_project_tree_os(str(absolute_project_path))
    return {
        "project_file_tree": the_project_tree
    }

################################################################################################################
#节点2：detect_exist_summary_md_node 大模型判断是否存在摘要md，存在则【纯Python解析加载JSON摘要】
def detect_exist_summary_md_node(state: CheckAndChangeCode) -> dict:
    tree_text = state["project_file_tree"]
    prompt = detect_summary_md_prompt.format(PROJECT_TREE=tree_text)
    new_has_md = False
    merge_summary = state["file_summary_list"].copy()
    try:
        resp = model_analyze_change.invoke(prompt)
        raw = clean_json_data(resp.content)
        detect_result = json.loads(raw)
        exist = detect_result.get("exist_summary_file", False)
        rel_path = detect_result.get("summary_file_rel_path", ".ai_file_summary/summary.md")
        # 防护：模型幻觉返回非md后缀路径直接作废
        if exist and not rel_path.endswith(".md"):
            exist = False
        if exist:
            real_md_path = os.path.join(absolute_project_path, rel_path)
            if os.path.exists(real_md_path):
                new_has_md = True
                # 读取磁盘JSON摘要
                loaded_summary_arr = parse_local_summary_md(real_md_path)
                exist_map = {item["rel_path"]: item for item in merge_summary}
                for s in loaded_summary_arr:
                    exist_map[s["rel_path"]] = s
                merge_summary = list(exist_map.values())
    except Exception as e:
        print(f"[detect_exist_summary_md_node 加载缓存md异常] {e}")

    return {
        "has_local_summary_md": new_has_md,
        "file_summary_list": merge_summary
    }

################################################################################################################
#节点3：analyze_task_node 任务分析节点
def analyze_task_node(state: CheckAndChangeCode) -> dict:
    user_req = state["user_change_requirement"]
    tree_text = state["project_file_tree"]
    old_summary_list = state["file_summary_list"]
    insufficient_reason = state.get("insufficient_reason")
    suggest_add_files = state.get("need_additional_file_list") or []
    processed_file_list = state.get("processed_file_list") or []

    summary_context = json.dumps(old_summary_list, ensure_ascii=False)
    ins_reason_text = insufficient_reason if insufficient_reason is not None else ""
    suggest_file_text = json.dumps(suggest_add_files, ensure_ascii=False)

    prompt = analyze_change_task_prompt.format(
        USER_CHANGE_REQ=user_req,
        PROJECT_TREE=tree_text,
        SUMMARY_CONTEXT=summary_context,
        INSUFFICIENT_REASON=ins_reason_text,
        SUGGEST_ADDITIONAL_FILES=suggest_file_text,
        PROCESSED_FILE_LIST=processed_file_list
    )
    try:
        resp = model_analyze_change.invoke(prompt)
        raw_json = clean_json_data(resp.content)
        result = json.loads(raw_json)
    except Exception as e:
        print(f"[analyze_task_node 模型调用异常] {e}")
        result = {
            "need_call_file_list": []
        }
    next_list = result.get("need_call_file_list", [])
    # 代码层二次过滤，过滤掉已经拥有摘要的文件，双重防护，防止模型违规输出
    existed_rel = {item["rel_path"] for item in old_summary_list}
    next_list = [p for p in next_list if p not in existed_rel]
    # 清空上一轮循环残留的状态，本轮重新开始
    return {
        "next_file_name_list": next_list,
        "insufficient_reason": None,
        "need_additional_file_list": None
    }

################################################################################################################
#节点4 read_target_file_node：读取磁盘指定文件，输出target_file_source_map，增加路径穿越防护
def read_target_file_node(state: CheckAndChangeCode) -> dict:
    rel_path_list = state["next_file_name_list"]
    source_map = {}
    base_abs = os.path.abspath(absolute_project_path)
    for rel in rel_path_list:
        full_p = os.path.join(absolute_project_path, rel)
        full_p = os.path.abspath(full_p)
        # 拦截路径逃逸 ../../
        if not full_p.startswith(base_abs):
            print(f"[路径逃逸拦截，跳过] {rel}")
            continue
        if os.path.isfile(full_p):
            try:
                with open(full_p, "r", encoding="utf-8") as f:
                    src = f.read()
                source_map[rel] = src
            except Exception as e:
                print(f"读取文件失败 {rel} : {e}")

    # key为文件相对路径，value为对应源码，天然标识每个代码归属哪个文件
    # 将本轮尝试读取过的全部文件加入已处理黑名单
    new_processed = state["processed_file_list"] + rel_path_list

    return {
        "target_file_source_map": source_map,
        "processed_file_list": new_processed
    }

################################################################################################################
#节点5 generate_file_summary_node：批量生成文件摘要，按rel_path覆盖合并，禁止直接append
def generate_file_summary_node(state: CheckAndChangeCode) -> dict:
    old_summaries = state["file_summary_list"]
    source_map = state.get("target_file_source_map") or {}
    exist_map = {item["rel_path"]: item for item in old_summaries}

    for rel_path, src in source_map.items():
        pmt = file_summary_prompt.format(REL_PATH=rel_path, SOURCE_CODE=src)
        try:
            r = model_file_summary.invoke(pmt)
            j_raw = clean_json_data(r.content)
            obj = json.loads(j_raw)
            exist_map[rel_path] = obj
        except Exception as e:
            print(f"生成摘要失败 {rel_path}: {e}")
    new_summary_list = list(exist_map.values())
    return {"file_summary_list": new_summary_list}

################################################################################################################
#节点6 execute_change_node：执行代码修改，支持信息不足分支，解析后平铺写入state
def execute_change_node(state: CheckAndChangeCode) -> dict:
    user_req = state["user_change_requirement"]
    tree_text = state["project_file_tree"]
    summary_list = state["file_summary_list"]
    source_map = state.get("target_file_source_map") or {}
    processed_file_list = state.get("processed_file_list") or []

    prompt = execute_change_code_prompt.format(
        USER_CHANGE_REQ=user_req,
        PROJECT_TREE=tree_text,
        SUMMARY_LIST_JSON=json.dumps(summary_list, ensure_ascii=False),
        FILE_SOURCE_DATA=json.dumps(source_map, ensure_ascii=False),
        PROCESSED_FILE_LIST=processed_file_list
    )
    try:
        resp = model_code_change.invoke(prompt)
        raw = clean_json_data(resp.content)
        change_result = json.loads(raw)
    except Exception as e:
        print(f"[execute_change_node 代码修改异常] {e}")
        change_result = {
            "is_enough_info": False,
            "insufficient_reason": "模型调用异常",
            "need_additional_file_list": [],
            "changed_files": {},
        }
    is_enough = change_result.get("is_enough_info", False)
    ins_reason = change_result.get("insufficient_reason", "")
    add_file_list = change_result.get("need_additional_file_list", [])
    changed_files = change_result.get("changed_files", {})
    new_processed = change_result.get("processed_file_list", [])

    file_names = list(changed_files.keys())
    contents = list(changed_files.values())

    return {
        "is_enough_info": is_enough,
        "insufficient_reason": ins_reason,
        "need_additional_file_list": add_file_list,
        "changed_file_map": changed_files,
        "file_name_list": file_names,
        "change_content_list": contents,
        "processed_file_list": new_processed
    }

################################################################################################################
# ✅新增节点：循环计数自增，条件边不能修改state
def inc_loop_count_node(state: CheckAndChangeCode) -> dict:
    return {"loop_count": state["loop_count"] + 1}

################################################################################################################
#节点7 re_summary_modified_node：对修改后的源码重新生成摘要，替换旧条目，只更新内存，不写磁盘
def re_summary_modified_node(state: CheckAndChangeCode) -> dict:
    change_result_dict = state.get("changed_file_map") or {}
    summary_list = state["file_summary_list"].copy()
    summary_map = {s["rel_path"]: s for s in summary_list}

    for rel_path, new_src in change_result_dict.items():
        pmt = re_summary_modified_prompt.format(REL_PATH=rel_path, CHANGED_FILE_SOURCE=new_src)
        try:
            r = model_file_summary.invoke(pmt)
            j_raw = clean_json_data(r.content)
            new_obj = json.loads(j_raw)
            summary_map[rel_path] = new_obj
        except Exception as e:
            print(f"刷新文件摘要失败 {rel_path}: {e}")
    return {"file_summary_list": list(summary_map.values())}

################################################################################################################
#节点8 write_file_node：写入修改后的源码文件，备份旧文件，新增路径穿越防护
def write_file_node(state: CheckAndChangeCode) -> dict:
    change_result_dict = state.get("changed_file_map") or {}
    modify_backup_dir = os.path.join(absolute_project_path, ".modify_backup")
    os.makedirs(modify_backup_dir, exist_ok=True)
    base_abs = os.path.abspath(absolute_project_path)
    ok_flag = True
    for rel_path, new_code in change_result_dict.items():
        full_path = os.path.join(absolute_project_path, rel_path)
        full_path = os.path.abspath(full_path)
        # 写入路径穿越防护，禁止写出项目目录
        if not full_path.startswith(base_abs):
            print(f"[写入路径逃逸拦截，跳过] {rel_path}")
            ok_flag = False
            continue
        # base64编码rel_path，彻底解决不同目录同名文件备份覆盖问题
        safe_filename = base64.urlsafe_b64encode(rel_path.encode("utf-8")).decode("utf-8")
        backup_path = os.path.join(modify_backup_dir, safe_filename)
        try:
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    old_code = f.read()
                with open(backup_path, "w", encoding="utf-8") as f:
                    f.write(old_code)
            parent = os.path.dirname(full_path)
            os.makedirs(parent, exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(new_code)
        except Exception as e:
            print(f"写入文件失败 {rel_path}: {e}")
            ok_flag = False
    return {"change_is_ok": ok_flag}

################################################################################################################
#节点9 dump_summary_md_node：纯Python做增量合并摘要，磁盘存储纯JSON数组，不再调用LLM
def dump_summary_md_node(state: CheckAndChangeCode) -> dict:
    os.makedirs(ai_summary_dir, exist_ok=True)
    memory_summary_list = state["file_summary_list"]
    old_disk_list = []
    if state["has_local_summary_md"] and os.path.exists(summary_md_path):
        old_disk_list = parse_local_summary_md(summary_md_path)

    old_disk_map = {item["rel_path"]: item for item in old_disk_list}
    memory_map = {item["rel_path"]: item for item in memory_summary_list}
    merged_map = {**old_disk_map, **memory_map}
    merged_final_list = list(merged_map.values())

    # 只有发生变更才写磁盘，减少IO
    if merged_final_list != list(old_disk_map.values()):
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(merged_final_list, ensure_ascii=False, indent=2))

    return {"file_summary_list": merged_final_list}

################################################################################################################
# -------------------------- 条件路由函数，放在节点6之后 --------------------------
def route_after_execute_change(state: CheckAndChangeCode):
    """
    节点6执行完成之后路由
    1.信息不足，且没有到达max_loop → 走到计数节点inc_loop_count_node
    2.信息不足，到达最大循环 → END
    3.信息充足 → 向下执行 re_summary_modified_node
    """
    is_enough = state.get("is_enough_info", False)
    loop_count = state["loop_count"]
    max_loop = state["max_loop"]
    if not is_enough:
        if loop_count < max_loop:
            return "inc_loop_count_node"
        else:
            return END
    else:
        return "re_summary_modified_node"

################################################################################################################
# -------------------------- 构建图 --------------------------
def build_change_graph():
    change_project_graph = StateGraph(CheckAndChangeCode)

    change_project_graph.add_node("read_local_project_tree_node", read_local_project_tree_node)
    change_project_graph.add_node("detect_exist_summary_md_node", detect_exist_summary_md_node)
    change_project_graph.add_node("analyze_task_node", analyze_task_node)
    change_project_graph.add_node("read_target_file_node", read_target_file_node)
    change_project_graph.add_node("generate_file_summary_node", generate_file_summary_node)
    change_project_graph.add_node("execute_change_node", execute_change_node)
    change_project_graph.add_node("inc_loop_count_node", inc_loop_count_node)
    change_project_graph.add_node("re_summary_modified_node", re_summary_modified_node)
    change_project_graph.add_node("write_file_node", write_file_node)
    change_project_graph.add_node("dump_summary_md_node", dump_summary_md_node)


    change_project_graph.add_edge(START, "read_local_project_tree_node")
    change_project_graph.add_edge("read_local_project_tree_node", "detect_exist_summary_md_node")
    change_project_graph.add_edge("detect_exist_summary_md_node", "analyze_task_node")
    change_project_graph.add_edge("analyze_task_node", "read_target_file_node")
    change_project_graph.add_edge("read_target_file_node", "generate_file_summary_node")
    change_project_graph.add_edge("generate_file_summary_node", "execute_change_node")

    # 节点6之后条件路由：跳计数节点 / re_summary_modified_node / 终止
    change_project_graph.add_conditional_edges(
        "execute_change_node",
        route_after_execute_change
    )
    # 计数完成，回到任务分析节点继续循环
    change_project_graph.add_edge("inc_loop_count_node", "analyze_task_node")

    change_project_graph.add_edge("re_summary_modified_node", "write_file_node")
    change_project_graph.add_edge("write_file_node", "dump_summary_md_node")
    change_project_graph.add_edge("dump_summary_md_node", END)

    # 编译图
    change_graph_app = change_project_graph.compile()
    return change_graph_app

################################################################################################################
# ====================== 使用示例 ======================
if __name__ == "__main__":
    change_graph_app = build_change_graph()
    init_state: CheckAndChangeCode = {
        "messages": [],
        "user_change_requirement": "实现模糊查询功能并更新在主函数中，实现功能，模糊查询，第五个功能",
        "change_is_ok": False,
        "file_name_list": [],
        "change_content_list": [],
        "project_file_tree": "",
        "file_summary_list": [],
        "next_file_name_list": [],
        "user_question": None,
        "loop_count": 0,
        "max_loop": 5,
        "has_local_summary_md": False,
        "is_enough_info": False,
        "insufficient_reason": None,
        "need_additional_file_list": None,
        "target_file_source_map": None,
        "changed_file_map": None,
        "processed_file_list": []
    }
    result = change_graph_app.invoke(init_state)
    print(result)