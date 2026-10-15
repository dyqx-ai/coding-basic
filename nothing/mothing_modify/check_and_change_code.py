import os
import json
import re
from typing import Annotated, TypedDict

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
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

def get_openai_model(model_provider: str, model_name: str):
    if model_provider == "ali":
        api_key = os.getenv("ALI_API_KEY")
    elif model_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None
    return api_key, model_name


# ========== 各个模型密钥、模型名称读取 ==========
# 项目修改分析模型（给当前子图节点2 analyze_task_node 使用）
api_key_project_analyze, model_name_project_analyze = get_openai_model(
    config["project_analyze_model"]["model_provider"],
    config["project_analyze_model"]["model_name"])

# 前向计划模型2参数
api_key_code_front_plan_two, model_name_code_front_plan_two = get_openai_model(
    config["front_plan_model_two"]["model_provider"],
    config["front_plan_model_two"]["model_name"])

# 后向计划模型参数
api_key_code_back, model_name_code_back = get_openai_model(
    config["back_plan_model"]["model_provider"],
    config["back_plan_model"]["model_name"])

#代码生成模型参数
api_key_code_generate, model_name_code_generate = get_openai_model(
    config["code_generate_model"]["model_provider"],
    config["code_generate_model"]["model_name"])

# 项目修改分析模型（给当前子图节点2 analyze_task_node 使用）
api_key_analyze_change, model_name_analyze_change = get_openai_model(
    config["analyze_change_model"]["model_provider"],
    config["analyze_change_model"]["model_name"])


# ========== 健壮性：API密钥校验 ==========
def _check_api_key(provider_name, key):
    if not key:
        raise RuntimeError(f"[{provider_name}] API密钥未配置，请检查.env文件")

_check_api_key("DeepSeek", api_key_code_front_plan_two)
_check_api_key("DeepSeek", api_key_code_back)
_check_api_key("DeepSeek", api_key_code_generate)
_check_api_key("DeepSeek", api_key_analyze_change)


# 项目修改分析模型实例（节点2调用）
model_project_analyze = ChatOpenAI(
    model=model_name_project_analyze,
    api_key=api_key_project_analyze,
    base_url=config["project_analyze_model"]["base_url"]
)

# 前向计划模型2实例
model_front_plan_two = ChatOpenAI(
    model=model_name_code_front_plan_two,
    api_key=api_key_code_front_plan_two,
    base_url=config["front_plan_model_two"]["base_url"],
    temperature=0.1
)

# 后向计划模型实例
model_back_plan = ChatOpenAI(
    model=model_name_code_back,
    api_key=api_key_code_back,
    base_url=config["back_plan_model"]["base_url"],
    temperature=0.1
)

#代码生成模型实例
model_code_generate = ChatOpenAI(
    model=model_name_code_generate,
    api_key=api_key_code_generate,
    base_url=config["code_generate_model"]["base_url"],
    temperature=0.1
)

# 项目修改分析模型实例（节点2调用）
model_analyze_change = ChatOpenAI(
    model=model_name_analyze_change,
    api_key=api_key_analyze_change,
    base_url=config["analyze_change_model"]["base_url"],
    temperature=0.1
)

################################################################################################################
# json数据清洗
def clean_json_data(json_str: str) -> str:
    """
    清洗大模型返回原始字符串，提取纯净JSON字符串
    1. 移除 ```json / ``` 代码标记
    2. 截取最外层第一组 {} JSON对象
    3. strip 多余空白换行
    """
    content = json_str.strip()

    # 去除markdown代码块标记
    if content.startswith("```json"):
        content = content.removeprefix("```json")
        content = content.removesuffix("```")
    elif content.startswith("```"):
        content = content.removeprefix("```")
        content = content.removesuffix("```")

    content = content.strip()

    # 正则提取最外层 {} 块，过滤模型前后夹带的多余说明文字
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    return content.strip()

################################################################################################################
#图状态定义
class CheckAndChangeCode(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 对话消息存储，以追加的方式存储
    user_change_requirement: str | None  # 用户的修改需求
    change_is_ok: bool  # 修改是否成功
    file_name_list: list[str]  # 需要修改的文件名列表
    change_content_list: list[str]  # 修改后的文件内容列表
    project_file_tree: str  # 项目文件树
    file_summary_list: list[object]  # 已经收集完成的全部文件摘要对象集合
    next_file_name_list: list[str]  # 下一轮需要读取的文件名列表
    user_question: list[str] | None  # 需要向用户询问的问题列表
    loop_count: int  # 当前循环计数
    max_loop: int  # 最大循环次数上限

################################################################################################################
#文件修改提示词
file_summary_prompt = """
你是Python代码分析专家，接收单个文件的源码，输出纯净JSON格式的文件摘要。

输出字段说明：
rel_path: 当前文件的相对路径
file_desc: 简明描述该文件的业务职责、主要功能
export_apis: 数组，只写对外暴露的类、函数签名，不要写实现代码
depend_on: 数组，该文件依赖的本项目内其他文件相对路径，无依赖则为空数组

硬性约束：
1.只输出JSON，禁止```、markdown、解释、思考文字。
2.禁止编造不存在函数、依赖。

【文件相对路径】
{REL_PATH}

【文件完整源码】
{SOURCE_CODE}
"""


analyze_change_task_prompt = """
你是项目修改分析师，结合用户修改需求、项目目录树、已经收集完成的全部文件摘要，做出判断。

【用户修改需求】
{USER_CHANGE_REQ}

【项目完整目录树】
{PROJECT_TREE}

【已经收集的文件摘要信息】
{SUMMARY_CONTEXT}

输出JSON只能包含下面3个key：
need_other_file: bool；true=还需要读取别的文件，false=信息足够，停止收集文件
current_file_summary: list[object]；本次处理得到的文件摘要；本次没有传入文件则为空数组[]
  对象字段：rel_path(相对路径),file_desc(文件职责),export_apis(对外接口签名数组),depend_on(依赖项目内文件数组)
need_call_file_list: list[str]；后续需要读取的文件相对路径，禁止重复已经拿到摘要的文件

样例A：还需要继续读取文件
{
  "need_other_file": true,
  "current_file_summary": [
    {
      "rel_path": "src/main.py",
      "file_desc": "程序入口，加载配置启动应用",
      "export_apis": ["def run_app()"],
      "depend_on": ["src/config.py"]
    }
  ],
  "need_call_file_list": ["src/config.py"]
}

样例B：信息足够，不需要再读文件
{
  "need_other_file": false,
  "current_file_summary": [],
  "need_call_file_list": []
}

硬性规则：
1.只返回纯净JSON字符串，不要任何额外文字、markdown、```标记。
2.不要编造不存在函数、依赖、文件路径。
3.不要把已经获取摘要的文件放入need_call_file_list。
"""


execute_change_code_prompt = """
你是资深Python开发工程师，基于用户需求对现有项目做代码修改。

【用户修改需求】
{USER_CHANGE_REQ}

【待修改文件源码】
{FILE_SOURCE_DATA}

硬性约束：
1.输出JSON，key为文件相对路径，value为修改完成后的完整源代码，不要输出diff片段。
2.保留原有业务逻辑，仅实现用户提出的变更，不要随意改动无关代码。
3.严格遵守各个文件对外接口契约，修改之后不能破坏跨文件调用。
4.新增文件直接填入完整代码。
5.禁止思考、解释、markdown、代码块标记，只输出纯粹JSON。
"""

################################################################################################################
#从本地项目读取工程结构
def scan_project_tree_os(root_dir: str):
    skip_dirs = {"venv", ".venv", "__pycache__", ".git", ".idea", ".vscode", "build", "dist", ".ai_file_summary", ".modify_backup"}
    lines = []
    root = os.path.abspath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root):
        # 过滤要跳过的文件夹，直接从dirnames删掉，os.walk就不会进入
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
#节点2：任务分析节点
def analyze_task_node(state: CheckAndChangeCode) -> dict:
    """
    分析修改任务，输出：是否需要继续读文件、本次摘要、下一轮要读取的文件列表
    """
    user_req = state["user_change_requirement"]
    tree_text = state["project_file_tree"]
    summary_list = state["file_summary_list"]

    # 把已有的摘要拼接成上下文文本
    summary_context = "\n".join([str(item) for item in summary_list])

    prompt = analyze_change_task_prompt.format(
        USER_CHANGE_REQ=user_req,
        PROJECT_TREE=tree_text,
        SUMMARY_CONTEXT=summary_context
    )
    try:
        resp = model_analyze_change.invoke(prompt)
        # clean_json_data 后续你自己实现
        raw_json = clean_json_data(resp.content)
        result = json.loads(raw_json)
    except Exception as e:
        print(f"[analyze_task_node 模型调用异常] {e}")
        # 异常兜底：不再读取新文件
        result = {
            "need_other_file": False,
            "current_file_summary": [],
            "need_call_file_list": []
        }

    return {
        "next_file_name_list": result.get("need_call_file_list", []),
        "file_summary_list": result.get("current_file_summary", [])
    }
################################################################################################################
#节点3：