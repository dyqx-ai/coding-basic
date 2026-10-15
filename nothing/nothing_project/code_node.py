import json
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph, START, END
import yaml

load_dotenv()

config_path = "../../config_file/my_config.yaml"

absolute_path = os.path.abspath(__file__)
# 修复：不要把脚本文件名混入项目根目录
script_dir = os.path.dirname(absolute_path)
save_path = os.path.abspath(os.path.join(script_dir, "../new_project"))

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def get_openai_model(model_provider: str, model_name: str):
    if model_provider == "ali":
        api_key = os.getenv("ALI_API_KEY")
    elif model_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        api_key = None
    return api_key, model_name


# 信息补全模型参数
api_key_ask, model_name_ask = get_openai_model(config["ask_model"]["model_provider"], config["ask_model"]["model_name"])

# 前向计划模型1参数
api_key_code_front_plan_one, model_name_code_front_plan_one = get_openai_model(
    config["front_plan_model_one"]["model_provider"],
    config["front_plan_model_one"]["model_name"])

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

# 信息补全模型实例
model_ask = ChatOpenAI(
    model=model_name_ask,
    api_key=api_key_ask,
    base_url=config["ask_model"]["base_url"],
)

# 前向计划模型1实例
model_front_plan_one = ChatOpenAI(
    model=model_name_code_front_plan_one,
    api_key=api_key_code_front_plan_one,
    base_url=config["front_plan_model_one"]["base_url"],
)

# 前向计划模型2实例
model_front_plan_two = ChatOpenAI(
    model=model_name_code_front_plan_two,
    api_key=api_key_code_front_plan_two,
    base_url=config["front_plan_model_two"]["base_url"],
)

# 后向计划模型实例
model_back_plan = ChatOpenAI(
    model=model_name_code_back,
    api_key=api_key_code_back,
    base_url=config["back_plan_model"]["base_url"],
)

#代码生成模型实例
model_code_generate = ChatOpenAI(
    model=model_name_code_generate,
    api_key=api_key_code_generate,
    base_url=config["code_generate_model"]["base_url"],
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
class VibeProjectState(TypedDict):
    # 原有全部字段保持不变
    messages: Annotated[list[dict], add_messages]  # 项目历史消息记录
    user_raw_requirement: str | None  # 用户最原始输入
    user_clarify_answers: list[str] | None  # 用户对澄清问题的多轮回答
    final_clear_requirement: str | None  # 经过补全之后、完备的最终需求（下游所有模型都使用这个，不再使用原始输入）
    need_user_input: bool  # True=需要暂停，等待用户补充输入
    clarify_questions: list[str] | None  # 需要向用户提出的问题列表
    info_enough: bool  # True=信息足够，无需继续提问
    front_plan_one: str | None  # 前向计划模型1输出
    front_plan_two: str | None  # 前向计划模型2输出
    final_plan: str | None  # 最终计划模型输出

################################################################################################################
# 信息补全模型提示词
fill_info_prompt = """
你是项目需求分析专家。
你的任务：结合用户原始描述以及历史补充回答，判断是否需要向用户进一步确认信息。

【重要意图识别规则，优先级最高】
如果用户语句包含如下含义：直接开始、别再问了、就这样、不用继续提问、跳过问题、直接做。
那么直接跳过提问，不要生成任何问题，把全部现有信息整理为 final_requirement，questions为空数组。次要信息允许你自行合理推断。

【内部参考维度，仅供判断，不需要每条都提问】
1.项目核心目标
2.编程语言（本项目固定使用Python）
3.是否需要web接口、对外API
4.是否需要数据库、持久化存储
5.是否调用第三方大模型API
6.是否需要异步逻辑
7.单元测试相关需求
8.部署方式：本地脚本 / docker /服务部署
9.项目规模：demo /正式工程

你的判断规则：
1.只对真正影响整体架构的关键缺失信息生成问题。
2.属于次要细节，不会改动目录结构的，不要提问，自行推断。
3.单次最多输出3个问题，不要一次性抛出大量问题。
4.信息收集完成，或者用户要求直接开始，输出完整final_requirement，questions为空。

严格只输出JSON，禁止多余文字、markdown、解释。
输出格式：
{
  "is_enough": true/false,
  "final_requirement": "整理合并后的完整需求文本",
  "questions": ["问题1","问题2"]
}

示例1 用户要求直接开始
{
  "is_enough": true,
  "final_requirement":"用户需要开发xxx，用户要求直接开始，部分细节由架构阶段推断补齐。",
  "questions":[]
}

示例2 需要补充信息
{
  "is_enough": false,
  "final_requirement":"",
  "questions":["是否需要数据库持久化？","项目是否需要对外web接口？"]
}

示例3 信息充足
{
  "is_enough": true,
  "final_requirement":"开发基于LangGraph的Agent，Python，本地脚本运行，接入deepseek，不需要数据库，编写单元测试。",
  "questions":[]
}

====输入内容====
原始用户需求：{{USER_RAW_INPUT}}
历史补充回答：{{USER_CLARIFY_ANSWERS}}
"""
################################################################################################################
# 前置工程化方案提示词
front_plan_prompt = """
你是专业软件架构师，请根据用户项目需求输出完整工程架构方案。

硬性约束：
1.输出仅使用Markdown，禁止输出JSON，禁止使用```代码块标记。
2.不要输出思考过程，直接输出方案正文。
3.必须包含下面5个固定标题，标题文字不可修改、不可缺失。
4.每个文件写明相对路径，写清职责，写明依赖的项目内其他文件。
5.需要包含配置文件、入口脚本、测试目录、开发依赖，不要遗漏工程组件。
6.【最重要强制规则】在文件清单的每一个文件描述中，明确列出**本文件对外暴露、可供其他文件调用的全部类、函数**；写明函数名称、入参名称、参数含义、返回值说明。
   - 如果该文件没有对外暴露接口，写明：无对外导出接口。
   - 仅写对外公开接口，内部私有函数不需要罗列。
   - 这部分信息是后续代码生成的契约，绝对不能省略、不能写错函数名与参数。

参考输出样例：
## 项目概述
实现一个简易LangGraph Agent，具备任务拆解与工具调用，提供本地脚本入口。

## 项目依赖
- 运行依赖(pip): langgraph, langchain-openai, python-dotenv
- 开发依赖(dev): pytest, black

## 架构整体说明
源码放置src目录；graph相关逻辑独立子模块；tests存放单元测试，业务代码与测试代码分离。

## 目录结构
- src/
  - graph/
  - utils/
- tests/
- .env.example

## 文件清单
- 文件路径：src/main.py
  职责：程序入口，组装并运行langgraph图，读取环境配置
  文件依赖：src/graph/state.py，src/graph/nodes.py
  对外导出接口：
  函数 run_agent_task(user_input: str) -> dict
    参数 user_input：用户输入任务字符串
    返回值：agent执行完成后的结果字典

- 文件路径：src/graph/state.py
  职责：定义Agent状态TypedDict
  文件依赖：无
  对外导出接口：
  类 AgentState(TypedDict)
    字段 user_query: str 用户原始输入
    字段 result: str 任务输出结果

- 文件路径：src/graph/nodes.py
  职责：实现各个业务节点逻辑
  文件依赖：src/graph/state.py
  对外导出接口：
  函数 parse_task_node(state: AgentState) -> AgentState
    参数 state：Agent状态对象
    返回值：更新后的AgentState

- 文件路径：src/utils/helper.py
  职责：通用工具函数，字符串处理，prompt组装
  文件依赖：无
  对外导出接口：
  函数 build_prompt(template: str, params: dict) -> str
    参数 template：prompt模板字符串；params：模板参数字典
    返回值：填充完成的prompt文本

- 文件路径：tests/test_graph.py
  职责：graph节点单元测试
  文件依赖：src/graph/nodes.py
  对外导出接口：无对外导出接口

- 文件路径：.env.example
  职责：环境变量模板，存放密钥配置
  文件依赖：无
  对外导出接口：无对外导出接口

====基于下面用户需求输出方案====
{{USER_REQUIREMENT}}
"""
################################################################################################################
# 【优化后】后置合并评审提示词 back_code_prompt
back_code_prompt = """
你是高级架构评审专家。接收两份Markdown架构方案A、方案B，以及用户原始项目需求。

执行步骤：
1.充分理解用户原始需求，通读方案A、方案B全部内容，重点提取每个文件的【对外导出接口】（类、函数、入参、返回值），这是后续代码生成的核心契约，不可丢失。
2.对比两份方案：目录树结构、模块划分、pip依赖、开发依赖、每个文件职责、文件之间依赖关系、每个文件对外导出接口契约。
3.择优融合：保留两份方案的合理设计，补齐缺失模块；删除重复目录、重复文件、循环依赖、无效依赖包；对冲突的对外接口，选择更贴合业务需求的一份，不允许随意修改函数名、参数定义。
4.输出严格完全匹配 vibe_project_schema_v1 的JSON对象，不要输出任何其他内容。

#强制硬性规则，必须严格遵守
1.仅输出纯净JSON字符串，不允许任何额外字符，禁止思考、说明、解释、小结。
2.禁止markdown标记，禁止```json / ```代码块包裹。
3.只允许使用schema内定义字段，**禁止新增自定义字段，禁止删除schema必填字段**；本次扩展每个file对象增加export_apis字段，用于存放本文件对外公开接口契约。
4.目录与文件路径全部相对于根目录：{{SAVE_PATH}}，schema内填写相对路径，不要填写绝对路径。
5.从两份方案提取信息填充schema：项目名称、项目描述、入口文件、pip运行依赖列表、dev_dependencies开发依赖列表、directory_tree目录树。
6.每个file对象规则：
  - file_path填写文件相对路径
  - file_purpose写清文件职责
  - depend_on_files填写本文件依赖的项目内部其他文件路径
  - is_test_file：测试文件填true，业务/配置文件填false
  - template_content固定为空字符串""
  - export_apis：数组，存放本文件对外暴露的类、函数接口契约，每个元素包含name(接口名称)、type(函数/类)、params(参数列表，包含参数名、含义)、return_desc(返回值说明)；无对外接口则为空数组[]；严禁丢失两份方案中已经定义好的接口签名。
7.递归处理子目录sub_dirs，完整还原嵌套目录结构，不要扁平化丢失层级。
8.依赖校验：depend_on_files内的路径必须是schema中真实存在的文件，禁止引用不存在的文件，杜绝循环依赖。

# vibe_project_schema_v1
{
  "$schema": "vibe_project_schema_v1",
  "project_name": "项目名称",
  "project_desc": "项目描述",
  "entry_file": "src/main.py",
  "dependencies": {
    "pip": [],
    "dev_dependencies": []
  },
  "directory_tree": [
    {
      "path": "src/api",
      "desc": "文件夹说明",
      "files": [
        {
          "file_path": "src/api/route.py",
          "file_purpose": "文件功能描述",
          "depend_on_files": [],
          "is_test_file": false,
          "template_content": "",
          "export_apis": [
            {
              "name": "example_func",
              "type": "函数",
              "params": [{"param_name":"arg1","desc":"参数说明"}],
              "return_desc": "返回值描述"
            }
          ]
        }
      ],
      "sub_dirs": []
    }
  ]
}

用户原始需求：
{{USER_REQUIREMENT}}

方案A：
{{TEXT_PLAN_A}}

方案B：
{{TEXT_PLAN_B}}

文件保存根目录(所有文件文件夹相对该目录)：
{{SAVE_PATH}}
"""
################################################################################################################
# 单文件源码生成提示词（抽离独立常量）
single_file_generate_prompt = """
你是Python高级开发工程师，严格按照契约生成完整可运行Python源码。

【全局项目需求】
{global_requirement}

【本文件信息】
文件路径：{file_path}
文件职责：{file_purpose}
本文件依赖项目内其他文件：{depend_list}
本文件对外导出接口契约(必须严格遵守：函数名、类名、参数名称、返回描述不能改动)：
{export_api_text}

硬性约束：
1.只输出完整文件代码，不要输出解释、思考、前置说明。
2.严格遵守上面对外导出接口契约，类、函数、参数签名不允许私自修改。
3.导入语句补齐，依赖的内部模块按照项目相对路径正确import。
4.不要输出markdown解释，不要输出```python ```外层标记，直接输出源码。
5.配置文件、.env.example这类非代码文件，输出对应文本内容。
"""
################################################################################################################
# 节点1：独立输入节点
def user_input_node(state: VibeProjectState) -> dict:
    """专门负责控制台读取用户输入，纯交互节点，不调用LLM"""
    if state["user_raw_requirement"] is None:
        user_text = input("请输入你的需求：")
        return {
            "user_raw_requirement": user_text
        }
    else:
        questions = state.get("clarify_questions") or []
        for q in questions:
            print(q)
        user_text = input("请按照提示补充信息，无需补充就回复无需补充：")
        answers = state.get("user_clarify_answers") or []
        answers.append(user_text)
        return {
            "user_clarify_answers": answers
        }

# 节点2：需求澄清解析
def parse_user_requirement_node(state: VibeProjectState) -> dict:
    """仅做LLM需求澄清，无input交互"""
    user_raw = state["user_raw_requirement"]
    answer_list = state.get("user_clarify_answers") or []
    user_clarify_text = "\n".join(answer_list)

    final_prompt = fill_info_prompt.format(
        USER_RAW_INPUT=user_raw,
        USER_CLARIFY_ANSWERS=user_clarify_text
    )
    response = model_ask.invoke(final_prompt)
    content = response.content
    json_content = clean_json_data(str(content))
    info_dict = json.loads(json_content)

    is_enough = info_dict.get("is_enough", False)
    final_req = info_dict.get("final_requirement", "")
    questions_out = info_dict.get("questions", [])

    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": content})

    if not is_enough:
        return {
            "messages": messages,
            "need_user_input": True,
            "info_enough": False,
            "clarify_questions": questions_out,
            "final_clear_requirement": None
        }
    else:
        return {
            "messages": messages,
            "need_user_input": False,
            "info_enough": True,
            "clarify_questions": None,
            "final_clear_requirement": final_req
        }

################################################################################################################
# 节点3：前向计划生成1
def pre_node_front_plan_one_node(state: VibeProjectState) -> dict:
    """前向计划生成1"""
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = front_plan_prompt.format(
        USER_REQUIREMENT=complete_info
    )
    resp = model_front_plan_one.invoke(real_front_plan_prompt)
    plan_content = resp.content
    return {
        "front_plan_one": plan_content
    }

# 节点4：前向计划生成2
def pre_node_front_plan_two_node(state: VibeProjectState) -> dict:
    """前向计划生成2"""
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = front_plan_prompt.format(
        USER_REQUIREMENT=complete_info
    )
    resp = model_front_plan_two.invoke(real_front_plan_prompt)
    plan_content = resp.content
    return {
        "front_plan_two": plan_content
    }
################################################################################################################
#分支节点1：是否走向前置计划
def decide_to_plan_addtional(state: VibeProjectState) -> str | list[str]:
    """是否走向前置计划"""
    if state.get("info_enough", False):
        return ["pre_node_front_plan_one_node", "pre_node_front_plan_two_node"]
    else:
        return "user_input_node_node"

################################################################################################################
# 节点5：最终计划生成
def pre_node_final_plan_node(state: VibeProjectState) -> dict:
    final_back_code_prompt = back_code_prompt.format(
        USER_REQUIREMENT=state.get("user_raw_requirement", ""),
        TEXT_PLAN_A=state.get("front_plan_one", ""),
        TEXT_PLAN_B=state.get("front_plan_two", ""),
        SAVE_PATH=save_path
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    md_save_path = os.path.join(save_path, "final_plan.md")
    resp = model_back_plan.invoke(final_back_code_prompt)
    final_plan_content = resp.content
    # 修复bug：只写入模型输出，不要把prompt写进文件
    with open(md_save_path, "w", encoding="utf‑8") as f:
        f.write(final_plan_content)
    return {
        "final_plan": final_plan_content
    }

################################################################################################################
# 节点6：最终计划输出(在指定的文件目录下先创建所有文件和文件夹)
def final_plan_output_node(state: VibeProjectState) -> dict:
    final_plan = state.get("final_plan", "")
    final_plan_json_str = clean_json_data(str(final_plan))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        schema_data = json.loads(final_plan_json_str)
    except json.JSONDecodeError as e:
        print(f"final_plan json解析失败:{e}")
        return {}

    def process_directory_tree(tree_list, root_base: str):
        for dir_item in tree_list:
            rel_dir = dir_item.get("path", "")
            abs_dir = os.path.join(root_base, rel_dir)
            os.makedirs(abs_dir, exist_ok=True)

            # 创建当前目录下所有文件
            file_list = dir_item.get("files", [])
            for file_item in file_list:
                rel_file_path = file_item.get("file_path", "")
                abs_file = os.path.join(root_base, rel_file_path)
                file_parent = os.path.dirname(abs_file)
                os.makedirs(file_parent, exist_ok=True)
                # 创建空文件
                with open(abs_file, "w", encoding="utf‑8") as fp:
                    pass

            # 递归处理子目录 sub_dirs
            sub_dirs = dir_item.get("sub_dirs", [])
            if sub_dirs:
                process_directory_tree(sub_dirs, root_base)

    directory_tree = schema_data.get("directory_tree", [])
    process_directory_tree(directory_tree, save_path)
    return {}

################################################################################################################
# 节点7：代码生成(在指定的文件目录下的所有文件中写入代码)
def coding_generate_node(state: VibeProjectState) -> dict:
    """
    根据schema逐个生成文件源码，严格遵守export_apis接口契约
    单个文件生成失败仅打印错误日志，继续处理剩余文件
    """
    final_plan = state.get("final_plan", "")
    final_plan_json_str = clean_json_data(str(final_plan))

    try:
        schema_data = json.loads(final_plan_json_str)
    except json.JSONDecodeError as e:
        print(f"[coding_generate_node] schema JSON解析失败: {e}")
        return {}

    # 递归提取目录树中全部file对象
    def collect_all_files(tree_list):
        file_collect = []
        for dir_item in tree_list:
            files = dir_item.get("files", [])
            file_collect.extend(files)
            sub_dirs = dir_item.get("sub_dirs", [])
            file_collect.extend(collect_all_files(sub_dirs))
        return file_collect

    all_file_items = collect_all_files(schema_data.get("directory_tree", []))
    user_full_requirement = state.get("final_clear_requirement", "")

    error_log = []
    for file_item in all_file_items:
        f_path = file_item.get("file_path", "")
        f_purpose = file_item.get("file_purpose", "")
        depend_list = file_item.get("depend_on_files", [])
        export_apis = file_item.get("export_apis", [])

        abs_target_file = os.path.join(save_path, f_path)
        abs_target_file = os.path.abspath(abs_target_file)
        # 路径防护，禁止跑出项目根目录
        if not abs_target_file.startswith(os.path.abspath(save_path)):
            err_msg = f"路径越界拦截：{f_path}"
            print(err_msg)
            error_log.append(err_msg)
            continue

        if export_apis:
            export_api_text = json.dumps(export_apis, ensure_ascii=False, indent=2)
        else:
            export_api_text = "无对外导出接口"

        prompt_content = single_file_generate_prompt.format(
            global_requirement=user_full_requirement,
            file_path=f_path,
            file_purpose=f_purpose,
            depend_list=str(depend_list),
            export_api_text=export_api_text
        )
        try:
            resp = model_code_generate.invoke(prompt_content)
            raw_code = resp.content
            code_content = raw_code.strip()
            # 剥离代码块标记
            if code_content.startswith("```python"):
                code_content = code_content.removeprefix("```python").removesuffix("```")
            elif code_content.startswith("```"):
                code_content = code_content.removeprefix("```").removesuffix("```")
            code_content = code_content.strip()

            with open(abs_target_file, "w", encoding="utf-8") as fw:
                fw.write(code_content)
            print(f"[OK] 生成文件：{f_path}")
        except Exception as exc:
            err_msg = f"[FAIL] {f_path} 生成异常：{str(exc)}"
            print(err_msg)
            error_log.append(err_msg)
    if error_log:
        print("\n=====部分文件生成失败汇总=====")
        for log in error_log:
            print(log)
    else:
        print("\n✅全部文件生成完成")
    return {}

################################################################################################################

def build_graph():
    builder = StateGraph(VibeProjectState)

    # add_node(图内节点ID, 真实函数对象，不要加引号)
    builder.add_node("user_input_node", user_input_node)
    builder.add_node("parse_user_requirement_node", parse_user_requirement_node)
    builder.add_node("pre_node_front_plan_one_node", pre_node_front_plan_one_node)
    builder.add_node("pre_node_front_plan_two_node", pre_node_front_plan_two_node)
    builder.add_node("pre_node_final_plan_node", pre_node_final_plan_node)
    builder.add_node("final_plan_output_node", final_plan_output_node)
    builder.add_node("coding_generate_node", coding_generate_node)

    # 入口点
    builder.add_edge(START, "user_input_node")

    # 用户输入完成 → 进入需求澄清解析
    builder.add_edge("user_input_node", "parse_user_requirement_node")

    # 条件路由：解析完成后执行 decide_to_plan_addtional 路由函数
    builder.add_conditional_edges(
        source="parse_user_requirement_node",
        path=decide_to_plan_addtional,
    )

    # 两个并行节点全部完成之后，汇合到合并架构节点
    builder.add_edge("pre_node_front_plan_one_node", "pre_node_final_plan_node")
    builder.add_edge("pre_node_front_plan_two_node", "pre_node_final_plan_node")

    # 串行后续流程
    builder.add_edge("pre_node_final_plan_node", "final_plan_output_node")
    builder.add_edge("final_plan_output_node", "coding_generate_node")
    builder.add_edge("coding_generate_node", END)

    graph = builder.compile()
    return graph

if __name__ == '__main__':
    graph = build_graph()




























