import json
import os
import re
from typing import Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import yaml
from langgraph.types import interrupt

load_dotenv()

config_path = "../../config_file/my_config.yaml"

# ========== 健壮性：配置读取增加异常捕获 ==========
try:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise RuntimeError(f"读取配置文件失败，请检查路径: {config_path}, 错误: {e}") from e

try:
    absolute_path = config["absolute_save_path"]
except:
    absolute_path = os.path.abspath(__file__)
# 修复：不要把脚本文件名混入项目根目录
script_dir = os.path.dirname(absolute_path)
save_path = os.path.abspath(os.path.join(script_dir, "./new_project"))

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

# ========== 健壮性：API密钥校验 ==========
def _check_api_key(provider_name, key):
    if not key:
        raise RuntimeError(f"[{provider_name}] API密钥未配置，请检查.env文件")

_check_api_key("千问", api_key_ask)
_check_api_key("千问", api_key_code_front_plan_one)
_check_api_key("DeepSeek", api_key_code_front_plan_two)
_check_api_key("DeepSeek", api_key_code_back)
_check_api_key("DeepSeek", api_key_code_generate)

# 信息补全模型实例
model_ask = ChatOpenAI(
    model=model_name_ask,
    api_key=api_key_ask,
    base_url=config["ask_model"]["base_url"],
    temperature=0.1
)

# 前向计划模型1实例
model_front_plan_one = ChatOpenAI(
    model=model_name_code_front_plan_one,
    api_key=api_key_code_front_plan_one,
    base_url=config["front_plan_model_one"]["base_url"],
    temperature=0.1
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
# ✅修复：user_clarify_answers增加operator.add归约器，实现列表自动追加，节点不用透传旧列表
class VibeProjectState(TypedDict):
    messages: Annotated[list[dict], add_messages]  # 项目历史消息记录
    user_raw_requirement: str | None  # 用户最原始输入
    user_clarify_answers: Annotated[list[str], operator.add]  # ✅归约器，列表自动拼接
    final_clear_requirement: str | None  # 经过补全之后、完备的最终需求（下游所有模型都使用这个，不再使用原始输入）
    need_user_input: bool  # True=需要暂停，等待用户补充输入
    clarify_questions: list[str] | None  # 需要向用户提出的问题列表
    info_enough: bool  # True=信息足够，无需继续提问
    front_plan_one: str | None  # 前向计划模型1输出
    front_plan_two: str | None  # 前向计划模型2输出
    final_plan: str | None  # 最终计划模型输出
    remaining_steps: int  # 新增：剩余澄清循环次数，耗尽强制进入架构生成
    config: dict | None  # 子图配置，用于创建项目子图时使用

################################################################################################################
# ✅重点修复：fill_info_prompt，所有示例JSON大括号全部改为{{ }}，避免format解析冲突
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
{{
  "is_enough": true/false,
  "final_requirement": "整理合并后的完整需求文本",
  "questions": ["问题1","问题2"]
}}

示例1 用户要求直接开始
{{
  "is_enough": true,
  "final_requirement":"用户需要开发xxx，用户要求直接开始，部分细节由架构阶段推断补齐。",
  "questions":[]
}}

示例2 需要补充信息
{{
  "is_enough": false,
  "final_requirement":"",
  "questions":["是否需要数据库持久化？","项目是否需要对外web接口？"]
}}

示例3 信息充足
{{
  "is_enough": true,
  "final_requirement":"开发基于LangGraph的Agent，Python，本地脚本运行，接入deepseek，不需要数据库，编写单元测试。",
  "questions":[]
}}

====输入内容====
原始用户需求：{USER_RAW_INPUT}
历史补充回答：{USER_CLARIFY_ANSWERS}
"""
################################################################################################################
# 前置工程化方案提示词【更新：增加对外接口、调用外部接口强制约束】
front_plan_prompt = """
你是专业软件架构师，请根据用户项目需求输出完整工程架构方案。
所有设计必须严格围绕用户原始需求展开，禁止超出需求范围过度设计，禁止遗漏核心需求。

【前置步骤：需求锚定（必须执行）】
1. 先逐条拆解用户原始需求中的核心功能点、约束点，形成清晰的需求点清单。
2. 后续所有设计必须对应需求点，每个需求点都要有明确的模块/文件实现，禁止出现需求无对应设计的情况。

硬性约束：
1. 输出仅使用Markdown，禁止输出JSON，禁止使用```代码块标记。
2. 不要输出思考过程，直接输出方案正文。
3. 必须包含下面6个固定标题，标题文字不可修改、不可缺失。
4. 每个文件写明相对路径，写清职责，写明依赖的项目内其他文件。
5. 需要包含配置文件、入口脚本、测试目录、开发依赖、日志处理、错误处理，不要遗漏工程组件。
6.【最重要强制规则1：接口契约精细化】在文件清单的每一个文件描述中，明确列出**本文件对外暴露、可供其他文件调用的全部类、函数**；
   - 必须写明：函数/类名称、入参名称+参数类型、参数含义、返回值类型+返回值说明、抛出的异常类型。
   - 如果该文件没有对外暴露接口，写明：无对外导出接口。
   - 仅写对外公开接口，内部私有函数不需要罗列。
   - 这部分信息是后续代码生成的契约，绝对不能省略、不能写错函数名与参数，禁止模糊表述。
7.【最重要强制规则2：调用关系精确化】每个文件必须增加【本文件调用外部接口】，预先写明本文件将会调用哪些其他文件的哪些公开接口。
   - 格式：依赖文件路径 + 调用的类/函数名称，参数完全复用对方对外接口定义。
   - 如果本文件不调用任何外部文件，写：本文件不调用任何外部接口。
   - 架构阶段就定死调用关系，后续代码生成阶段不允许新增、修改外部调用。
8.【最重要强制规则3：职责边界刚性约束】本文件自身实现的函数、类，必须和对外导出接口列表完全对齐：不能多定义函数，不能少定义函数，禁止私自新增不在对外接口列表的公开函数。私有内部函数不受此约束。
   - 严格禁止跨文件写业务逻辑，每个文件职责单一，边界清晰。

【必须包含的6个固定标题】
## 项目概述
## 项目依赖
## 架构整体说明
## 目录结构
## 文件清单
## 需求覆盖自查

参考输出样例：
## 项目概述
实现一个简易LangGraph Agent，具备任务拆解与工具调用，提供本地脚本入口。

## 项目依赖
- 运行依赖(pip): langgraph, langchain-openai, python-dotenv
- 开发依赖(dev): pytest, black

## 架构整体说明
源码放置src目录；graph相关逻辑独立子模块；tests存放单元测试，业务代码与测试代码分离。
整体数据流向：入口脚本接收输入 → 调用graph节点处理 → 工具函数执行 → 返回结果。

## 目录结构
- src/
  - graph/
  - utils/
- tests/
- .env.example

## 文件清单
- 文件路径：src/main.py
  职责：程序入口，组装并运行langgraph图，读取环境配置，统一异常捕获与日志
  文件依赖：src/graph/state.py，src/graph/nodes.py
  对外导出接口：
  函数 run_agent_task(user_input: str) -> dict
    参数 user_input：str类型，用户输入任务字符串
    返回值：dict类型，agent执行完成后的结果字典，包含code、msg、data三个字段
    异常：无对外抛出异常，内部统一捕获
  本文件调用外部接口：
  调用 src/graph/nodes.py : parse_task_node(state)
  调用 src/graph/state.py : AgentState

- 文件路径：src/graph/state.py
  职责：定义Agent状态TypedDict
  文件依赖：无
  对外导出接口：
  类 AgentState(TypedDict)
    字段 user_query: str 用户原始输入
    字段 result: str 任务输出结果
  本文件调用外部接口：
  本文件不调用任何外部接口

- 文件路径：src/graph/nodes.py
  职责：实现各个业务节点逻辑
  文件依赖：src/graph/state.py
  对外导出接口：
  函数 parse_task_node(state: AgentState) -> AgentState
    参数 state：AgentState类型，Agent状态对象
    返回值：AgentState类型，更新后的Agent状态
    异常：参数校验失败抛出ValueError
  本文件调用外部接口：
  调用 src/graph/state.py : AgentState

- 文件路径：src/utils/helper.py
  职责：通用工具函数，字符串处理，prompt组装
  文件依赖：无
  对外导出接口：
  函数 build_prompt(template: str, params: dict) -> str
    参数 template：str类型，prompt模板字符串；params：dict类型，模板参数字典
    返回值：str类型，填充完成的prompt文本
    异常：模板缺失占位符抛出KeyError
  本文件调用外部接口：
  本文件不调用任何外部接口

- 文件路径：tests/test_graph.py
  职责：graph节点单元测试
  文件依赖：src/graph/nodes.py
  对外导出接口：无对外导出接口
  本文件调用外部接口：
  调用 src/graph/nodes.py : parse_task_node

- 文件路径：.env.example
  职责：环境变量模板，存放密钥配置
  文件依赖：无
  对外导出接口：无对外导出接口
  本文件调用外部接口：
  本文件不调用任何外部接口

## 需求覆盖自查
1. ✅ LangGraph Agent核心能力：对应graph模块实现
2. ✅ 任务拆解功能：对应parse_task_node节点实现
3. ✅ 工具调用能力：对应节点内部逻辑实现
4. ✅ 本地脚本入口：对应main.py入口函数

====基于下面用户需求输出方案====
{USER_REQUIREMENT}
"""

################################################################################################################
# 【优化后】后置合并评审提示词 back_code_prompt 新增call_other_apis字段
back_code_prompt = """
你是高级架构评审专家。接收两份Markdown架构方案A、方案B，以及用户原始项目需求。
你的核心目标：输出一份100%贴合用户需求、接口契约完整无歧义、调用关系闭环可执行的标准schema，作为后续代码生成的唯一依据。

执行步骤：
1.【需求一致性校验（优先级最高）】
   ① 逐条拆解用户原始需求，形成需求点清单。
   ② 分别校验方案A、方案B对需求点的覆盖度，以覆盖度更高的方案为合并基底。
   ③ 未覆盖的需求点，必须从另一份方案中补充，或自行基于需求补全设计，禁止最终schema遗漏任何用户需求。

2.【信息完整提取】
   充分理解用户原始需求，通读方案A、方案B全部内容。
   重点提取：
   ①每个文件的【对外导出接口】（类、函数、入参名+类型、含义、返回值类型+说明、异常），这是后续代码生成的核心契约，不可丢失、不可模糊。
   ②每个文件【本文件调用外部接口】调用关系契约，完整记录哪个文件调用其他文件的哪些接口，后续代码生成阶段禁止私自新增调用。

3.【对比与择优融合】
   对比两份方案：目录树结构、模块划分、pip依赖、开发依赖、每个文件职责、文件之间依赖关系、每个文件对外导出接口契约、调用外部接口契约。
   择优融合规则（优先级从高到低）：
   ① 需求覆盖度：优先保留对用户需求覆盖更完整的设计。
   ② 接口完整性：优先保留接口定义更精确、参数类型更清晰的设计。
   ③ 结构合理性：优先保留模块边界清晰、职责单一、无循环依赖的设计。
   ④ 实现成本：优先保留结构简洁、易于代码生成的设计。
   - 保留两份方案的合理设计，补齐缺失模块；删除重复目录、重复文件、循环依赖、无效依赖包；
   - 对冲突的对外接口，选择更贴合业务需求的一份，**不允许随意修改函数名、参数定义**。
   - 对冲突的外部调用关系，保留合理调用，禁止凭空新增调用关系。

4.【接口闭环校验（强制执行）】
   遍历所有文件的call_other_apis：
   - 每一条调用记录，必须能在目标文件的export_apis中找到名称、参数完全匹配的接口。
   - 找不到匹配接口的，要么修正调用关系，要么在目标文件补全对应接口，绝对不允许出现“调用不存在的接口”的情况。
   - 禁止循环依赖：A调用B、B调用A的情况必须修正，调整为单向依赖。

5.【schema填充与自检】
   输出严格完全匹配 vibe_project_schema_v1 的JSON对象，不要输出任何其他内容。
   输出前必须自检：
   ① 字段完整性：所有schema定义的必填字段都存在，无缺失、无多余自定义字段。
   ② 路径合法性：所有文件、目录路径都是相对根目录的合法路径，无越界、无绝对路径。
   ③ 接口闭环：所有调用关系都有对应导出接口，无循环依赖。
   ④ 需求覆盖：所有用户需求点都有对应文件实现。
   ⑤ 类型一致性：同一个接口在导出和调用处的参数、返回值描述完全一致。

#强制硬性规则，必须严格遵守
1.仅输出纯净JSON字符串，不允许任何额外字符，禁止思考、说明、解释、小结。
2.禁止markdown标记，禁止```json / ```代码块包裹。
3.只允许使用schema内定义字段，**禁止新增自定义字段，禁止删除schema必填字段**；
本次扩展每个file对象增加：
- export_apis字段：存放本文件对外公开接口契约
- call_other_apis字段：数组，记录本文件需要调用的外部文件接口，每一项包含 target_file（被调用文件路径）、api_name（调用的接口名称）
4.目录与文件路径全部相对于根目录：{SAVE_PATH}，schema内填写相对路径，不要填写绝对路径。
5.从两份方案提取信息填充schema：项目名称、项目描述、入口文件、pip运行依赖列表、dev_dependencies开发依赖列表、directory_tree目录树。
6.每个file对象规则：
  - file_path填写文件相对路径
  - file_purpose写清文件职责
  - depend_on_files填写本文件依赖的项目内部其他文件路径
  - is_test_file：测试文件填true，业务/配置文件填false
  - template_content固定为空字符串""
  - export_apis：数组，存放本文件对外暴露的类、函数接口契约；无对外接口则为空数组[]；严禁丢失两份方案中已经定义好的接口签名。
    每个元素必须包含：
    - name：接口名称
    - type：函数/类
    - params：数组，每个元素包含param_name（参数名）、param_type（参数类型）、desc（参数含义）
    - return_desc：返回值说明，必须包含返回值类型
    - exception：抛出的异常类型，无异常则写“无”
  - call_other_apis：数组，本文件架构阶段确定的外部调用关系；无外部调用则为空数组[]；禁止凭空编造调用。
7.递归处理子目录sub_dirs，完整还原嵌套目录结构，不要扁平化丢失层级。
8.依赖校验：depend_on_files内的路径必须是schema中真实存在的文件，禁止引用不存在的文件，杜绝循环依赖。
9.接口数量约束：本文件生成的公开函数/类，必须和export_apis数量一一对应，不能多、不能少。

# vibe_project_schema_v1
{{
  "$schema": "vibe_project_schema_v1",
  "project_name": "项目名称",
  "project_desc": "项目描述",
  "entry_file": "src/main.py",
  "dependencies": {{
    "pip": [],
    "dev_dependencies": []
  }},
  "directory_tree": [
    {{
      "path": "src/api",
      "desc": "文件夹说明",
      "files": [
        {{
          "file_path": "src/api/route.py",
          "file_purpose": "文件功能描述",
          "depend_on_files": [],
          "is_test_file": false,
          "template_content": "",
          "export_apis": [
            {{
              "name": "example_func",
              "type": "函数",
              "params": [{{"param_name":"arg1","param_type":"str","desc":"参数说明"}}],
              "return_desc": "str类型，返回值描述",
              "exception": "无"
            }}
          ],
          "call_other_apis":[
            {{
                "target_file":"src/utils/common.py",
                "api_name":"helper_func"
            }}
          ]
        }}
      ],
      "sub_dirs": []
    }}
  ]
}}

用户原始需求：
{USER_REQUIREMENT}

方案A：
{TEXT_PLAN_A}

方案B：
{TEXT_PLAN_B}

文件保存根目录(所有文件文件夹相对该目录)：
{SAVE_PATH}
"""

################################################################################################################
# 单文件源码生成提示词【更新：类实例校验、契约强制约束】
single_file_generate_prompt = """
你是Python高级开发工程师，**严格按照架构契约生成零偏差、可直接运行的Python源码**。
⚠️ 所有接口签名、调用关系、文件职责完全由schema契约锁定，禁止任何自由发挥、禁止新增功能、禁止修改契约定义。
生成的代码必须语法正确、导入正确、调用匹配、无逻辑遗漏，可直接运行不报错。

【全局项目需求】
{global_requirement}

【本文件信息】
文件路径：{file_path}
文件职责：{file_purpose}
本文件依赖项目内其他文件：{depend_list}

【本文件对外导出接口契约（最高优先级，必须100%遵守）】
{export_api_text}
✅ 强制遵守：
1. 公开接口数量必须和列表完全一致，不能多、不能少；名称必须完全匹配，不得改名。
2. 每个接口的**函数名/类名、参数名、参数数量、参数类型、返回值类型、抛出异常**，必须和契约定义一字不差。
3. 所有公开接口必须添加和契约一致的类型注解（参数+返回值），禁止无类型定义。
4. 如果是类，必须明确定义__init__构造函数，入参和契约完全匹配；实例方法必须带self参数。
5. 私有内部函数不受限制，但禁止和公开接口重名。

【架构预先约定：本文件允许调用的外部接口列表】
{call_other_api_text}
✅ 强制遵守：
1. 只能调用此列表内的接口，禁止调用任何未约定的接口、禁止私自新增调用关系。
2. 调用时参数名必须和对方契约定义完全一致，优先使用关键字参数传参，避免位置参数错误。

【所有依赖文件完整对外公开接口总览（只能使用下面出现的接口）】
{depend_all_export_text}

===== 外部接口调用强制规则（重中之重，绝对不能违反）=====
1. 模块级函数调用：先import对应模块 → 模块名.函数名(参数名=值)
2. 类接口调用：先import对应模块 → 实例化类（传入构造参数）→ 实例.方法名(参数名=值)
3. ❌ 严禁直接通过模块名调用类的实例方法（必须先实例化）
4. ❌ 严禁给接口改名、加参数、改参数类型、改返回值类型
5. 导入规则：
   - 项目内部模块使用相对导入，路径和依赖文件路径严格对应
   - 只导入需要使用的类/函数，禁止通配符导入
   - 禁止循环导入，禁止导入不存在的模块

===== 工程化编码强制规范 =====
1. 所有公开接口必须编写完整文档字符串（docstring），说明功能、参数含义、返回值、抛出异常。
2. 入参必须做基础合法性校验（空值、类型、范围），非法输入抛出对应异常。
3. 外部依赖调用必须做异常捕获，禁止裸调用；关键操作增加日志输出。
4. 禁止硬编码绝对路径、硬编码密钥、硬编码配置项，配置从外部读取。
5. 代码职责单一，不超出本文件职责范围，不跨文件写业务逻辑。
6. 配置文件、.env.example、README等非Python代码文件，按契约格式输出纯文本内容。

===== 生成前强制校验清单（必须逐一核对再生成）=====
1. 本文件所有公开接口的名称、参数、返回值，我都明确了吗？
2. 我要调用的外部接口，都在允许调用列表里吗？
3. 依赖的类接口，我会先实例化再调用方法吗？
4. 导入路径和依赖文件路径能对应上吗？
5. 我没有新增任何契约外的公开接口吧？

===== 生成后强制自检清单（生成后逐一验证）=====
1. 公开接口数量和契约一致，没有多写、少写。
2. 所有接口名称、参数名、参数类型、返回值和契约完全匹配。
3. 所有外部调用的接口，都存在于依赖文件的导出列表中。
4. 所有导入路径正确，没有拼写错误，没有循环导入。
5. 类的实例方法都带self，构造函数参数正确。
6. 没有硬编码绝对路径、密钥等敏感信息。
7. 代码没有语法错误，可直接运行。

# 参考示例
【依赖文件 storage.py 的对外契约】
类 MemoStorage(data_file: str)
函数：add_memo(self, content: str) -> bool
函数：get_all_memos(self) -> list[str]
函数：delete_memo(self, memo_id: int) -> bool

✅【正确调用示例】
import storage
store = storage.MemoStorage(data_file="memos.json")
store.add_memo(content="测试内容")
memos = store.get_all_memos()
ok = store.delete_memo(memo_id=1)

❌【错误调用示例，严禁这样写】
import storage
storage.add_memo(content="测试内容")   # 错误！add_memo是类实例方法，不是模块顶层函数
storage.get_all_memos()               # 错误！没有实例化直接调用类方法

硬性约束：
1. 只输出完整文件代码，不要输出解释、思考、前置说明。
2. 不要输出markdown解释，不要输出```python ```外层标记，直接输出源码。
"""

################################################################################################################
# 节点1：独立输入节点
def user_input_node(state: VibeProjectState) -> dict:
    """专门负责控制台读取用户输入，纯交互节点，不调用LLM"""
    if state["user_raw_requirement"] is None:
        user_text = interrupt({
            "question": "请输入你的需求：",
        })
        return {
            "user_raw_requirement": user_text
        }
    elif state["user_raw_requirement"] is not None and not state.get("clarify_questions"):
        return {"user_raw_requirement": state["user_raw_requirement"]}
    else:
        questions = state.get("clarify_questions") or []
        for q in questions:
            print(q)
        question_text = "\n".join(f"- {q}" for q in questions)
        user_text = interrupt({
            "question": f"请按照提示补充信息，无需补充就回复无需补充：\n{question_text}",
        })
        # ✅operator.add归约器，只返回【新增单元素列表】，框架自动拼接旧列表，不要手动append
        return {
            "user_clarify_answers": [user_text]
        }

################################################################################################################
# 节点2：需求澄清解析（询问用户是否需要补充信息）
def parse_user_requirement_node(state: VibeProjectState) -> dict:
    """仅做LLM需求澄清，无input交互"""
    print("开始向模型请求需求澄清结果")
    user_raw = state["user_raw_requirement"]
    answer_list = state.get("user_clarify_answers") or []
    user_clarify_text = "\n".join(answer_list)

    final_prompt = fill_info_prompt.format(
        USER_RAW_INPUT=user_raw,
        USER_CLARIFY_ANSWERS=user_clarify_text
    )

    # ========== 健壮性：模型调用异常兜底 ==========
    try:
        response = model_ask.invoke(final_prompt)
        content = response.content
    except Exception as e:
        print(f"[WARN] 需求澄清模型调用失败，直接使用原始需求；错误: {e}")
        return {
            "info_enough": True,
            "final_clear_requirement": user_raw,
            "clarify_questions": None
        }

    json_content = clean_json_data(str(content))

    # ========== 健壮性：JSON解析失败兜底 ==========
    try:
        info_dict = json.loads(json_content)
    except Exception as e:
        print(f"[WARN] 澄清结果JSON解析失败，直接使用原始需求；错误: {e}")
        return {
            "info_enough": True,
            "final_clear_requirement": user_raw,
            "clarify_questions": None
        }

    is_enough = info_dict.get("is_enough", False)
    final_req = info_dict.get("final_requirement", "")
    questions_out = info_dict.get("questions", [])

    # ✅修复：add_messages归约器，禁止手动messages.append，只返回新增消息
    new_assistant_msg = {"role": "assistant", "content": content}

    if not is_enough:
        # ========== 新增：剩余澄清次数扣减 ==========
        remain = state.get("remaining_steps", 0) - 1
        return {
            "messages": [new_assistant_msg],
            "need_user_input": True,
            "info_enough": False,
            "clarify_questions": questions_out,
            "final_clear_requirement": None,
            "remaining_steps": remain
            # ✅不再需要透传user_clarify_answers，operator.add归约器自动保留旧值
        }
    else:
        return {
            "messages": [new_assistant_msg],
            "need_user_input": False,
            "info_enough": True,
            "clarify_questions": None,
            "final_clear_requirement": final_req
        }

################################################################################################################
# 节点3：前向计划生成1
def pre_node_front_plan_one_node(state: VibeProjectState) -> dict:
    """前向计划生成1"""
    print("前向计划1开始生成")
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = front_plan_prompt.format(
        USER_REQUIREMENT=complete_info
    )
    # ========== 健壮性：模型调用异常兜底 ==========
    try:
        resp = model_front_plan_one.invoke(real_front_plan_prompt)
        plan_content = resp.content
    except Exception as e:
        print(f"[ERR] 方案1生成失败；错误: {e}")
        plan_content = ""
    return {
        "front_plan_one": plan_content
    }

# 节点4：前向计划生成2
def pre_node_front_plan_two_node(state: VibeProjectState) -> dict:
    """前向计划生成2"""
    print("前向计划2开始生成")
    complete_info = ("用户原始需求：\n" + (state.get("user_raw_requirement", "")) + "\n" +
                     "最终需求：\n" + (state.get("final_clear_requirement", "")))

    real_front_plan_prompt = front_plan_prompt.format(
        USER_REQUIREMENT=complete_info
    )
    # ========== 健壮性：模型调用异常兜底 ==========
    try:
        resp = model_front_plan_two.invoke(real_front_plan_prompt)
        plan_content = resp.content
    except Exception as e:
        print(f"[ERR] 方案2生成失败；错误: {e}")
        plan_content = ""
    return {
        "front_plan_two": plan_content
    }
################################################################################################################
#分支节点1：是否走向前置计划
def decide_to_plan_addtional(state: VibeProjectState) -> str | list[str]:
    """是否走向前置计划；新增次数限制，耗尽强制进入架构生成"""
    info_enough = state.get("info_enough", False)
    remain = state.get("remaining_steps", 0)

    if info_enough or remain <= 0:
        if remain <= 0 and not info_enough:
            print(f"⚠️ 澄清循环次数已耗尽，强制进入架构生成阶段，剩余次数={remain}")
        return ["pre_node_front_plan_one_node", "pre_node_front_plan_two_node"]
    else:
        return "user_input_node"

################################################################################################################
# 节点5：最终计划生成
def pre_node_final_plan_node(state: VibeProjectState) -> dict:
    print("最终计划开始生成")
    final_back_code_prompt = back_code_prompt.format(
        USER_REQUIREMENT=state.get("user_raw_requirement", ""),
        TEXT_PLAN_A=state.get("front_plan_one", ""),
        TEXT_PLAN_B=state.get("front_plan_two", ""),
        SAVE_PATH=save_path
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    md_save_path = os.path.join(save_path, "final_plan.md")

    # ========== 健壮性：模型调用异常兜底 ==========
    try:
        resp = model_back_plan.invoke(final_back_code_prompt)
        final_plan_content = resp.content
    except Exception as e:
        print(f"[ERR] 架构合并模型调用失败；错误: {e}")
        final_plan_content = ""

    # 修复bug：只写入模型输出，不要把prompt写进文件
    # ========== 健壮性：文件写入异常捕获 ==========
    try:
        with open(md_save_path, "w", encoding="utf-8") as f:
            f.write(final_plan_content)
    except Exception as e:
        print(f"[WARN] 写入final_plan.md失败；错误: {e}")

    return {
        "final_plan": final_plan_content
    }

################################################################################################################
# 节点6：最终计划输出(在指定的文件目录下先创建所有文件和文件夹)
def final_plan_output_node(state: VibeProjectState) -> dict:
    print("在本地生成文件")
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
            # ========== 健壮性：路径越界防护 ==========
            abs_dir = os.path.abspath(abs_dir)
            if not abs_dir.startswith(os.path.abspath(root_base)):
                print(f"[WARN] 目录路径越界，已跳过: {rel_dir}")
                continue
            os.makedirs(abs_dir, exist_ok=True)

            # 创建当前目录下所有文件
            file_list = dir_item.get("files", [])
            for file_item in file_list:
                rel_file_path = file_item.get("file_path", "")
                abs_file = os.path.join(root_base, rel_file_path)
                abs_file = os.path.abspath(abs_file)
                # ========== 健壮性：文件路径越界防护 ==========
                if not abs_file.startswith(os.path.abspath(root_base)):
                    print(f"[WARN] 文件路径越界，已跳过: {rel_file_path}")
                    continue
                file_parent = os.path.dirname(abs_file)
                os.makedirs(file_parent, exist_ok=True)
                # 创建空文件
                try:
                    with open(abs_file, "w", encoding="utf-8") as fp:
                        pass
                except Exception as e:
                    print(f"[WARN] 创建空文件失败 {rel_file_path}; 错误: {e}")

            # 递归处理子目录 sub_dirs
            sub_dirs = dir_item.get("sub_dirs", [])
            if sub_dirs:
                process_directory_tree(sub_dirs, root_base)

    directory_tree = schema_data.get("directory_tree", [])
    process_directory_tree(directory_tree, save_path)
    return {}

################################################################################################################
# 节点7：代码生成【完整更新，构建全局文件映射，注入依赖文件全部export_apis】
def coding_generate_node(state: VibeProjectState) -> dict:
    """
    根据schema逐个生成文件源码，严格遵守export_apis、call_other_apis接口契约
    单个文件生成失败仅打印错误日志，继续处理剩余文件
    """
    print("开始向文件中写入代码")
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

    # 构建全局文件路径 -> file对象映射表，方便快速查找依赖文件export_apis
    file_path_map = {item["file_path"]: item for item in all_file_items}

    error_log = []
    for file_item in all_file_items:
        f_path = file_item.get("file_path", "")
        f_purpose = file_item.get("file_purpose", "")
        depend_list = file_item.get("depend_on_files", [])
        export_apis = file_item.get("export_apis", [])
        call_other_apis = file_item.get("call_other_apis", [])

        abs_target_file = os.path.join(save_path, f_path)
        abs_target_file = os.path.abspath(abs_target_file)
        # 路径防护，禁止跑出项目根目录
        if not abs_target_file.startswith(os.path.abspath(save_path)):
            err_msg = f"路径越界拦截：{f_path}"
            print(err_msg)
            error_log.append(err_msg)
            continue

        # 格式化本文件对外接口
        if export_apis:
            export_api_text = json.dumps(export_apis, ensure_ascii=False, indent=2)
        else:
            export_api_text = "无对外导出接口"

        # 格式化本文件架构约定的外部调用契约
        if call_other_apis:
            call_other_api_text = json.dumps(call_other_apis, ensure_ascii=False, indent=2)
        else:
            call_other_api_text = "本文件架构约定：不允许调用任何外部接口"

        # 遍历所有依赖文件，把依赖文件完整export_apis全部格式化输出给模型看
        depend_export_lines = []
        for dep_file_path in depend_list:
            dep_file_obj = file_path_map.get(dep_file_path)
            if not dep_file_obj:
                depend_export_lines.append(f"----------依赖文件：{dep_file_path} 未在schema找到该文件----------")
                continue
            dep_export = dep_file_obj.get("export_apis", [])
            depend_export_lines.append(f"----------依赖文件：{dep_file_path} 对外公开接口----------")
            for api in dep_export:
                api_name = api.get("name","")
                api_type = api.get("type","")
                params_raw = api.get("params",[])
                param_text_list = []
                for p in params_raw:
                    p_name = p.get("param_name","")
                    param_text_list.append(p_name)
                param_str = ", ".join(param_text_list)
                depend_export_lines.append(f"{api_type} {api_name}({param_str})")
        depend_all_export_text = "\n".join(depend_export_lines)

        prompt_content = single_file_generate_prompt.format(
            global_requirement=user_full_requirement,
            file_path=f_path,
            file_purpose=f_purpose,
            depend_list=str(depend_list),
            export_api_text=export_api_text,
            call_other_api_text=call_other_api_text,
            depend_all_export_text=depend_all_export_text
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

################################################################################################################

# def build_create_graph():
#     builder = StateGraph(VibeProjectState)
#
#     # add_node(图内节点ID, 真实函数对象，不要加引号)
#     builder.add_node("user_input_node", user_input_node)
#     builder.add_node("parse_user_requirement_node", parse_user_requirement_node)
#     builder.add_node("pre_node_front_plan_one_node", pre_node_front_plan_one_node)
#     builder.add_node("pre_node_front_plan_two_node", pre_node_front_plan_two_node)
#     builder.add_node("pre_node_final_plan_node", pre_node_final_plan_node)
#     builder.add_node("final_plan_output_node", final_plan_output_node)
#     builder.add_node("coding_generate_node", coding_generate_node)
#
#     # 入口点
#     builder.add_edge(START, "user_input_node")
#
#     # 用户输入完成 → 进入需求澄清解析
#     builder.add_edge("user_input_node", "parse_user_requirement_node")
#
#     # 条件路由：解析完成后执行 decide_to_plan_addtional 路由函数
#     builder.add_conditional_edges(
#         source="parse_user_requirement_node",
#         path=decide_to_plan_addtional,
#     )
#
#     # 两个并行节点全部完成之后，汇合到合并架构节点
#     builder.add_edge("pre_node_front_plan_one_node", "pre_node_final_plan_node")
#     builder.add_edge("pre_node_front_plan_two_node", "pre_node_final_plan_node")
#
#     # 串行后续流程
#     builder.add_edge("pre_node_final_plan_node", "final_plan_output_node")
#     builder.add_edge("final_plan_output_node", "coding_generate_node")
#     builder.add_edge("coding_generate_node", END)
#
#     # ========== 新增：会话持久化 MemorySaver ==========
#     memory_checkpointer = MemorySaver()
#     graph = builder.compile(checkpointer=memory_checkpointer)
#     return graph

def build_create_graph():
    builder = StateGraph(VibeProjectState)

    builder.add_node("user_input_node", user_input_node)
    builder.add_node("parse_user_requirement_node", parse_user_requirement_node)
    builder.add_node("pre_node_front_plan_one_node", pre_node_front_plan_one_node)
    builder.add_node("pre_node_front_plan_two_node", pre_node_front_plan_two_node)
    builder.add_node("pre_node_final_plan_node", pre_node_final_plan_node)
    builder.add_node("final_plan_output_node", final_plan_output_node)
    builder.add_node("coding_generate_node", coding_generate_node)

    builder.add_edge(START, "user_input_node")
    builder.add_edge("user_input_node", "parse_user_requirement_node")

    builder.add_conditional_edges(
        source="parse_user_requirement_node",
        path=decide_to_plan_addtional,
    )

    builder.add_edge("pre_node_front_plan_one_node", "pre_node_final_plan_node")
    builder.add_edge("pre_node_front_plan_two_node", "pre_node_final_plan_node")

    builder.add_edge("pre_node_final_plan_node", "final_plan_output_node")
    builder.add_edge("final_plan_output_node", "coding_generate_node")
    builder.add_edge("coding_generate_node", END)

    # ✅重点修改：子图编译不要绑定checkpointer！交给外层主图
    graph = builder.compile()
    return graph



if __name__ == '__main__':
    graph = build_create_graph()

    # 从配置读取最大澄清循环次数
    max_clarify_steps = config.get("max_clarify_loop_steps", 3)
    # 会话线程ID，用于持久化隔离
    thread_config = {"configurable": {"thread_id": "vibe_project_main_001"}}

    init_state = {
        "messages": [],
        "user_raw_requirement": None,
        "user_clarify_answers": [], # ✅初始空列表，不能为None
        "final_clear_requirement": None,
        "need_user_input": False,
        "clarify_questions": None,
        "info_enough": False,
        "front_plan_one": None,
        "front_plan_two": None,
        "final_plan": None,
        "remaining_steps": max_clarify_steps
    }

    # 启动图，传入持久化配置
    for event in graph.stream(init_state, config=thread_config):
        print(event)