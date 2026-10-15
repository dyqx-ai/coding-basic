"""
模式判断模块 - 独立于主图之外
职责：根据用户输入和历史上下文，判断应该使用哪种模式
输入：用户消息 + 最近N条历史消息
输出：{"mode": "simple" | "demo" | "complete", "reason": "判断依据", "score": 0~100}
"""

import re
from typing import Dict, List, Optional, Callable


class ModeJudge:
    """模式判断器"""

    # ========== 各维度关键词库 ==========

    # 1. 明确模式标识（一旦命中，直接决定模式）
    EXPLICIT_KEYWORDS = {
        "complete": ["完整工程", "完整模式", "正式项目", "生产级", "full project", "complete project", "complete e-commerce", "complete system"],
        "demo": ["demo版", "演示版", "demo模式", "演示模式", "示例项目", "demo project"],
        "simple": ["极简版", "简单版", "单文件", "脚本版", "快速版", "simple project"],
    }

    # 2. 技术栈信号（表明项目复杂度）
    TECH_STACK_HIGH = [
        "flask", "django", "spring", "vue", "react", "前后端", "前端", "后端",
        "数据库", "mysql", "mongodb", "postgresql", "redis", "orm",
        "api", "restful", "rest api", "接口", "websocket", "rpc",
        "docker", "kubernetes", "k8s", "部署", "nginx",
        "spring boot", "fastapi", "graphql", "微服务", "分布式",
        "database", "postgres", "full stack", "frontend", "backend"
    ]
    TECH_STACK_MEDIUM = [
        "json", "csv", "sqlite", "配置文件", "logging",
        "存储", "保存", "读取", "加载",
        "调用", "请求", "定时", "cron", "调度", "监控",
        "查询", "获取", "抓取", "解析", "提取", "转换", "格式化",
        "打卡", "记录", "追踪", "管理",
    ]

    # 3. 功能复杂度信号
    COMPLEXITY_HIGH = [
        "用户管理", "权限", "角色", "登录", "注册", "认证", "授权",
        "借阅", "归还", "预约", "续借", "逾期", "催还",
        "报表", "统计", "分析", "看板", "dashboard", "监控",
        "多馆", "分馆", "协作", "团队", "组织",
        "流程", "审批", "工作流", "workflow",
        "支付", "订单", "交易", "账单",
        "消息", "通知", "邮件", "短信",
        "企业级", "生产级", "高并发", "多人协作",
        "课程管理", "选课", "排课", "成绩", "考试",
        "库存", "采购", "供应商", "进销存",
        "患者", "挂号", "门诊", "住院", "病历",
        "user management", "permission", "role", "login", "register",
        "report", "statistics", "dashboard", "workflow",
        "payment", "order", "inventory", "procurement"
    ]
    COMPLEXITY_MEDIUM = [
        "crud", "增删改查", "列表", "详情", "搜索", "筛选",
        "导出", "导入", "excel", "pdf",
        "缓存", "队列", "异步"
    ]

    # 4. 项目名词隐含复杂度
    PROJECT_NOUN_HIGH = ["管理系统", "管理平台", "后台系统", "ERP", "CRM", "OA"]
    PROJECT_NOUN_MEDIUM = ["服务", "工具", "助手", "模块", "组件"]
    PROJECT_NOUN_LOW = ["脚本", "demo", "测试", "示例", "sample"]

    # 5. 功能领域分组（同一领域只计一次加分）
    COMPLEXITY_GROUPS = {
        "图书管理": ["借阅", "归还", "预约", "续借", "逾期", "催还"],
        "用户管理": ["用户管理", "权限", "角色", "登录", "注册", "认证", "授权"],
        "报表": ["报表", "统计", "分析", "看板", "dashboard"],
        "库存": ["库存", "采购", "供应商", "进销存"],
        "教育": ["课程", "选课", "排课", "成绩", "考试"],
        "医疗": ["患者", "挂号", "门诊", "住院", "病历"],
        "流程": ["流程", "审批", "工作流", "workflow"],
        "交易": ["支付", "订单", "交易", "账单"],
    }

    # 6. 一击必杀信号（直接锁定 complete）
    KILLER_SIGNALS = [
        "企业级", "生产级", "高并发", "分布式", "微服务",
        "前后端分离", "full stack",
        "多人协作", "权限系统", "工作流",
        "多门店", "多租户", "连锁"
    ]

    # 7. 否定词（用于降级）
    NEGATION_PATTERNS = [
        r"不要\s*前后端分离",
        r"别\s*前后端分离",
        r"不用\s*前后端分离",
        r"取消\s*前后端分离",
        r"不需要\s*前后端分离",
        r"别整.*复杂",
        r"不要.*完整",
        r"不用.*数据库",
        r"不需要.*数据库",
        r"不要.*前后端",
    ]

    # ========== 评分权重 ==========
    SCORE_WEIGHTS = {
        "tech_high": 4,
        "tech_medium": 2,
        "complexity_high": 4,
        "complexity_medium": 2,
        "noun_high": 3,
        "noun_medium": 2,
        "file_hint": 2,
        "length_bonus": 2,
        "emphasis_bonus": 2,
        "history_bonus": 2,
        "group_bonus": 2,
    }

    # ========== 模式阈值 ==========
    THRESHOLDS = {
        "complete_min": 10,
        "demo_min": 5,
        "simple_min": 0,
    }

    # ========== 模糊区间 ==========
    FUZZY_LOW = 5
    FUZZY_HIGH = 9

    # ========== 保底加分 ==========
    FALLBACK_BONUS = 2

    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback
        self._last_decision = None

    def route(
        self,
        user_input: str,
        history: Optional[List[str]] = None,
        llm_callback: Optional[Callable[[str], str]] = None
    ) -> Dict:
        """
        主入口：判断模式
        :param user_input: 当前用户输入
        :param history: 历史消息列表（可选）
        :param llm_callback: 大模型兜底函数
        :return: {"mode": "simple"|"demo"|"complete", "reason": str, "score": int, "used_llm": bool}
        """
        if history is None:
            history = []

        text = user_input

        # 第一步：检测明确模式关键词
        explicit_mode = self._check_explicit_keywords(text)
        if explicit_mode:
            return {
                "mode": explicit_mode,
                "reason": f"用户明确指定了'{explicit_mode}'模式",
                "score": 100,
                "used_llm": False
            }

        # 第二步：检测否定词包裹的强信号（不触发一击必杀）
        # 注意：_check_killer_signals 内部已做否定词检查
        killer_mode = self._check_killer_signals(text)
        if killer_mode:
            return {
                "mode": killer_mode,
                "reason": f"检测到强力信号，直接锁定 '{killer_mode}' 模式",
                "score": 95,
                "used_llm": False
            }

        # 第三步：多维度综合评分
        score = self._calculate_score(text, history)
        mode = self._score_to_mode(score)

        # 第四步：模糊区间 LLM 兜底
        if self._in_fuzzy_zone(score) and self.use_llm_fallback and llm_callback:
            llm_mode = self._llm_fallback(text, history, llm_callback)
            if llm_mode:
                return {
                    "mode": llm_mode,
                    "reason": f"规则评分模糊({score}分)，由LLM兜底判定",
                    "score": score,
                    "used_llm": True
                }

        return {
            "mode": mode,
            "reason": f"综合评分{score}分，落在{self._score_range_name(score)}区间",
            "score": score,
            "used_llm": False
        }

    # ========== 核心评分逻辑 ==========

    def _check_explicit_keywords(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        for mode, keywords in self.EXPLICIT_KEYWORDS.items():
            for kw in keywords:
                if kw in text or kw.lower() in text_lower:
                    return mode
        return None

    def _has_negation(self, text: str) -> bool:
        """检测否定词"""
        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _check_killer_signals(self, text: str) -> Optional[str]:
        """检测一击必杀信号，先检查否定词"""
        # 如果有否定词，不触发一击必杀
        if self._has_negation(text):
            return None
        text_lower = text.lower()
        for signal in self.KILLER_SIGNALS:
            if signal in text or signal.lower() in text_lower:
                return "complete"
        return None

    def _calculate_score(self, text: str, history: List[str]) -> int:
        score = 0
        reasons = []

        # 1. 技术栈信号（累加）
        tech_score = self._score_tech_stack(text)
        score += tech_score
        if tech_score > 0:
            reasons.append(f"技术栈{tech_score}分")

        # 2. 功能复杂度（累加 + 领域去重）
        complexity_score = self._score_complexity(text)
        score += complexity_score
        if complexity_score > 0:
            reasons.append(f"功能复杂度{complexity_score}分")

        # 3. 项目名词
        noun_score = self._score_noun_implicit(text)
        score += noun_score
        if noun_score > 0:
            reasons.append(f"项目名词{noun_score}分")

        # 4. 文件数提示
        file_hint = self._score_file_hint(text)
        score += file_hint
        if file_hint > 0:
            reasons.append(f"文件数{file_hint}分")

        # 5. 需求长度
        length_score = self._score_length(text)
        score += length_score
        if length_score > 0:
            reasons.append(f"长需求{length_score}分")

        # 6. 用户强调语气
        emphasis_score = self._score_emphasis(text)
        score += emphasis_score
        if emphasis_score > 0:
            reasons.append(f"强调{emphasis_score}分")

        # 7. 历史上下文（含升级信号）
        history_score = self._score_history(text, history)
        score += history_score
        if history_score > 0:
            reasons.append(f"历史{history_score}分")

        # 8. 保底加分
        if re.search(r"系统|平台|管理|后台", text):
            score += self.FALLBACK_BONUS
            reasons.append(f"保底+{self.FALLBACK_BONUS}")

        # 9. 降级词
        if re.search(r"简单|简易|basic|简单版|别整那么复杂", text, re.I):
            score -= 2
            reasons.append("降级词-2")

        # 10. 爬虫/Web应用/界面类保底（提升到5分，包括后台管理）
        if re.search(r"爬虫|web应用|数据可视化|带界面|web app|工具|后台管理", text, re.I):
            if score < 5:
                score = 5
                reasons.append("爬虫/Web/后台类保底5分")

        # 11. "完整"信号额外加分（但被"不要"否定词覆盖）
        if "完整" in text and not re.search(r"不要|别|不用", text):
            if len(text) > 10:
                score += 2
                reasons.append("完整信号+2")

        score = max(0, min(score, 100))

        self._last_decision = {
            "score": score,
            "reasons": reasons,
            "text_preview": text[:50] + "..." if len(text) > 50 else text
        }

        return score

    def _score_tech_stack(self, text: str) -> int:
        score = 0
        text_lower = text.lower()
        for kw in self.TECH_STACK_HIGH:
            if kw in text or kw.lower() in text_lower:
                score += self.SCORE_WEIGHTS["tech_high"]
        for kw in self.TECH_STACK_MEDIUM:
            if kw in text or kw.lower() in text_lower:
                score += self.SCORE_WEIGHTS["tech_medium"]
        return min(score, 20)

    def _score_complexity(self, text: str) -> int:
        score = 0
        hit_groups = set()
        text_lower = text.lower()

        # 遍历高权重词
        for kw in self.COMPLEXITY_HIGH:
            if kw in text or kw.lower() in text_lower:
                # 检查属于哪个领域组
                for group_name, group_kws in self.COMPLEXITY_GROUPS.items():
                    if kw in group_kws:
                        hit_groups.add(group_name)
                        break
                score += self.SCORE_WEIGHTS["complexity_high"]

        # 中权重词（不分组，直接加）
        for kw in self.COMPLEXITY_MEDIUM:
            if kw in text or kw.lower() in text_lower:
                score += self.SCORE_WEIGHTS["complexity_medium"]

        # 领域覆盖加分（不同领域覆盖越多，项目越复杂）
        score += len(hit_groups) * self.SCORE_WEIGHTS["group_bonus"]

        return min(score, 25)

    def _score_noun_implicit(self, text: str) -> int:
        score = 0

        # 特殊高权重
        if "前后端分离" in text or "full stack" in text.lower():
            score += 5
        if "企业级" in text or "生产级" in text:
            score += 3
        # 数据库强信号（提升到6分）
        if "数据库" in text or "database" in text.lower():
            if not re.search(r"不要|别|不用|不需要", text):
                score += 6

        for kw in self.PROJECT_NOUN_HIGH:
            if kw in text:
                score += self.SCORE_WEIGHTS["noun_high"]
        for kw in self.PROJECT_NOUN_MEDIUM:
            if kw in text:
                score += self.SCORE_WEIGHTS["noun_medium"]

        return min(score, 15)

    def _score_file_hint(self, text: str) -> int:
        match = re.search(r"(\d+)\s*个文件", text)
        if match:
            num = int(match.group(1))
            if num >= 10:
                return self.SCORE_WEIGHTS["file_hint"] * 2
            elif num >= 5:
                return self.SCORE_WEIGHTS["file_hint"]
            elif num >= 3:
                return self.SCORE_WEIGHTS["file_hint"] // 2
        return 0

    def _score_length(self, text: str) -> int:
        char_count = len(text)
        word_count = char_count / 1.5
        if word_count > 200:
            return self.SCORE_WEIGHTS["length_bonus"] * 2
        elif word_count > 80:
            return self.SCORE_WEIGHTS["length_bonus"]
        return 0

    def _score_emphasis(self, text: str) -> int:
        score = 0
        if text.count("!") >= 2:
            score += self.SCORE_WEIGHTS["emphasis_bonus"]
        if re.search(r"(完整|全部|所有|功能).*\1", text):
            score += self.SCORE_WEIGHTS["emphasis_bonus"]
        return min(score, 4)

    def _score_history(self, text: str, history: List[str]) -> int:
        if not history:
            return 0

        score = 0
        recent = history[-3:] if len(history) >= 3 else history

        # 检查历史中是否有“太简单/不够”等升级触发词
        has_upgrade_trigger = False
        for msg in recent:
            if re.search(r"太简单|不够|太简陋|不够完整", msg, re.I):
                has_upgrade_trigger = True
                score += self.SCORE_WEIGHTS["history_bonus"]
                break

        # 如果有升级触发，且当前消息含“完整/升级/更好的”，额外加分
        if has_upgrade_trigger:
            if re.search(r"完整|升级|更好的|更多的", text, re.I):
                score += 3

        # 检查历史中是否有明确指定模式（但不含否定）
        for msg in recent:
            if re.search(r"demo|演示|极简|完整工程", msg, re.I):
                if not re.search(r"不要|别|不用", msg, re.I):
                    score += 1
                    break

        return min(score, 6)

    # ========== 模式映射 ==========

    def _score_to_mode(self, score: int) -> str:
        if score >= self.THRESHOLDS["complete_min"]:
            return "complete"
        elif score >= self.THRESHOLDS["demo_min"]:
            return "demo"
        else:
            return "simple"

    def _score_range_name(self, score: int) -> str:
        if score >= self.THRESHOLDS["complete_min"]:
            return "完整工程"
        elif score >= self.THRESHOLDS["demo_min"]:
            return "Demo"
        else:
            return "极简"

    def _in_fuzzy_zone(self, score: int) -> bool:
        return self.FUZZY_LOW <= score <= self.FUZZY_HIGH

    def _llm_fallback(self, text: str, history: List[str], llm_func: Callable[[str], str]) -> Optional[str]:
        try:
            hist_text = "\n".join(history[-3:]) if history else "无"
            prompt = f"""
你是模式判断专家。用户需求如下：

用户输入：{text}
历史对话：{hist_text}

请判断该需求应该使用哪种生成模式：
1. simple（极简模式）：单文件或少量代码，5分钟能完成，无数据库、无前端
2. demo（演示模式）：多文件但结构简单，适合快速演示，5-15个文件
3. complete（完整工程模式）：大项目，含数据库、前后端、多模块、权限等

只输出一个词：simple / demo / complete
"""
            result = llm_func(prompt)
            result = result.strip().lower()
            if result in ["simple", "demo", "complete"]:
                return result
        except Exception as e:
            print(f"[ModeJudge] LLM兜底失败: {e}")
        return None

    # ========== 调试接口 ==========

    def get_last_decision(self) -> Optional[Dict]:
        return self._last_decision

    def print_decision(self):
        if not self._last_decision:
            print("暂无决策记录")
            return
        print("=" * 50)
        print(f"📊 综合评分: {self._last_decision['score']} 分")
        print(f"📝 加分项: {', '.join(self._last_decision['reasons']) if self._last_decision['reasons'] else '无'}")
        print(f"📄 输入预览: {self._last_decision['text_preview']}")
        print("=" * 50)


# ========== 独立测试 ==========
if __name__ == "__main__":
    judge = ModeJudge(use_llm_fallback=False)

    test_cases = [
        # ===== 1. 明确模式关键词 =====
        ("帮我写一个demo版的图书管理系统", []),
        ("用完整工程模式生成一个电商后台", []),
        ("极简版：一个计算器", []),
        ("我要一个完整模式的爬虫项目", []),
        ("来个演示版的数据看板", []),
        ("单文件版，一个脚本搞定", []),
        ("full project 用户管理系统", []),
        ("simple project 随机数生成器", []),

        # ===== 2. 高复杂度 =====
        ("写一个图书管理系统，该有的全部给我安排上，足够完整，不是几行代码就搞定的那种，前后端都得有吧", []),
        ("写一个完整的电商后台系统，包含用户管理、订单管理、商品管理、权限控制，前后端分离", []),
        ("开发一个企业级OA系统，包含审批流程、考勤管理、工资核算、部门协作", []),
        ("帮我做一个医院挂号系统，患者端、医生端、管理员端，预约、缴费、病历管理", []),
        ("需要一个学校教务管理系统，学生选课、成绩录入、课程安排、教师管理、排课冲突检测", []),
        ("开发一个餐饮连锁店管理系统，多门店、库存管理、采购审批、销售统计、员工排班", []),
        ("帮我写一个完整的博客平台，用户注册登录、文章发布、评论、分类、标签、搜索", []),
        ("做一个项目管理系统，任务拆分、进度追踪、多人协作、甘特图、里程碑", []),
        ("开发一个在线教育平台，课程管理、章节进度、作业提交、考试成绩、师生互动", []),
        ("帮我做一个社区团购小程序的后台，商品管理、订单处理、配送调度、财务结算", []),

        # ===== 3. 中等复杂度 =====
        ("帮我写一个简单的图书管理系统，能增删改查就行", []),
        ("写一个天气查询工具，输入城市名返回天气信息", []),
        ("做一个股票数据监控脚本，定时获取并保存到csv", []),
        ("帮我生成一个个人博客的简单后端，Flask + SQLite", []),
        ("写一个爬虫，爬取新闻网站标题和摘要，保存到json", []),
        ("开发一个单词打卡工具，记录每天学习的单词", []),
        ("帮我做一个待办事项管理工具，支持添加、完成、删除", []),
        ("写一个邮件群发工具，读取Excel名单发送模板邮件", []),

        # ===== 4. 低复杂度 =====
        ("计算器", []),
        ("随机数生成器", []),
        ("帮我写一个hello world", []),
        ("简单脚本：统计文本文件中的单词数", []),
        ("一个函数，把列表里的数字都乘以2", []),
        ("写一个装饰器，打印函数执行时间", []),
        ("一个简单的数据清洗脚本，去除空行和重复", []),
        ("帮我写个函数，读csv文件算平均值", []),
        ("把一个字符串转换成MD5", []),
        ("请求一个API，把结果存成json文件", []),

        # ===== 5. 模糊需求 =====
        ("写一个系统", []),
        ("帮我做一个管理平台", []),
        ("写一个爬虫", []),
        ("做一个Web应用", []),
        ("数据可视化工具", []),
        ("帮我写一个工具", []),
        ("一个带界面的小工具", []),
        ("做一个后台管理", []),
        ("自动化脚本", []),

        # ===== 6. 情绪化输入 =====
        ("写一个图书管理系统！！不是那种几行代码的！！！！", []),
        ("太简单了！我要一个完整的！！！！", ["之前的代码太简单了"]),
        ("就写个计算器，别整那么复杂", []),
        ("帮我写一个简单的demo演示", []),
        ("随便写一个小工具就行", []),

        # ===== 7. 技术栈信号 =====
        ("用Flask写一个博客", []),
        ("用Django开发一个内容管理系统", []),
        ("前后端分离：Vue + FastAPI 做一个记账本", []),
        ("Spring Boot + MySQL 做一个员工管理系统", []),
        ("用React写一个商城前端", []),
        ("用Go写一个高并发服务器", []),

        # ===== 8. 功能复杂度信号 =====
        ("用户登录注册、权限管理、角色分配", []),
        ("借书还书、预约、逾期罚款、续借", []),
        ("报表统计、数据看板、导出Excel", []),
        ("多仓库管理、库存预警、采购申请", []),

        # ===== 9. 英文需求 =====
        ("write a complete e-commerce system with user management and order system", []),
        ("make a demo project for api testing", []),
        ("simple script to fetch data from database", []),
        ("full stack blog application with react and nodejs", []),

        # ===== 10. 历史上下文 =====
        ("写一个图书管理系统", ["帮我写一个demo版"]),
        ("还是写一个完整的吧", ["之前那个太简单了，不够用"]),
        ("算了，简单点就行", ["刚才说的那个太复杂了"]),
        ("不是这个，我要带数据库的", ["帮我写一个计算器"]),
        ("对，就是这样的系统", ["写一个完整的图书管理系统"]),
        ("再改一下，不要前后端分离了", ["写一个完整的前后端分离的博客"]),
        ("还是用之前那个方案", ["极简版：一个脚本"]),
        ("我刚说错了，不是demo版，是完整版", ["帮我写一个demo版的图书管理系统"]),

        # ===== 11. 极端边界 =====
        ("", []),
        ("？", []),
        ("111", []),
        ("。。。。", []),
        ("系统系统系统系统系统", []),
        ("前后端前后端前后端", []),
        ("完整完整完整完整完整", []),
        ("demo demo demo demo demo", []),
        ("极简极简极简极简极简", []),
        ("写一个超级超级超级超级复杂的系统", []),
    ]

    print("=" * 80)
    print("🧪 模式判断测试报告")
    print("=" * 80)

    stats = {"simple": 0, "demo": 0, "complete": 0}

    for idx, (text, hist) in enumerate(test_cases, 1):
        result = judge.route(text, hist)
        mode = result["mode"]
        stats[mode] = stats.get(mode, 0) + 1

        status = ""
        if result["used_llm"]:
            status = "🧠 LLM"
        elif result["score"] == 100:
            status = "🎯 明确命中"
        elif result["score"] >= 10:
            status = "🔴 高复杂"
        elif result["score"] >= 5:
            status = "🟡 中等"
        else:
            status = "🟢 低复杂"

        display_text = text[:40] + "..." if len(text) > 40 else text
        hist_display = " 📚" if hist else ""

        print(f"{idx:2d}. [{mode:8s}] [{result['score']:3d}分] {status:10s} | {display_text}{hist_display}")

    print("=" * 80)
    print(f"📊 统计: 极简 {stats['simple']} 个, Demo {stats['demo']} 个, 完整 {stats['complete']} 个, 总计 {len(test_cases)} 个")
    print("=" * 80)