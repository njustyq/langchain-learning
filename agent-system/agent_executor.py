"""
SmartAgent —— 基于 LangChain 1.2.x + LangGraph 的智能 Agent 系统

核心改进（对比旧版手动 ReAct 循环）：
─────────────────────────────────────────────────────────────────────────────
旧版问题：
  - 手动调用 LLM → 用正则解析 "Action:" / "Action Input:" 文本 → 手动调用工具
  - 文本格式脆弱：任何输出格式变化都会导致解析失败
  - 无真正的工具调用，实际上是字符串匹配游戏

新版方案（LangChain 1.2.x `create_agent`）：
  - 利用模型原生 Function Calling 能力，以结构化 JSON 声明工具调用
  - AgentState 消息流：HumanMessage → AIMessage(tool_calls) → ToolMessage → AIMessage(最终答案)
  - LangGraph 状态机自动处理循环：推理 → 工具执行 → 继续推理
  - 内置 MemorySaver 支持多轮对话，线程安全
─────────────────────────────────────────────────────────────────────────────
"""

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from typing import List

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "langchain-test"))


class SmartAgent:
    """智能 Agent 系统

    基于 LangChain 1.2.x `create_agent`（LangGraph 驱动），核心能力：
    - 真实工具调用：模型通过 Function Calling 以 JSON 精确指定工具名称和参数
    - 自动推理循环：推理 → 调用工具 → 观察结果 → 继续推理，直到给出最终答案
    - 多工具协作：自动决定需要调用哪些工具、调用几次
    - 可选对话记忆：跨多轮问答保持上下文（通过 LangGraph Checkpointer）
    """

    def __init__(
        self,
        tools: List[BaseTool],
        verbose: bool = True,
        enable_memory: bool = False,
        max_iterations: int = 10,
        system_prompt: str | None = None,
    ):
        """初始化智能 Agent

        Args:
            tools:          可用工具列表（langchain_core.tools.BaseTool 实例）
            verbose:        是否在控制台打印每一步的推理过程
            enable_memory:  是否启用多轮对话记忆（不同 session_id 彼此隔离）
            max_iterations: 最大工具调用轮次，防止无限循环
            system_prompt:  自定义系统提示，None 时使用默认提示
        """
        self.tools = tools
        self.verbose = verbose
        self.enable_memory = enable_memory

        from baseDef import createLlm
        self.llm = createLlm()

        # 系统提示：赋予 Agent 角色与工作原则
        _system_prompt = system_prompt or (
            "你是一个强大的智能助手，可以借助工具解决各种问题。\n\n"
            "工作原则：\n"
            "1. 仔细理解用户的问题，判断是否需要调用工具\n"
            "2. 选择最合适的工具，并提供准确的参数\n"
            "3. 综合多次工具调用的结果来回答复杂问题\n"
            "4. 如果可以直接回答，无需调用工具\n"
            "5. 始终用中文回复用户"
        )

        # ── 创建 Agent（LangGraph 驱动的工具调用循环）────────────────────────────
        # create_agent 返回 CompiledStateGraph，内部自动处理：
        #   1. 模型推理（决定是否调用工具及调用参数）
        #   2. 工具执行（并发或串行）
        #   3. 将工具结果追加到消息历史，触发下一轮推理
        #   4. 直到模型不再调用工具，输出最终 AIMessage
        #
        # checkpointer 启用持久化记忆：每个 thread_id 对应独立的对话历史
        self._checkpointer = MemorySaver() if enable_memory else None

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=_system_prompt,
            checkpointer=self._checkpointer,
        )

    # ── 内部辅助方法 ────────────────────────────────────────────────────────────

    def _extract_tool_steps(self, messages: List[BaseMessage]) -> list[tuple]:
        """从消息历史中提取工具调用步骤（工具名 + 输入 + 输出）

        Args:
            messages: AgentState 中的完整消息列表

        Returns:
            list of (tool_name, tool_input, tool_output)
        """
        steps = []
        # 构建 tool_call_id → (tool_name, tool_input) 映射
        call_map: dict[str, tuple[str, str]] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        call_map[tc_id] = (tc["name"], str(tc["args"]))
        # 将 ToolMessage 与对应的调用配对
        for msg in messages:
            if isinstance(msg, ToolMessage):
                name, inp = call_map.get(msg.tool_call_id, ("unknown", ""))
                steps.append((name, inp, msg.content))
        return steps

    def _print_steps(self, steps: list[tuple]) -> None:
        """格式化打印工具调用的中间步骤"""
        if not steps:
            return
        print(f"\n{'─' * 60}")
        print(f"[步骤汇总] 共执行了 {len(steps)} 个工具调用：")
        for i, (tool_name, tool_input, tool_output) in enumerate(steps, 1):
            print(f"\n  [{i}] 工具：{tool_name}")
            print(f"       输入：{tool_input}")
            obs = str(tool_output)
            print(f"       结果：{obs[:300]}{'...' if len(obs) > 300 else ''}")
        print(f"{'─' * 60}\n")

    # ── 公开方法 ────────────────────────────────────────────────────────────────

    def run(self, question: str, session_id: str = "default") -> str:
        """执行单次或多轮查询

        Args:
            question:   用户问题
            session_id: 多轮对话的会话标识（仅 enable_memory=True 时有效，
                        不同 session_id 拥有独立的对话历史）

        Returns:
            Agent 的最终回答文本
        """
        print(f"\n{'=' * 80}")
        print(f"[问题] {question}")
        print(f"{'=' * 80}\n")

        # 构建调用配置（多轮记忆靠 thread_id 隔离不同会话）
        config: RunnableConfig = {"configurable": {"thread_id": session_id}} if self.enable_memory else {}

        try:
            if self.verbose:
                # stream 模式：实时打印每一步的消息，便于观察推理过程
                final_messages: List[BaseMessage] = []
                for chunk in self.agent.stream(
                    {"messages": [HumanMessage(content=question)]},
                    config=config,
                    stream_mode="values",  # 每步输出完整状态
                ):
                    # chunk["messages"] 是当前状态的完整消息列表
                    msgs = chunk.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        # 打印工具调用请求
                        if isinstance(last, AIMessage) and last.tool_calls:
                            for tc in last.tool_calls:
                                print(f"  >> 调用工具 [{tc['name']}]，参数：{tc['args']}")
                        # 打印工具执行结果
                        elif isinstance(last, ToolMessage):
                            obs = str(last.content)
                            print(f"  << 工具返回：{obs[:200]}{'...' if len(obs) > 200 else ''}")
                    final_messages = msgs

                # 打印所有工具调用步骤汇总
                steps = self._extract_tool_steps(final_messages)
                self._print_steps(steps)

                # 最后一条 AIMessage 即最终答案
                for msg in reversed(final_messages):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        content = msg.content
                        return str(content) if content else "（模型返回了空回答）"

            else:
                # invoke 模式：一次性获取结果
                result = self.agent.invoke(
                    {"messages": [HumanMessage(content=question)]},
                    config=config,
                )
                messages: List[BaseMessage] = result.get("messages", [])
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        content = msg.content
                        return str(content) if content else "（模型返回了空回答）"

            return "未能获取最终答案"

        except Exception as e:
            return f"执行出错: {str(e)}"

    def show_tools(self) -> None:
        """列出所有可用工具及其描述"""
        print("\n[可用工具列表]")
        print("=" * 80)
        for i, tool in enumerate(self.tools, 1):
            print(f"\n  {i}. [{tool.name}]")
            desc = tool.description.strip().replace("\n", " ")
            print(f"     {desc[:150]}{'...' if len(desc) > 150 else ''}")
        print("=" * 80)

    def clear_memory(self, session_id: str = "default") -> None:
        """清除指定 session 的对话历史（仅 enable_memory=True 时有效）

        Note:
            MemorySaver 不提供直接的 clear 接口，切换新的 session_id 即可开启新对话。

        Args:
            session_id: 要清除的会话标识
        """
        if not self.enable_memory:
            print("[警告] 当前 Agent 未启用记忆功能")
            return
        # MemorySaver 使用新 thread_id 即等同于新会话，提示用户即可
        print(f"[提示] 使用新的 session_id（如 '{session_id}_new'）即可开启全新对话")


# ── 测试入口 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tools.weather_tool import weather_tool
    from tools.calculator_tool import calculator_tool
    from tools.search_tool import search_tool

    # 创建 Agent（开启记忆，便于测试多轮对话）
    agent = SmartAgent(
        tools=[weather_tool, calculator_tool, search_tool],
        verbose=True,
        enable_memory=True,
    )

    # 显示工具
    agent.show_tools()

    # ── 单工具测试 ────────────────────────────────────────────────────────────
    single_tool_questions = [
        "北京今天天气怎么样？",
        "计算 (25 + 15) * 2",
        "Python 是什么？",
    ]
    for question in single_tool_questions:
        answer = agent.run(question)
        print(f"\n[答案] {answer}\n")

    # ── 多工具联合测试 ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[多工具联合测试] 查询两个城市天气后做计算")
    answer = agent.run(
        "查询北京和上海的温度，然后计算两个城市温度差的平方",
        session_id="multi_tool_test",
    )
    print(f"\n[答案] {answer}\n")

    # ── 多轮对话测试 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[多轮对话测试] 同一 session 保持上下文")
    session = "conversation_test"
    multi_turn_questions = [
        "深圳今天天气怎么样？",
        "那上海呢？",                           # 延续上文语境
        "两个城市的温度相差多少度？",             # 联合前两轮信息作计算
    ]
    for q in multi_turn_questions:
        answer = agent.run(q, session_id=session)
        print(f"\n[答案] {answer}\n")
