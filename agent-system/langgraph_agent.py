"""
LangGraphAgent —— 基于原生 LangGraph StateGraph 的智能 Agent

与 SmartAgent（基于 create_agent 高层 API）的核心对比：
────────────────────────────────────────────────────────────────────────
SmartAgent（agent_executor.py）：
  - 调用 create_agent() 一行创建图，内部结构不可见
  - 只能得到最终消息，无法感知每个节点的执行状态

LangGraphAgent（本文件）：
  - 手动定义 StateGraph，节点/边/路由条件全部显式声明
  - AgentState 携带 node_history + tool_results，记录每个节点快照
  - 可在节点间传递任意结构化状态（不仅限于消息流）
  - 支持流式输出、状态可视化、多轮记忆
────────────────────────────────────────────────────────────────────────

图结构（ReAct 循环）：

  START
    │
    ▼
  ┌──────┐  有 tool_calls   ┌─────┐
  │think │─────────────────▶│ act │
  │(推理)│                  │(执行)│
  └──────┘◀─────────────────└─────┘
    │       返回工具结果
    │ 无 tool_calls
    ▼
   END
"""

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "langchain-test"))


# ── 状态类型定义 ──────────────────────────────────────────────────────────────

class NodeRecord(TypedDict):
    """单个节点的执行快照，追加到 AgentState.node_history"""
    node: str            # 节点名称：'think' | 'act'
    iteration: int       # 所处的推理轮次（从 1 开始）
    input_summary: str   # 本次输入摘要（供调试）
    output_summary: str  # 本次输出摘要（供调试）


class ToolResult(TypedDict):
    """单次工具调用的完整记录，追加到 AgentState.tool_results"""
    tool: str      # 工具名称
    args: Any      # 调用参数
    result: str    # 工具返回内容
    iteration: int # 所处的推理轮次


class AgentState(TypedDict):
    """
    Agent 全局状态，在整个 StateGraph 执行过程中流动。

    LangGraph Reducer 机制：
    ┌──────────────┬────────────────────────────────────────────┐
    │   字段        │  Reducer（节点输出如何合并到现有状态）        │
    ├──────────────┼────────────────────────────────────────────┤
    │ messages     │ add_messages：自动去重追加，保留 id 一致性    │
    │ node_history │ operator.add：列表拼接，保留全部节点快照      │
    │ tool_results │ operator.add：列表拼接，保留全部工具调用记录  │
    │ iteration    │ 无 Reducer：后写覆盖（取最新轮次编号）        │
    └──────────────┴────────────────────────────────────────────┘
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    node_history: Annotated[list[NodeRecord], operator.add]
    tool_results: Annotated[list[ToolResult], operator.add]
    iteration: int


# ── LangGraphAgent 主类 ───────────────────────────────────────────────────────

class LangGraphAgent:
    """
    基于原生 LangGraph StateGraph 构建的智能 Agent。

    核心特性：
    1. 工具调用：模型通过 Function Calling 以结构化 JSON 声明工具及参数
    2. 自动推理循环：think → act → think，直到模型不再调用工具
    3. 节点状态记录：每个节点执行后将快照写入 AgentState，可随时查阅
    4. 可选对话记忆：MemorySaver Checkpointer，不同 session_id 独立隔离
    """

    def __init__(
        self,
        tools: list[BaseTool],
        verbose: bool = True,
        enable_memory: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            tools:          工具列表（langchain_core.tools.BaseTool 实例）
            verbose:        是否实时打印每个节点的执行过程
            enable_memory:  是否启用多轮对话记忆（MemorySaver）
            system_prompt:  自定义系统提示，None 时使用内置默认提示
        """
        self.tools = tools
        self.verbose = verbose
        self.enable_memory = enable_memory

        from baseDef import createLlm  # type: ignore[import-not-found]

        # 将工具绑定到 LLM，使其支持 Function Calling
        base_llm = createLlm()
        self.llm = base_llm.bind_tools(tools)

        self._system_prompt = system_prompt or (
            "你是一个强大的智能助手，可以借助工具解决各种问题。\n\n"
            "工作原则：\n"
            "1. 仔细理解用户的问题，判断是否需要调用工具\n"
            "2. 选择最合适的工具，并提供准确的参数\n"
            "3. 综合多次工具调用的结果来回答复杂问题\n"
            "4. 如果可以直接回答，无需调用工具\n"
            "5. 始终用中文回复用户"
        )

        # ToolNode 负责工具执行：自动匹配 tool_call_id，处理异常
        self._tool_node = ToolNode(tools)

        # Checkpointer：启用后每个 thread_id 对应独立的对话历史快照
        self._checkpointer = MemorySaver() if enable_memory else None

        # 构建并编译状态图
        self.graph = self._build_graph()

    # ── 图构建 ─────────────────────────────────────────────────────────────────

    def _build_graph(self):
        """
        构建 ReAct 推理图：
          START → think ─(有工具调用)→ act → think（循环）
                        ─(无工具调用)→ END
        """
        builder = StateGraph(AgentState)

        # 注册节点
        builder.add_node("think", self._think_node)
        builder.add_node("act", self._act_node)

        # 入口：从 think 开始
        builder.set_entry_point("think")

        # 条件路由：think 之后根据模型输出决定走向
        builder.add_conditional_edges(
            "think",
            self._route_after_think,
            {"act": "act", END: END},
        )

        # act 执行完工具后，无条件回到 think 继续推理
        builder.add_edge("act", "think")

        return builder.compile(checkpointer=self._checkpointer)

    # ── 节点实现 ───────────────────────────────────────────────────────────────

    def _think_node(self, state: AgentState) -> dict:
        """
        推理节点（think）：
        - 将当前消息历史 + 系统提示发送给 LLM
        - 模型返回 AIMessage（可能包含 tool_calls，也可能是最终答案）
        - 将执行快照写入 node_history
        """
        messages = list(state["messages"])
        iteration = state.get("iteration", 0) + 1

        # 系统提示注入：如果第一条不是 system 消息则插入
        from langchain_core.messages import SystemMessage
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self._system_prompt)] + messages

        response: AIMessage = self.llm.invoke(messages)  # type: ignore[assignment]

        # 构建节点快照
        if response.tool_calls:
            tool_names = [tc["name"] for tc in response.tool_calls]
            output_summary = f"请求工具调用: {tool_names}"
        else:
            content_preview = str(response.content)[:80]
            output_summary = f"最终回答: {content_preview}{'...' if len(str(response.content)) > 80 else ''}"

        record: NodeRecord = {
            "node": "think",
            "iteration": iteration,
            "input_summary": f"消息数={len(messages)}，最后一条={type(messages[-1]).__name__}",
            "output_summary": output_summary,
        }

        if self.verbose:
            print(f"\n  [think #{iteration}] {output_summary}")

        return {
            "messages": [response],
            "node_history": [record],
            "tool_results": [],
            "iteration": iteration,
        }

    def _act_node(self, state: AgentState) -> dict:
        """
        执行节点（act）：
        - 取最后一条 AIMessage 中的 tool_calls
        - 通过 ToolNode 调用对应工具并获取 ToolMessage 列表
        - 将每次调用的工具名/参数/结果记录到 tool_results
        - 将节点快照写入 node_history
        """
        messages = list(state["messages"])
        iteration = state.get("iteration", 0)
        last_ai: AIMessage = messages[-1]  # type: ignore[assignment]

        # 收集待执行的工具调用信息（用于事后配对结果）
        planned: dict[str, dict] = {}
        if isinstance(last_ai, AIMessage) and last_ai.tool_calls:
            for tc in last_ai.tool_calls:
                tc_id: str = tc.get("id") or ""
                if tc_id:
                    planned[tc_id] = {"tool": tc["name"], "args": tc["args"]}

        if self.verbose:
            for info in planned.values():
                print(f"  [act  #{iteration}] >> 调用工具 [{info['tool']}]，参数：{info['args']}")

        # ToolNode 执行：输入当前消息列表，返回 {"messages": [ToolMessage, ...]}
        tool_output: dict = self._tool_node.invoke({"messages": messages})
        new_messages: list[ToolMessage] = tool_output.get("messages", [])

        # 将工具结果与调用请求配对，构建 ToolResult 记录
        tool_results: list[ToolResult] = []
        for tm in new_messages:
            if isinstance(tm, ToolMessage):
                info = planned.get(tm.tool_call_id, {"tool": "unknown", "args": {}})
                result_preview = str(tm.content)[:200]
                tool_results.append(
                    ToolResult(
                        tool=info["tool"],
                        args=info["args"],
                        result=str(tm.content),
                        iteration=iteration,
                    )
                )
                if self.verbose:
                    print(
                        f"  [act  #{iteration}] << [{info['tool']}] 返回："
                        f"{result_preview}{'...' if len(str(tm.content)) > 200 else ''}"
                    )

        record: NodeRecord = {
            "node": "act",
            "iteration": iteration,
            "input_summary": f"工具调用数={len(planned)}，工具列表={list(info['tool'] for info in planned.values())}",
            "output_summary": f"获得 {len(tool_results)} 个工具结果",
        }

        return {
            "messages": new_messages,
            "node_history": [record],
            "tool_results": tool_results,
            "iteration": iteration,
        }

    # ── 路由函数 ───────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_think(state: AgentState) -> str:
        """
        条件路由：检查 think 节点输出的最后一条消息
        - AIMessage 含 tool_calls → 走向 'act' 节点继续执行工具
        - AIMessage 不含 tool_calls → 走向 END，结束推理循环
        """
        last = list(state["messages"])[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "act"
        return END

    # ── 公开方法 ───────────────────────────────────────────────────────────────

    def run(self, question: str, session_id: str = "default") -> str:
        """
        执行一次推理，返回最终答案文本。

        Args:
            question:   用户问题
            session_id: 多轮对话标识（仅 enable_memory=True 时生效）

        Returns:
            Agent 最终回答文本
        """
        print(f"\n{'=' * 80}")
        print(f"[问题] {question}")
        print(f"{'=' * 80}")

        config: RunnableConfig = (
            {"configurable": {"thread_id": session_id}} if self.enable_memory else {}
        )

        try:
            final_state: dict | None = None

            if self.verbose:
                # stream_mode="values"：每步输出完整状态快照，便于实时观察
                for snapshot in self.graph.stream(
                    {
                        "messages": [HumanMessage(content=question)],
                        "node_history": [],
                        "tool_results": [],
                        "iteration": 0,
                    },
                    config=config,
                    stream_mode="values",
                ):
                    final_state = snapshot
            else:
                final_state = self.graph.invoke(
                    {
                        "messages": [HumanMessage(content=question)],
                        "node_history": [],
                        "tool_results": [],
                        "iteration": 0,
                    },
                    config=config,
                )

            if final_state is None:
                return "未能获取结果"

            # 提取最终答案（最后一条无 tool_calls 的 AIMessage）
            for msg in reversed(list(final_state["messages"])):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    answer = str(msg.content) if msg.content else "（模型返回空回答）"
                    return answer

            return "未能获取最终答案"

        except Exception as e:
            return f"执行出错: {e}"

    def stream_run(self, question: str, session_id: str = "default"):
        """
        以生成器方式流式返回每一步的状态快照，供外部逐步消费。

        Yields:
            每一步的 AgentState 快照字典
        """
        config: RunnableConfig = (
            {"configurable": {"thread_id": session_id}} if self.enable_memory else {}
        )
        yield from self.graph.stream(
            {
                "messages": [HumanMessage(content=question)],
                "node_history": [],
                "tool_results": [],
                "iteration": 0,
            },
            config=config,
            stream_mode="values",
        )

    def show_state(self, final_state: AgentState) -> None:
        """
        格式化打印 AgentState 中记录的节点执行历史与工具调用详情。

        Args:
            final_state: 执行结束后的完整状态（由 run 内部获取，或外部传入）
        """
        node_history: list[NodeRecord] = final_state.get("node_history", [])
        tool_results: list[ToolResult] = final_state.get("tool_results", [])
        total_iter: int = final_state.get("iteration", 0)

        print(f"\n{'─' * 70}")
        print(f"[状态报告] 共推理 {total_iter} 轮，执行 {len(tool_results)} 次工具调用")
        print(f"{'─' * 70}")

        print("\n  节点执行历史：")
        for i, rec in enumerate(node_history, 1):
            tag = "🧠 think" if rec["node"] == "think" else "⚙️  act  "
            print(f"    [{i:02d}] {tag} | 轮次={rec['iteration']}")
            print(f"          输入: {rec['input_summary']}")
            print(f"          输出: {rec['output_summary']}")

        if tool_results:
            print("\n  工具调用详情：")
            for j, tr in enumerate(tool_results, 1):
                print(f"    [{j}] 工具={tr['tool']}  轮次={tr['iteration']}")
                print(f"        参数: {tr['args']}")
                result_str = str(tr["result"])
                print(
                    f"        结果: {result_str[:300]}{'...' if len(result_str) > 300 else ''}"
                )

        print(f"{'─' * 70}\n")

    def show_tools(self) -> None:
        """列出所有已注册工具及描述"""
        print("\n[可用工具列表]")
        print("=" * 80)
        for i, tool in enumerate(self.tools, 1):
            print(f"\n  {i}. [{tool.name}]")
            desc = tool.description.strip().replace("\n", " ")
            print(f"     {desc[:160]}{'...' if len(desc) > 160 else ''}")
        print("=" * 80)

    def show_graph(self) -> None:
        """打印 LangGraph 状态图的 ASCII 结构（需要 pygraphviz 或 mermaid）"""
        try:
            print(self.graph.get_graph().draw_ascii())
        except Exception:
            print(
                "[图结构]\n"
                "  START\n"
                "    │\n"
                "    ▼\n"
                "  [think] ──(有 tool_calls)──▶ [act] ──┐\n"
                "    │                                   │\n"
                "    │◀──────────────────────────────────┘\n"
                "    │\n"
                "  (无 tool_calls)\n"
                "    │\n"
                "    ▼\n"
                "   END"
            )


# ── 测试入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tools.weather_tool import weather_tool
    from tools.calculator_tool import calculator_tool
    from tools.search_tool import search_tool

    # ── 初始化 Agent ───────────────────────────────────────────────────────────
    agent = LangGraphAgent(
        tools=[weather_tool, calculator_tool, search_tool],
        verbose=True,
        enable_memory=True,
    )

    agent.show_tools()
    agent.show_graph()

    # ── 单工具测试 ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[测试 1] 单工具调用：查询天气")
    answer = agent.run("北京今天天气怎么样？")
    print(f"\n[答案] {answer}")

    print("\n" + "=" * 80)
    print("[测试 2] 单工具调用：数学计算")
    answer = agent.run("计算 (25 + 15) * 2 的结果")
    print(f"\n[答案] {answer}")

    # ── 多工具联合测试 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[测试 3] 多工具协作：查询两城市天气 → 计算温差平方")

    final_state: AgentState | None = None
    for snapshot in agent.stream_run(
        "查询北京和上海的温度，然后计算两个城市温度差的平方",
        session_id="multi_tool_test",
    ):
        final_state = snapshot  # type: ignore[assignment]

    if final_state:
        for msg in reversed(list(final_state["messages"])):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                print(f"\n[答案] {msg.content}")
                break
        # 打印完整状态报告
        agent.show_state(final_state)

    # ── 多轮对话测试（利用 MemorySaver 跨轮保持上下文）─────────────────────────
    print("\n" + "=" * 80)
    print("[测试 4] 多轮对话：同一 session 保持上下文")
    session = "conversation_test"
    turns = [
        "深圳今天天气怎么样？",
        "那上海呢？",
        "两个城市的温度相差多少度？",
    ]
    for q in turns:
        ans = agent.run(q, session_id=session)
        print(f"\n[答案] {ans}\n")
