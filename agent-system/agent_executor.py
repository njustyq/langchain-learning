
from langchain_core.tools import Tool
from typing import List

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent /"langchain-test"))

class SmartAgent:
    """智能 Agent 系统"""
    
    def __init__(self, tools: List[Tool], verbose: bool = True):
        """初始化智能 Agent 系统"""
        self.tools = tools
        self.verbose = verbose
        
        from baseDef import createLlm, createAgentToolsPrompt
        # 创建 LLM
        self.llm = createLlm()
        
        # 创建 Prompt（自定义 ReAct Prompt）
        self.prompt = createAgentToolsPrompt()
        # 工具名称到 Tool 实例的映射，方便通过名字调用
        self.tool_map = {tool.name: tool for tool in self.tools}
    
    def run(self, question: str) -> str:
        """执行查询（简单 ReAct 循环，手动调度工具）"""
        import re

        print(f"\n{'='*80}")
        print(f"问题: {question}")
        print(f"{'='*80}\n")

        scratchpad = ""
        max_iterations = 10

        try:
            for _ in range(max_iterations):
                # 将工具信息格式化到 Prompt 中
                tools_desc = "\n".join(
                    f"{tool.name}: {tool.description}" for tool in self.tools
                )
                formatted_prompt = self.prompt.format(
                    tools=tools_desc,
                    agent_scratchpad=scratchpad,
                    input=question,
                )

                llm_response = self.llm.invoke(formatted_prompt)
                if hasattr(llm_response, "content"):
                    response_text = llm_response.content
                else:
                    response_text = str(llm_response)

                # 如果已经给出最终答案，直接返回
                if "Final Answer:" in response_text:
                    final_answer = response_text.split("Final Answer:", 1)[1].strip()
                    return final_answer

                # 解析 Action 和 Action Input
                action_match = re.search(r"Action\s*:\s*(.+)", response_text)
                input_match = re.search(
                    r"Action Input\s*:\s*(.+)", response_text, re.DOTALL
                )

                if not action_match or not input_match:
                    # 无法解析工具调用，直接返回当前模型输出
                    return f"模型输出（未能解析工具调用）：\n{response_text}"

                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip().strip('"').strip("'")

                tool = self.tool_map.get(tool_name)
                if tool is None:
                    observation = f"工具 {tool_name} 不存在，请检查工具名称。"
                else:
                    try:
                        observation = tool.run(tool_input)
                    except Exception as e:
                        observation = f"调用工具 {tool_name} 出错：{e}"

                # 将这一步的 Thought/Action/Observation 添加到 scratchpad 中，供下一轮推理使用
                scratchpad += f"\n{response_text}\nObservation: {observation}\n"

            return "达到最大迭代次数，但未给出最终答案。"
        except Exception as e:
            return f"执行出错: {str(e)}"
    def show_tools(self):
        """显示可用工具"""
        print("\n可用工具：")
        print("=" * 80)
        for i, tool in enumerate(self.tools, 1):
            print(f"\n{i}. {tool.name}")
            print(f"   描述: {tool.description[:100]}...")
        print("=" * 80)

# 测试
if __name__ == "__main__":
    from tools.weather_tool import weather_tool
    from tools.calculator_tool import calculator_tool
    from tools.search_tool import search_tool
    
    # 创建 Agent
    agent = SmartAgent(
        tools=[weather_tool, calculator_tool, search_tool],
        verbose=True
    )
    
    # 显示工具
    agent.show_tools()
    
    # 测试问题
    test_questions = [
        "北京今天天气怎么样？",
        "计算 (25 + 15) * 2",
        "Python 是什么？"
    ]
    
    for question in test_questions:
        answer = agent.run(question)
        print(f"\n答案: {answer}\n")