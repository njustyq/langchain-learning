"""
调试脚本：展示 prompt 和 llm 之间的数据流动过程
运行此脚本可以看到每一步的数据转换
"""
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# 定义输出结构
class TranslationResult(BaseModel):
    """翻译结果的结构"""
    translation: str = Field(description="英文翻译")
    quality_score: int = Field(description="翻译质量评分（1-10）")
    notes: str = Field(description="翻译说明或注意事项")


def debug_data_flow():
    """逐步展示数据流动过程"""
    # 加载环境变量
    load_dotenv()
    
    os.environ["OPENAI_API_KEY"] = "GSkkITz67tvyRj0eZncG"
    os.environ["OPENAI_API_BASE"] = "https://genaiapi.cloudsway.net/v1/ai/ZoDFMXKuoyxNhQjg"
    
    # 创建组件
    llm = ChatOpenAI(model="MaaS 3.7 Sonnet", temperature=0)
    parser = PydanticOutputParser(pydantic_object=TranslationResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的翻译专家，擅长将中文翻译成地道的英文。"),
        ("human", """请将以下中文翻译成英文，并评估翻译质量。

{format_instructions}

中文文本：
{text}
""")
    ])
    
    # 输入数据
    input_data = {
        "text": "人工智能正在改变世界",
        "format_instructions": parser.get_format_instructions()
    }
    
    print("=" * 80)
    print("步骤 1: 输入数据")
    print("=" * 80)
    print(f"输入字典: {input_data}")
    print(f"类型: {type(input_data)}")
    print()
    
    # 步骤 2: Prompt 格式化
    print("=" * 80)
    print("步骤 2: Prompt 格式化 (prompt.format())")
    print("=" * 80)
    formatted_prompt = prompt.invoke(input_data)
    print(f"格式化后的对象类型: {type(formatted_prompt)}")
    print(f"对象类名: {formatted_prompt.__class__.__name__}")
    print()
    
    # 转换为消息列表
    messages = formatted_prompt.to_messages()
    print("消息列表:")
    for i, msg in enumerate(messages):
        print(f"  消息 {i+1}:")
        print(f"    类型: {type(msg).__name__}")
        print(f"    内容: {msg.content[:100]}..." if len(msg.content) > 100 else f"    内容: {msg.content}")
        print()
    
    # 步骤 3: 传递给 LLM
    print("=" * 80)
    print("步骤 3: 传递给 LLM (llm.invoke())")
    print("=" * 80)
    print("LLM 接收的数据:")
    print(f"  类型: {type(formatted_prompt)}")
    print(f"  消息数量: {len(messages)}")
    print()
    print("LLM 内部会将消息转换为 API 请求格式:")
    print("  {")
    print('    "model": "MaaS 3.7 Sonnet",')
    print('    "messages": [')
    for msg in messages:
        role = "system" if msg.__class__.__name__ == "SystemMessage" else "user"
        print(f'      {{"role": "{role}", "content": "..."}},')
    print('    ],')
    print('    "temperature": 0')
    print("  }")
    print()
    
    # 调用 LLM
    print("正在调用 LLM API...")
    llm_response = llm.invoke(formatted_prompt)
    print()
    
    print("=" * 80)
    print("步骤 4: LLM 响应")
    print("=" * 80)
    print(f"响应对象类型: {type(llm_response)}")
    print(f"响应类名: {llm_response.__class__.__name__}")
    print(f"响应内容类型: {type(llm_response.content)}")
    print(f"响应内容: {llm_response.content}")
    print()
    
    # 步骤 5: Parser 解析
    print("=" * 80)
    print("步骤 5: Parser 解析 (parser.parse())")
    print("=" * 80)
    print(f"Parser 接收的数据: {llm_response.content}")
    print(f"数据类型: {type(llm_response.content)}")
    print()
    
    parsed_result = parser.invoke(llm_response)
    print("=" * 80)
    print("步骤 6: 最终结果")
    print("=" * 80)
    print(f"解析后的对象类型: {type(parsed_result)}")
    print(f"解析后的对象: {parsed_result}")
    print()
    print(f"翻译：{parsed_result.translation}")
    print(f"质量评分：{parsed_result.quality_score}/10")
    print(f"说明：{parsed_result.notes}")
    print()
    
    # 总结
    print("=" * 80)
    print("数据流总结")
    print("=" * 80)
    print("""
数据流动路径：
1. Dict[str, Any] (输入字典)
   ↓ prompt.invoke()
2. ChatPromptValue (包含消息列表)
   ↓ llm.invoke()
3. AIMessage (LLM 响应对象)
   ↓ parser.invoke()
4. TranslationResult (Pydantic 模型实例)
    """)


if __name__ == "__main__":
    debug_data_flow()
