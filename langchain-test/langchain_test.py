import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache


# 定义输出结构
class TranslationResult(BaseModel):
    """翻译结果的结构"""
    translation: str = Field(description="英文翻译")
    quality_score: int = Field(description="翻译质量评分（1-10）")
    notes: str = Field(description="翻译说明或注意事项")

def createLlm():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "GSkkITz67tvyRj0eZncG"
    os.environ["OPENAI_API_BASE"] = "https://genaiapi.cloudsway.net/v1/ai/ZoDFMXKuoyxNhQjg" 
    return ChatOpenAI(model="MaaS 3.7 Sonnet", temperature=0)

def createPrompt():
    return ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译专家，擅长将中文翻译成地道的英文。"),
    ("human", """请将以下中文翻译成英文，并评估翻译质量。

{format_instructions}

中文文本：
{text}
""")
])  

def main():
    llm = createLlm()

    # 创建 Prompt（注入格式说明）
    prompt = createPrompt()
  # 创建输出解析器
    parser = PydanticOutputParser(pydantic_object=TranslationResult)
      
    # set_llm_cache(InMemoryCache())
    # 组合成 Chain
    # chain = prompt | llm | parser
    chain = prompt | llm 

    print("=" * 50)
    print("Parser 生成的格式指令：")
    print(parser.get_format_instructions())
    print("=" * 50)


    result = chain.invoke({
        "text": "人工智能正在改变世界",
        "format_instructions": parser.get_format_instructions()
    })
    # print(f"翻译：{result.translation}")
    # print(f"质量评分：{result.quality_score}/10")
    # print(f"说明：{result.notes}")
    print(result)


if __name__ == "__main__":
    main()