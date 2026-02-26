import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks import StdOutCallbackHandler
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

# 单条翻译结果
class SingleTranslation(BaseModel):
    original: str = Field(description="原始中文")
    translation: str = Field(description="英文翻译")
    quality_score: int = Field(description="质量评分1-10")

# 批量翻译结果
class BatchTranslationResult(BaseModel):
    # Q: List[SingleTranslation] 这种嵌套结构，Parser 能正确处理吗？
    # A: 是的，LangChain 的 PydanticOutputParser 支持这种嵌套结构。LLM 根据 format_instructions 指令，通常能按照 Pydantic 模型定义嵌套生成符合要求的 JSON。

    # Q: LLM 会输出什么样的 JSON？
    # A: 例如，如果 style="formal" 且有3个句子输入，则 LLM 输出示例可能如下：
    # {
    #   "style": "formal",
    #   "translations": [
    #     {"original": "人工智能正在改变世界", "translation": "Artificial intelligence is changing the world.", "quality_score": 9},
    #     {"original": "这个算法的时间复杂度是O(n)", "translation": "The time complexity of this algorithm is O(n).", "quality_score": 10},
    #     {"original": "今天天气真好", "translation": "The weather is great today.", "quality_score": 8}
    #   ],
    #   "average_quality": 9.0
    # }
    style: str = Field(description="使用的翻译风格")
    translations: List[SingleTranslation] = Field(description="翻译列表")
    average_quality: float = Field(description="平均质量分")

def createChainWithDebug(parser):
    # 1. 定义翻译风格枚举
    TranslationStyle = Literal["formal", "casual", "technical"]
    # 2. 设置环境变量
    os.environ["OPENAI_API_KEY"] = "GSkkITz67tvyRj0eZncG"
    os.environ["OPENAI_API_BASE"] = "https://genaiapi.cloudsway.net/v1/ai/ZoDFMXKuoyxNhQjg" 
    
    load_dotenv()

    # 3. 创建动态 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业翻译专家。翻译风格说明：
    - formal: 正式、书面语，适合商务文档
    - casual: 口语化、轻松，适合日常对话
    - technical: 专业术语准确，适合技术文档
    """),
        ("human", """请使用 {style} 风格翻译以下中文句子。

    {format_instructions}

    待翻译内容：
    {texts}
    """)
    ])
        
    # 5. 创建 LLM
    llm = ChatOpenAI(model="MaaS 3.7 Sonnet", temperature=0)
    
    return prompt | llm | parser

if __name__ == "__main__":
     # 创建解析器
    parser = PydanticOutputParser(pydantic_object=BatchTranslationResult)
    chain_with_debug = createChainWithDebug(parser)
    result = chain_with_debug.invoke(
        {
            "texts": "人工智能正在改变世界",
            "style": "formal",
            "format_instructions": parser.get_format_instructions()
        },
        config={"callbacks": [StdOutCallbackHandler()]}
    )
    print(result)