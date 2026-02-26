import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Literal
from dotenv import load_dotenv

# 1. 定义翻译风格枚举
TranslationStyle = Literal["formal", "casual", "technical"]
os.environ["OPENAI_API_KEY"] = "GSkkITz67tvyRj0eZncG"
os.environ["OPENAI_API_BASE"] = "https://genaiapi.cloudsway.net/v1/ai/ZoDFMXKuoyxNhQjg" 

load_dotenv()

# 2. 单条翻译结果
class SingleTranslation(BaseModel):
    original: str = Field(description="原始中文")
    translation: str = Field(description="英文翻译")
    quality_score: int = Field(description="质量评分1-10")

# 3. 批量翻译结果
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

# 4. 创建解析器
parser = PydanticOutputParser(pydantic_object=BatchTranslationResult)

# 5. 创建动态 Prompt
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

# 创建 LLM
llm = ChatOpenAI(model="MaaS 3.7 Sonnet", temperature=0)

# 6. 创建 Chain
# 如果某一句翻译失败（如 LLM 未能输出有效的 JSON 或漏译），parser 解析时会直接抛出异常（如 ValidationError），
# 这是“全-or-无（all-or-nothing）”模式：只要有一句不合规，整个 chain.invoke() 会报错，不返回部分翻译结果。
chain = prompt | llm | parser

# 7. 测试函数
def translate_batch(texts: List[str], style: TranslationStyle = "formal"):
    """批量翻译函数"""
    # 将列表格式化为带编号的文本
    formatted_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    print(f"编号后的内容: {formatted_texts} \n")
    result = chain.invoke({
        "texts": formatted_texts,
        "style": style,
        "format_instructions": parser.get_format_instructions()
    })
    
    return result

# 8. 测试
if __name__ == "__main__":
    # test_texts = [
    #     "人工智能正在改变世界",
    #     "这个算法的时间复杂度是O(n)",
    #     "今天天气真好",
    #     "这是一个超级超级超级超级超级超级长的句子" * 50  # 故意制造问题
    # ]
    
    # print("=" * 60)
    # print("正式风格翻译：")
    # print("=" * 60)
    # try:
    #     result_formal = translate_batch(test_texts, style="formal")
    #     print(f"成功翻译 {len(result.translations)} 条")
    # except Exception as e:
    #     print(f"正式风格翻译失败: {e}")
    #     print("-" * 40)
    #     print(f"平均质量：{result_formal.average_quality}/10\n")

    #     for text in test_texts[:3]:  # 只处理前3条
    #         try:
    #             result = translate_batch([text], style="formal")
    #             print(f"✅ {text[:20]}... → {result.translations[0].translation[:30]}...")
    #         except Exception as e:
    #             print(f"❌ {text[:20]}... 翻译失败")
    
    # print("=" * 60)
    # print("口语风格翻译：")
    # print("=" * 60)
    # result_casual = translate_batch(test_texts, style="casual")
    # for trans in result_casual.translations:
    #     print(f"原文：{trans.original}")
    #     print(f"译文：{trans.translation}")
    #     print(f"评分：{trans.quality_score}/10")
    #     print("-" * 40)
    # print(f"平均质量：{result_casual.average_quality}/10")

    test_texts = ["人工智能正在改变世界"]
    
    # 第一次调用
    start = time.time()
    result1 = translate_batch(test_texts, style="formal")
    time1 = time.time() - start
    
    # 第二次调用（相同输入）
    start = time.time()
    result2 = translate_batch(test_texts, style="formal")
    time2 = time.time() - start
    
    print(f"第一次耗时：{time1:.2f}秒")
    print(f"第二次耗时：{time2:.2f}秒（使用缓存）")
    print(f"加速比：{time1/time2:.1f}x")