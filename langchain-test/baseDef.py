import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import BaseMessage

# 在模块导入时即执行，确保所有 import 本文件的地方都自动加载环境变量和配置
def _ensure_env_loaded():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "GSkkITz67tvyRj0eZncG"
    os.environ["OPENAI_API_BASE"] = "https://genaiapi.cloudsway.net/v1/ai/ZoDFMXKuoyxNhQjg" 

_ensure_env_loaded()

def createLlm():
       return ChatOpenAI(model="MaaS 3.7 Sonnet", temperature=0)   

def createSmartLlm(handler):
    return ChatOpenAI(
        model="MaaS 3.7 Sonnet", 
        temperature=0.3,
        streaming=True,
        callbacks=[handler]
        )

def createSingleTranslationPrompt():
    return ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译专家，擅长将中文翻译成地道的英文。"),
    ("human", """请将以下中文翻译成英文，并评估翻译质量。

{format_instructions}

中文文本：
{text}
""")
])  

def createBatchTranslationPrompt():
    return ChatPromptTemplate.from_messages([
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

def createTranslatorWithMemoryPrompt():
    """
    MessagesPlaceholder vs 普通变量 {input} 的区别：
    
    1. **数据类型不同**：
       - MessagesPlaceholder: 接收的是消息对象列表 (List[BaseMessage])
         例如: [HumanMessage("你好"), AIMessage("Hello!")]
       - {input}: 接收的是普通字符串 (str)
         例如: "你好"
    
    2. **在 Prompt 中的位置不同**：
       - MessagesPlaceholder: 可以插入到消息序列的任意位置，作为完整的消息对象
       - {input}: 只能作为单个消息的内容部分
    
    3. **用途不同**：
       - MessagesPlaceholder: 用于插入多轮对话历史，保持对话的上下文和角色信息
         模型能看到完整的对话历史，包括谁说了什么
       - {input}: 用于插入当前用户输入的文本内容
    
    4. **格式化方式不同**：
       - MessagesPlaceholder: LangChain 会将消息列表直接插入到消息序列中
         保持消息的 role (human/ai/system) 和 content 结构
       - {input}: 只是简单的字符串替换，插入到 ("human", "{input}") 的 content 部分
    
    5. **实际效果示例**：
       使用 MessagesPlaceholder 时，发送给模型的完整消息可能是：
       [
         SystemMessage("你是专业翻译助手..."),
         HumanMessage("我喜欢口语化翻译"),  # 来自 chat_history
         AIMessage("好的，我会用口语化风格"),  # 来自 chat_history
         HumanMessage("翻译：你好")  # 当前 input
       ]
       
       如果只用 {input}，模型只能看到：
       [
         SystemMessage("你是专业翻译助手..."),
         HumanMessage("翻译：你好")  # 只有当前输入，没有历史
       ]
    """
    return ChatPromptTemplate.from_messages([
    ("system", """你是专业翻译助手。你会记住用户的偏好：
- 如果用户之前选择过某种风格，优先使用该风格
- 如果用户提到特定领域（如技术、商务），自动调整术语
- 保持翻译风格的一致性
"""),
    # MessagesPlaceholder: 插入历史对话消息列表，保持完整的对话上下文
    # 变量 chat_history 必须是一个消息对象列表，如 [HumanMessage(...), AIMessage(...)]
    MessagesPlaceholder(variable_name="chat_history"),
    # {input}: 普通字符串变量，插入当前用户输入
    # 变量 input 是一个字符串，如 "翻译：你好"
    ("human", "{input}")
])


def createSmartTranslatorPrompt():
    return ChatPromptTemplate.from_messages([
    ("system", """你是高级翻译助手，具备以下能力：

1. **偏好记忆**：记住用户喜欢的翻译风格（正式/口语/技术）
2. **上下文理解**：根据之前的对话调整翻译
3. **主动建议**：如果发现用户可能需要特定风格，主动询问

当前对话摘要：
{chat_history}
"""),
    ("human", "{input}")
])

class LLMWithTokenCounter(ChatOpenAI):
    """包装 ChatOpenAI 以提供 token 计数功能"""
    def get_num_tokens_from_messages(self, messages):
        """
        估算消息的 token 数量
        使用简单的字符数估算方法（约 4 个字符 = 1 token）
        对于中文，约 1.5 个字符 = 1 token
        """
        total_chars = 0
        for message in messages:
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, str):
                    # 简单估算：中文字符按 1.5 字符/token，其他按 4 字符/token
                    chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
                    other_chars = len(content) - chinese_chars
                    # 估算 token 数
                    estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
                    total_chars += estimated_tokens
                elif isinstance(content, list):
                    # 处理内容为列表的情况（如包含图片等）
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text = item['text']
                            chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
                            other_chars = len(text) - chinese_chars
                            estimated_tokens = int(chinese_chars / 1.5 + other_chars / 4)
                            total_chars += estimated_tokens
        # 添加一些额外的 token 用于消息格式（role, 分隔符等）
        return total_chars + len(messages) * 3

def createSmartTranslatorMemory():
    # 使用包装后的 LLM，提供 token 计数功能
    llm_with_counter = LLMWithTokenCounter(
        model="MaaS 3.7 Sonnet", 
        temperature=0
    )
    return ConversationSummaryBufferMemory(
        llm=llm_with_counter,
        max_token_limit=500,  # 超过500 tokens时自动总结
        return_messages=True,
        memory_key="chat_history"
    )