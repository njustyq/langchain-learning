from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
from baseDef import createSmartLlm, createSmartTranslatorPrompt, createSmartTranslatorMemory



# 1. 使用更高级的 Memory：ConversationSummaryBufferMemory
# 它会自动总结旧对话，只保留摘要
memory = createSmartTranslatorMemory()

# 2. 更智能的系统提示
prompt = createSmartTranslatorPrompt()

# 3. 流式回调
class SmartStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.text = ""
        self.token_count = 0
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.token_count += 1
        print(token, end="", flush=True)
    
    def on_llm_end(self, response, **kwargs):
        print(f"\n[Token 数: {self.token_count}]")

# 4. 创建 Chain
def create_chain():
    handler = SmartStreamHandler()
    llm = createSmartLlm(handler)
    chain = prompt | llm
    return chain, handler

# 5. 智能对话
def smart_chat():
    print("=" * 60)
    print("🧠 智能翻译助手 Pro")
    print("💡 我会记住你的偏好，并在长对话中自动总结历史")
    print("=" * 60)
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            # 退出前显示对话摘要
            print("\n📝 对话摘要：")
            print(memory.load_memory_variables({})["chat_history"])
            break
        
        if not user_input:
            continue
        
        chain, handler = create_chain()
        
        print("\n助手: ", end="", flush=True)
        response = chain.invoke({
            "input": user_input,
            "chat_history": memory.load_memory_variables({})["chat_history"]
        })
        
        # 保存（会自动触发总结）
        memory.save_context(
            {"input": user_input},
            {"output": handler.text}
        )

if __name__ == "__main__":
    smart_chat()