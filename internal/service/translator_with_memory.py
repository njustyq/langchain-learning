from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from baseDef import createLlm, createTranslatorWithMemoryPrompt

# 1. 创建 Memory（存储对话历史）
memory = ConversationBufferMemory(
    return_messages=True,  # 返回消息对象而非字符串
    memory_key="chat_history"  # 在 Prompt 中的变量名
)

# 2. 创建带 Memory 的 Prompt
prompt = createTranslatorWithMemoryPrompt()

# 3. 创建 LLM
llm = createLlm()

# 4. 组合成带 Memory 的 Chain
# 使用 RunnablePassthrough.assign 来添加 chat_history
# chat_history 应该是一个消息列表，而不是字典
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: memory.load_memory_variables({})["chat_history"]
    )
    | prompt
    | llm
)

# 7. 交互式测试
def chat():
    """交互式对话函数"""
    print("=" * 60)
    print("🌐 智能翻译助手（输入 'quit' 退出）")
    print("=" * 60)
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("再见！👋")
            break
        
        if not user_input:
            continue
        
        # 调用 Chain
        response = chain.invoke({"input": user_input})
        
        # 保存到 Memory
        memory.save_context(
            {"input": user_input},
            {"output": response.content}
        )
        
        print(f"\n助手: {response.content}")

if __name__ == "__main__":
    chat()  