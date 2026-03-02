from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from vectorizer import VectorStoreManager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import importlib.util
from pathlib import Path

# 动态导入 langchain-test 目录中的模块（目录名带连字符，不能直接导入）
parent_dir = Path(__file__).parent.parent
base_def_path = parent_dir / "langchain-test" / "baseDef.py"

if not base_def_path.exists():
    raise ImportError(f"找不到 baseDef.py 文件: {base_def_path}")

spec = importlib.util.spec_from_file_location("baseDef", base_def_path)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载 baseDef 模块: {base_def_path}")

baseDef = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseDef)


class ConversationalRAG:
    """支持多轮对话的 RAG 系统"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        
        # 确保向量存储已加载
        if self.vector_manager.vectorstore is None:
            self.vector_manager.load_vectorstore()
        
        # 创建 Memory
        self.memory = ConversationBufferMemory(
            k=10,
            memory_key="chat_history",  # 在 Prompt 中的变量名
            return_messages=True,
            output_key="answer",
        )

        # 创建 LLM
        self.llm = baseDef.createLlm()
        
        # 创建检索器
        self.retriever = self.vector_manager.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # 创建对话式检索链
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )
    
    def ask(self, question: str) -> Dict:
        """提问（支持上下文）"""
        print(f"\n💬 问题: {question}")
        print("🔍 正在检索相关文档...")
        
        # 执行查询
        result = self.chain({"question": question})
        
        # 显示检索到的文档
        print(f"📚 参考了 {len(result['source_documents'])} 个文档片段")
        
        return {
            "question": question,
            "answer": result["answer"],
            "sources": [
                {
                    "source": doc.metadata.get('source', 'unknown'),
                    "content": doc.page_content
                }
                for doc in result["source_documents"]
            ]
        }
    
    def chat(self):
        """交互式对话"""
        print("\n" + "=" * 80)
        print("💬 多轮对话 RAG 系统")
        print("=" * 80)
        print("\n💡 提示：")
        print("  - 可以进行连续提问，系统会记住上下文")
        print("  - 输入 /clear 清除对话历史")
        print("  - 输入 /quit 退出")
        print("=" * 80)
        
        while True:
            try:
                user_input = input("\n💬 你: ").strip()
                
                if not user_input:
                    continue
                
                if user_input == "/quit":
                    print("👋 再见！")
                    break
                
                if user_input == "/clear":
                    self.memory.clear()
                    print("🗑️  对话历史已清除")
                    continue
                
                # 提问
                result = self.ask(user_input)
                print(f"\n🤖 答案: {result['answer']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

# 测试
if __name__ == "__main__":
    vector_manager = VectorStoreManager()
    vector_manager.load_vectorstore()
    
    conv_rag = ConversationalRAG(vector_manager)
    conv_rag.chat()