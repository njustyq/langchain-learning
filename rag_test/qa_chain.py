import sys
import importlib.util
from pathlib import Path
from typing import List, Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vectorizer import VectorStoreManager

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

createLlm = baseDef.createLlm
createRAGPrompt = baseDef.createRAGPrompt

class RAGQASystem:
    """RAG 问答系统"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        self.vector_manager = vector_manager
        
        # 确保向量存储已加载
        if self.vector_manager.vectorstore is None:
            self.vector_manager.load_vectorstore()
        
        # 赋给局部变量，让 Pylance 能正确收窄类型（Optional[Chroma] -> Chroma）
        vectorstore = self.vector_manager.vectorstore
        if vectorstore is None:
            raise ValueError("向量存储加载失败，请先创建向量存储")
        
        # 创建检索器
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 检索前3个最相关的文档
        )
        
        # 创建 LLM
        self.llm = createLlm()
        
        # 创建 Prompt
        self.prompt = createRAGPrompt()
        
        # 构建 RAG Chain
        self.chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs: List) -> str:
        """格式化检索到的文档"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'unknown')
            formatted.append(f"[文档 {i} - 来源: {source}]\n{doc.page_content}\n")
        return "\n".join(formatted)
    
    def ask(self, question: str) -> str:
        """提问"""
        print(f"\n[问题] {question}")
        print("[检索] 正在检索相关文档...")
        
        # 先显示检索到的文档（使用 invoke 方法）
        docs = self.retriever.invoke(question)
        print(f"[结果] 找到 {len(docs)} 个相关文档片段\n")
        
        for i, doc in enumerate(docs, 1):
            print(f"--- 片段 {i} ---")
            print(f"来源: {doc.metadata.get('source', 'unknown')}")
            print(f"内容: {doc.page_content[:100]}...")
            print()
        
        # 生成答案
        print("[生成] 正在生成答案...\n")
        answer = self.chain.invoke(question)
        
        return answer
    
    def ask_with_sources(self, question: str) -> Dict:
        """提问并返回来源信息"""
        docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        
        sources = [
            {
                "source": doc.metadata.get('source', 'unknown'),
                "chunk_id": doc.metadata.get('chunk_id', -1),
                "content": doc.page_content
            }
            for doc in docs
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }

# 测试
if __name__ == "__main__":
    # 加载向量存储（使用与创建时相同的 embedding_type）
    # 注意：如果之前使用的是 huggingface，这里也要用 huggingface
    vector_manager = VectorStoreManager(embedding_type="huggingface")
    vector_manager.load_vectorstore()
    
    # 创建问答系统
    qa_system = RAGQASystem(vector_manager)
    
    # 测试问题
    test_questions = [
        "什么是 LangChain？",
        "LangChain 有哪些核心组件？",
        "如何使用 LangChain 构建应用？"
    ]
    
    for question in test_questions:
        answer = qa_system.ask(question)
        print(f"[答案] {answer}\n")
        print("=" * 80 + "\n")