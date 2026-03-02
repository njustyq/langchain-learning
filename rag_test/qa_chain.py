import sys
import importlib.util
from pathlib import Path
from typing import List, Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from vectorizer import VectorStoreManager
from pydantic import BaseModel, Field

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

    def ask_with_confidence(self, question: str, min_score: float = 0.7) -> Dict:
        """带置信度筛选的提问"""
        # 获取带分数的检索结果
        results = self.vector_manager.vectorstore.similarity_search_with_score(
            question, k=5
        )
        
        # 筛选高相关度的文档
        filtered_docs = [
            doc for doc, score in results if score < min_score  # 注意：分数越小越相似
        ]
        
        if not filtered_docs:
            return {
                "question": question,
                "answer": "抱歉，我在文档中没有找到足够相关的信息来回答这个问题。",
                "confidence": "low",
                "sources": []
            }
        
        # 格式化上下文
        context = self._format_docs(filtered_docs)
        
        # 生成答案
        answer = self.chain.invoke(question)
        
        confidence = "high" if len(filtered_docs) >= 3 else "medium"
        
        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "sources": [
                {
                    "source": doc.metadata.get('source'),
                    "content": doc.page_content,
                    "score": score
                }
                for doc, score in results[:len(filtered_docs)]
            ]
        }

class RelevanceScore(BaseModel):
    """相关性评分"""
    chunk_id: int = Field(description="文档块ID")
    score: int = Field(description="相关性评分 1-10")
    reason: str = Field(description="评分理由")

class ReRanker:
    """文档重排序器"""
    
    def __init__(self):
        self.llm = baseDef.createLlm()
        self.parser = JsonOutputParser(pydantic_object=RelevanceScore)
        self.prompt = baseDef.createRAGPromptWithReranking()        

    def rerank(self, question: str, documents: List) -> List:
        """对文档进行重排序"""
        scored_docs = []
        
        print(f"\n🔄 正在对 {len(documents)} 个文档进行重排序...")
        
        for i, doc in enumerate(documents):
            try:
                result = (self.prompt | self.llm | self.parser).invoke({
                    "question": question,
                    "document": doc.page_content[:500],  # 只取前500字符
                    "format_instructions": self.parser.get_format_instructions()
                })
                
                scored_docs.append({
                    "doc": doc,
                    "score": result["score"],
                    "reason": result["reason"]
                })
                
                print(f"  文档 {i+1}: {result['score']}/10 - {result['reason'][:50]}...")
                
            except Exception as e:
                print(f"  文档 {i+1}: 评分失败 - {e}")
                scored_docs.append({
                    "doc": doc,
                    "score": 5,  # 默认中等分数
                    "reason": "评分失败"
                })
        
        # 按分数降序排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        print(f"✅ 重排序完成\n")
        
        return [item["doc"] for item in scored_docs]

class RAGWithReranking(RAGQASystem):
    """带重排序的 RAG 系统"""
    
    def __init__(self, vector_manager: VectorStoreManager):
        super().__init__(vector_manager)
        self.reranker = ReRanker()
    
    def ask(self, question: str, use_reranking: bool = True) -> str:
        """提问（可选重排序）"""
        print(f"\n💬 问题: {question}")
        print("🔍 正在检索相关文档...")
        
        # 初始检索（多检索一些）
        docs = self.retriever.get_relevant_documents(question)
        print(f"📚 初步检索到 {len(docs)} 个文档片段")
        
        # 重排序
        if use_reranking:
            docs = self.reranker.rerank(question, docs)
            docs = docs[:3]  # 只保留前3个
        
        # 格式化上下文
        context = self._format_docs(docs)
        
        # 生成答案
        print("🤖 正在生成答案...\n")
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        return answer

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