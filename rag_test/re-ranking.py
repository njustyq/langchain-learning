from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

# 集成到 RAG 系统
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