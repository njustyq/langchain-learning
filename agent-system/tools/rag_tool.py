from langchain_core.tools import Tool
from pathlib import Path
import sys

# 添加父目录到路径（假设 RAG 系统在上级目录）
# sys.path.append(str(Path(__file__).parent.parent.parent /"langchain-test" ))
sys.path.append(str(Path(__file__).parent.parent.parent /"rag_test"))
# parent_dir = Path(__file__).parent.parent
# base_def_path = parent_dir / "langchain-test" / "baseDef.py"
def query_documents(question: str) -> str:
    """查询本地文档库
    
    此函数会被 Tool.run() 方法调用。
    调用路径：rag_tool.run(question) → Tool.run() → self.func(question) → query_documents(question)
    
    Args:
        question: 问题
    
    Returns:
        基于文档的答案
    """
    try:
        from vectorizer import VectorStoreManager
        from qa_chain import RAGQASystem
        from document_loader import DocumentLoader
        from text_splitter import SmartTextSplitter

        # 1. 加载文档
        print("\n[步骤 1/4] 加载文档...")
        loader = DocumentLoader()
        documents = loader.load_directory()
        print(f"[OK] 成功加载 {len(documents)} 个文档")

        # 2. 切分文档
        print("\n[步骤 2/4] 切分文档...")
        splitter = SmartTextSplitter()
        chunks = splitter.split_documents(documents)
        print(f"[OK] 成功切分为 {len(chunks)} 个文档块")
        
        # 3. 加载向量存储
        vector_manager = VectorStoreManager()
        vector_manager.create_vectorstore(chunks)
        print("[OK] 向量存储创建成功！")

        # 4. 创建 QA 系统
        qa_system = RAGQASystem(vector_manager)
        
        # 5. 查询
        result = qa_system.ask_with_sources(question)
        
        # 6. 格式化返回
        answer = result["answer"]
        sources = [s["source"] for s in result["sources"]]
        
        return f"{answer}\n\n[来源: {', '.join(sources)}]"
        
    except Exception as e:
        return f"错误：{str(e)}"

# ============================================================================
# 调用链说明：
# ============================================================================
# 当执行 rag_tool.run("什么是 LangChain？") 时，调用链如下：
#
#   1. rag_tool.run("什么是 LangChain？")
#        ↓
#   2. Tool.run() 方法内部执行：
#        self.func("什么是 LangChain？")
#        ↓
#   3. 因为创建 Tool 时传入了 func=query_documents（函数引用，不是函数调用）
#        ↓
#   4. query_documents("什么是 LangChain？") 被调用
#        ↓
#   5. 执行 query_documents 函数内的所有逻辑：
#        - 加载文档
#        - 切分文档
#        - 创建向量存储
#        - 创建 RAGQASystem
#        - 调用 qa_system.ask_with_sources(question)
#        - 格式化返回结果
#        ↓
#   6. 返回结果字符串给 Tool.run()
#        ↓
#   7. Tool.run() 返回结果给调用者
#
# 关键点：
#   - func=query_documents 是传入函数引用（函数对象），不是函数调用
#   - Tool 类内部保存这个函数引用为 self.func
#   - 当调用 run() 时，Tool 内部会执行 self.func(input)
# ============================================================================

# 创建 LangChain Tool
rag_tool = Tool(
    name="document_query",
    func=query_documents,  # 传入函数引用，不是函数调用（注意：没有括号）
    description="""查询本地文档库中的信息。

适用场景：
- 查询已上传的文档内容
- 获取公司内部知识
- 查找技术文档

输入格式：具体的问题
输出格式：基于文档的答案 + 来源标注

使用示例：
- 输入："LangChain 的核心组件有哪些？"
- 输出："LangChain 的核心组件包括... [来源: langchain_intro.txt]"

注意：
1. 只能查询已上传到文档库的内容
2. 问题要具体明确
3. 如果文档库为空，会返回错误提示"""
)

# 测试
if __name__ == "__main__":
    print("测试 RAG 工具：")
    # 调用链：rag_tool.run() → Tool.run() → query_documents()
    result = rag_tool.run("什么是 LangChain？")
    print(result)