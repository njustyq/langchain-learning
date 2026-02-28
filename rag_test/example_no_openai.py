"""
使用免费的 HuggingFace Embeddings 示例
适用于没有 OpenAI API key 的情况
"""

from document_loader import DocumentLoader
from text_splitter import SmartTextSplitter
from vectorizer import VectorStoreManager

def main():
    print("=" * 60)
    print("RAG 系统示例 - 使用免费的 HuggingFace Embeddings")
    print("=" * 60)
    
    # 1. 加载文档
    print("\n[步骤 1/4] 加载文档...")
    loader = DocumentLoader()
    documents = loader.load_directory()
    print(f"✓ 成功加载 {len(documents)} 个文档")
    
    # 2. 切分文档
    print("\n[步骤 2/4] 切分文档...")
    splitter = SmartTextSplitter()
    chunks = splitter.split_documents(documents)
    print(f"✓ 成功切分为 {len(chunks)} 个文档块")
    
    # 3. 创建向量存储（使用免费的 HuggingFace）
    print("\n[步骤 3/4] 创建向量存储...")
    print("⚠️  首次运行会下载模型（约 470MB），请耐心等待...")
    
    vector_manager = VectorStoreManager(
        persist_directory="./vector_store",
        embedding_type="huggingface"  # 使用免费的 HuggingFace
    )
    
    vector_manager.create_vectorstore(chunks)
    print("✓ 向量存储创建成功！")
    
    # 4. 测试搜索
    print("\n[步骤 4/4] 测试搜索功能...")
    
    # 示例查询
    queries = [
        "什么是 LangChain？",
        "如何使用文档加载器？",
        "RAG 是什么？"
    ]
    
    for query in queries:
        print("\n" + "=" * 60)
        vector_manager.similarity_search_with_score(query, k=2)
    
    print("\n" + "=" * 60)
    print("✓ 所有步骤完成！")
    print("\n💡 提示：")
    print("  - 向量存储已保存到 ./vector_store 目录")
    print("  - 下次运行时使用 vector_manager.load_vectorstore() 可直接加载")
    print("  - 完全免费，无需任何 API key！")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"\n❌ 缺少依赖包: {e}")
        print("\n请运行以下命令安装：")
        print("pip install langchain-huggingface sentence-transformers")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
