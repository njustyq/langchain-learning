import argparse
from pathlib import Path
from document_loader import DocumentLoader
from text_splitter import SmartTextSplitter
from vectorizer import VectorStoreManager
from qa_chain import RAGQASystem

class RAGApplication:
    """RAG 应用主程序"""
    
    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = SmartTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vector_manager = VectorStoreManager()
        self.qa_system = None
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n" + "=" * 80)
        print("📚 构建知识库")
        print("=" * 80)
        
        # 1. 加载文档
        print("\n[步骤 1/3] 加载文档...")
        documents = self.loader.load_directory()
        
        if not documents:
            print("❌ 没有找到文档，请将文档放入 ./documents 目录")
            return False
        
        # 2. 切分文档
        print("\n[步骤 2/3] 切分文档...")
        chunks = self.splitter.split_documents(documents)
        
        # 3. 向量化
        print("\n[步骤 3/3] 向量化并存储...")
        self.vector_manager.create_vectorstore(chunks)
        
        print("\n✅ 知识库构建完成！")
        return True
    
    def add_documents(self, file_paths: list):
        """添加新文档"""
        print("\n" + "=" * 80)
        print("➕ 添加新文档")
        print("=" * 80)
        
        all_chunks = []
        for file_path in file_paths:
            # 加载单个文件
            documents = self.loader.load_single_file(file_path)
            # 切分
            chunks = self.splitter.split_documents(documents)
            all_chunks.extend(chunks)
        
        # 添加到向量存储
        self.vector_manager.add_documents(all_chunks)
        print("\n✅ 文档添加完成！")
    
    def start_qa_session(self):
        """启动问答会话"""
        print("\n" + "=" * 80)
        print("💬 RAG 问答系统")
        print("=" * 80)
        
        # 加载向量存储
        if self.vector_manager.load_vectorstore() is None:
            print("❌ 知识库不存在，请先运行 'python main.py build' 构建知识库")
            return
        
        # 创建问答系统
        self.qa_system = RAGQASystem(self.vector_manager)
        
        print("\n💡 提示：")
        print("  - 输入问题进行提问")
        print("  - 输入 /sources 查看上次答案的来源")
        print("  - 输入 /search <关键词> 搜索文档")
        print("  - 输入 /quit 退出")
        print("=" * 80)
        
        last_result = None
        
        while True:
            try:
                user_input = input("\n💬 你: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                if user_input == "/quit":
                    print("👋 再见！")
                    break
                
                elif user_input == "/sources":
                    if last_result:
                        print("\n📚 答案来源：")
                        for i, source in enumerate(last_result["sources"], 1):
                            print(f"\n[来源 {i}]")
                            print(f"文件: {source['source']}")
                            print(f"内容: {source['content'][:200]}...")
                    else:
                        print("⚠️  还没有提问过")
                    continue
                
                elif user_input.startswith("/search"):
                    keyword = user_input.replace("/search", "").strip()
                    if keyword:
                        self.vector_manager.similarity_search_with_score(keyword, k=5)
                    else:
                        print("⚠️  请提供搜索关键词")
                    continue
                
                # 正常提问
                last_result = self.qa_system.ask_with_sources(user_input)
                print(f"\n🤖 答案: {last_result['answer']}")
                print(f"\n📎 参考了 {len(last_result['sources'])} 个文档片段")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG 文档问答系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # build 命令
    subparsers.add_parser("build", help="构建知识库")
    
    # add 命令
    add_parser = subparsers.add_parser("add", help="添加新文档")
    add_parser.add_argument("files", nargs="+", help="文档路径")
    
    # ask 命令
    ask_parser = subparsers.add_parser("ask", help="单次提问")
    ask_parser.add_argument("question", help="问题")
    
    # chat 命令
    subparsers.add_parser("chat", help="启动交互式问答")
    
    # search 命令
    search_parser = subparsers.add_parser("search", help="搜索文档")
    search_parser.add_argument("keyword", help="搜索关键词")
    
    args = parser.parse_args()
    
    app = RAGApplication()
    
    if args.command == "build":
        app.build_knowledge_base()
    
    elif args.command == "add":
        app.add_documents(args.files)
    
    elif args.command == "ask":
        app.vector_manager.load_vectorstore()
        qa_system = RAGQASystem(app.vector_manager)
        result = qa_system.ask_with_sources(args.question)
        print(f"\n🤖 答案: {result['answer']}")
    
    elif args.command == "chat":
        app.start_qa_session()
    
    elif args.command == "search":
        app.vector_manager.load_vectorstore()
        app.vector_manager.similarity_search_with_score(args.keyword, k=5)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()