from langchain_chroma import Chroma  
from langchain_core.documents import Document  
from typing import List, Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class VectorStoreManager:
    """向量存储管理器"""
    
    def __init__(self, 
                 persist_directory: str = "./vector_store",
                 collection_name: str = "documents",
                 embedding_type: Literal["openai", "huggingface", "ollama"] = "huggingface"):
        """
        初始化向量存储管理器
        
        Args:
            persist_directory: 向量存储目录
            collection_name: 集合名称
            embedding_type: embedding 类型
                - "openai": 使用 OpenAI (需要 API key)
                - "huggingface": 使用 HuggingFace 本地模型 (免费，推荐)
                - "ollama": 使用 Ollama (需要先安装 Ollama)
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        self.collection_name = collection_name
        self.embedding_type = embedding_type
        
        # 创建 Embedding 模型
        self.embeddings = self._create_embeddings(embedding_type)
        
        self.vectorstore: Optional[Chroma] = None
    
    def _create_embeddings(self, embedding_type: str):
        """创建 Embedding 模型"""
        if embedding_type == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings 
                print("[Embedding] 使用 OpenAI Embeddings (需要 API key)")
                return OpenAIEmbeddings(model="text-embedding-ada-002")
            except ImportError:
                raise ImportError("请安装: pip install langchain-openai")
                
        elif embedding_type == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings 
                print("[Embedding] 使用 HuggingFace 本地模型（免费）")
                print("[提示] 首次使用会自动下载模型，请稍候...")
                return HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except ImportError:
                raise ImportError("请安装: pip install langchain-huggingface sentence-transformers")
                
        elif embedding_type == "ollama":
            try:
                from langchain_ollama import OllamaEmbeddings  # pyright: ignore[reportMissingImports]
                print("[Embedding] 使用 Ollama Embeddings (需要先安装 Ollama)")
                return OllamaEmbeddings(model="nomic-embed-text")
            except ImportError:
                raise ImportError("请安装: pip install langchain-ollama")
        else:
            raise ValueError(f"不支持的 embedding 类型: {embedding_type}")
    
    def create_vectorstore(self, chunks: List[Document]) -> Chroma:
        """创建向量存储"""
        print(f"\n[向量化] 开始向量化 {len(chunks)} 个文档块...")
        print("[等待] 这可能需要几分钟，请耐心等待...")
        
        # 创建向量存储
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory)
        )
        
        print(f"[完成] 向量化完成！")
        print(f"[保存] 已保存到：{self.persist_directory}")
        
        return self.vectorstore
    
    def load_vectorstore(self) -> Optional[Chroma]:
        """加载已存在的向量存储"""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            
            # 检查是否有数据
            count = self.vectorstore._collection.count()
            print(f"[加载] 已加载向量存储：{count} 个文档块")
            
            return self.vectorstore
        except Exception as e:
            print(f"[错误] 加载失败：{e}")
            return None
    
    def add_documents(self, chunks: List[Document]):
        """向现有向量存储添加文档"""
        if self.vectorstore is None:
            self.vectorstore = self.load_vectorstore()
        
        if self.vectorstore is None:
            print("[警告] 向量存储不存在，将创建新的")
            return self.create_vectorstore(chunks)
        
        print(f"[添加] 添加 {len(chunks)} 个新文档块...")
        self.vectorstore.add_documents(chunks)
        print("[完成] 添加完成")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """相似度搜索"""
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        print(f"\n[查询] 查询：{query}")
        print(f"[结果] 找到 {len(results)} 个相关文档：\n")
        
        for i, doc in enumerate(results, 1):
            print(f"--- 结果 {i} ---")
            print(f"来源：{doc.metadata.get('source', 'unknown')}")
            print(f"内容：{doc.page_content[:150]}...\n")
        
        return results
    
    def similarity_search_with_score(self, query: str, k: int = 3):
        """带相似度分数的搜索"""
        if self.vectorstore is None:
            raise ValueError("向量存储未初始化")
        
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"\n[查询] 查询：{query}")
        print(f"[结果] 找到 {len(results)} 个相关文档：\n")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"--- 结果 {i} (相似度: {score:.4f}) ---")
            print(f"来源：{doc.metadata.get('source', 'unknown')}")
            print(f"内容：{doc.page_content[:150]}...\n")
        
        return results

# 测试
if __name__ == "__main__":
    from document_loader import DocumentLoader
    from text_splitter import SmartTextSplitter
    
    # 1. 加载文档
    loader = DocumentLoader()
    documents = loader.load_directory()
    
    # 2. 切分文档
    splitter = SmartTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_documents(documents)
    
    # 3. 创建向量存储（使用免费的 HuggingFace 模型）
    vector_manager = VectorStoreManager(embedding_type="huggingface")
    vector_manager.create_vectorstore(chunks)
    
    # 4. 测试搜索
    vector_manager.similarity_search_with_score("什么是 LangChain？")