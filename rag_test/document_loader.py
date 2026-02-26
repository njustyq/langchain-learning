from langchain_community.document_loaders import (  # pyright: ignore[reportMissingImports]
    TextLoader,
    PyPDFLoader,
    DirectoryLoader
)
from langchain_core.documents import Document  # pyright: ignore[reportMissingImports]
from pathlib import Path
from typing import List
import os

class DocumentLoader:
    """统一的文档加载器"""
    
    def __init__(self, documents_dir: str = "./documents"):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(exist_ok=True)
    
    def load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        file_path = Path(file_path) 
        
        if not file_path.exists():  
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件类型选择加载器
        if file_path.suffix == ".pdf":  
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix in [".txt", ".md"]:  
            loader = TextLoader(str(file_path), encoding='utf-8')
        else:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
        
        documents = loader.load()
        
        # 添加元数据
        for doc in documents:
            doc.metadata["source"] = file_path.name
            doc.metadata["file_type"] = file_path.suffix
        
        print(f"[OK] 已加载：{file_path.name}")
        print(f"   页数/段落数：{len(documents)}")
        
        return documents
    
    def load_directory(self, file_types: List[str] = [".txt", ".pdf", ".md"]) -> List[Document]:
        """加载目录下的所有文档"""
        all_documents = []
        
        for file_type in file_types:
            pattern = f"**/*{file_type}"
            
            if file_type == ".pdf":
                loader = DirectoryLoader(
                    str(self.documents_dir),
                    glob=pattern,
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
            else:
                loader = DirectoryLoader(
                    str(self.documents_dir),
                    glob=pattern,
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                    show_progress=True
                )
            
            documents = loader.load()
            all_documents.extend(documents)
        
        print(f"\n[INFO] 总共加载了 {len(all_documents)} 个文档片段")
        return all_documents

# 测试
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # 创建测试文档
    test_file = loader.documents_dir / "test.txt"
    test_file.write_text("""
LangChain 是一个用于开发由语言模型驱动的应用程序的框架。

核心组件：
1. Models - 语言模型接口
2. Prompts - 提示词管理
3. Chains - 组件组合
4. Agents - 智能体系统

LangChain 使应用程序能够连接到上下文，并依靠语言模型进行推理。
    """, encoding='utf-8')
    
    # 加载单个文件
    docs = loader.load_single_file(test_file)
    print(f"\n文档内容预览：")
    print(docs[0].page_content[:200])