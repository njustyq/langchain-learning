from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

class SmartTextSplitter:
    """智能文本切分器"""
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 separators: List[str] = None):
        """
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
            separators: 分隔符优先级列表
        """
        if separators is None:
            # 默认分隔符（按优先级）
            separators = [
                "\n\n",  # 段落
                "\n",    # 行
                "。",    # 中文句号
                "！",    # 中文感叹号
                "？",    # 中文问号
                "；",    # 中文分号
                ".",     # 英文句号
                " ",     # 空格
                ""       # 字符
            ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        chunks = self.splitter.split_documents(documents)
        
        # 为每个块添加 chunk_id
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        print(f"✂️  文档已切分为 {len(chunks)} 个块")
        self._print_statistics(chunks)
        
        return chunks
    
    def _print_statistics(self, chunks: List[Document]):
        """打印切分统计信息"""
        lengths = [len(chunk.page_content) for chunk in chunks]
        
        print(f"   平均长度：{sum(lengths) / len(lengths):.0f} 字符")
        print(f"   最短：{min(lengths)} 字符")
        print(f"   最长：{max(lengths)} 字符")
        
        # 显示前3个块的预览
        print(f"\n📄 前3个块预览：")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"来源：{chunk.metadata.get('source', 'unknown')}")
            print(f"内容：{chunk.page_content[:100]}...")

# 测试
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_directory()
    
    # 切分文档
    splitter = SmartTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)