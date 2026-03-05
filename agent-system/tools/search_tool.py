from langchain_core.tools import Tool
import json

def web_search(query: str) -> str:
    """模拟网络搜索
    
    Args:
        query: 搜索关键词
    
    Returns:
        搜索结果摘要
    """
    # 模拟搜索结果数据库
    search_db = {
        "python": "Python 是一种高级编程语言，由 Guido van Rossum 于1991年创建。它以简洁易读的语法著称，广泛应用于 Web 开发、数据科学、人工智能等领域。",
        "langchain": "LangChain 是一个用于开发大语言模型应用的框架，由 Harrison Chase 创建。它提供了 Chains、Agents、Memory 等核心组件，简化了 LLM 应用的开发流程。",
        "openai": "OpenAI 是一家人工智能研究公司，成立于2015年。其开发的 GPT 系列模型在自然语言处理领域取得了突破性进展，包括 GPT-3、GPT-4 等。",
        "机器学习": "机器学习是人工智能的一个分支，使计算机能够从数据中学习而无需显式编程。主要包括监督学习、无监督学习和强化学习三大类。"
    }
    
    # 模糊匹配
    query_lower = query.lower()
    for key, value in search_db.items():
        if key.lower() in query_lower or query_lower in key.lower():
            return json.dumps({
                "query": query,
                "result": value,
                "source": "模拟搜索引擎"
            }, ensure_ascii=False)
    
    # 未找到结果
    return json.dumps({
        "query": query,
        "result": f"未找到关于 '{query}' 的相关信息",
        "source": "模拟搜索引擎"
    }, ensure_ascii=False)

# 创建 LangChain Tool
search_tool = Tool(
    name="web_search",
    func=web_search,
    description="""在互联网上搜索信息。

输入格式：搜索关键词或问题
输出格式：JSON 格式的搜索结果

使用场景：
- 查询最新信息
- 查找知识库中没有的内容
- 获取背景资料

使用示例：
- 输入："Python 是什么"
- 输出：{"query": "Python 是什么", "result": "Python 是一种...", "source": "..."}

注意：输入简洁的关键词或问题，避免冗长的描述。"""
)

# 测试
if __name__ == "__main__":
    print("测试搜索工具：")
    print(search_tool.run("Python"))
    print(search_tool.run("LangChain"))