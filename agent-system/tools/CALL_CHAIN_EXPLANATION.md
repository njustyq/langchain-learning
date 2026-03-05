# RAG Tool 调用链详解

## 问题
`rag_tool.run("什么是 LangChain？")` 是如何执行到 `query_documents` 函数里的逻辑的？

## 答案：函数引用传递机制

### 核心概念

在 Python 中，函数是一等公民（first-class citizen），可以作为参数传递：

```python
# 函数定义
def my_function(x):
    return x * 2

# 函数引用（没有括号）
func_ref = my_function  # 这是函数对象本身

# 函数调用（有括号）
result = my_function(5)  # 这是执行函数，返回结果
```

### 调用链可视化

```
┌─────────────────────────────────────────────────────────────┐
│ 第 84 行：rag_tool.run("什么是 LangChain？")                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Tool.run() 方法内部（LangChain 源码）                        │
│                                                              │
│  def run(self, tool_input: str) -> str:                     │
│      # 内部执行：                                            │
│      return self.func(tool_input)  ← 调用保存的函数引用      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ self.func 是什么？                                          │
│                                                              │
│ 在创建 Tool 时（第 60 行）：                                │
│   rag_tool = Tool(                                          │
│       func=query_documents,  ← 传入函数引用（不是调用）      │
│       ...                                                    │
│   )                                                          │
│                                                              │
│ 所以：self.func = query_documents                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 第 10 行：query_documents("什么是 LangChain？")             │
│                                                              │
│ 执行函数内的所有逻辑：                                       │
│   1. 导入模块                                                │
│   2. 加载文档                                                │
│   3. 切分文档                                                │
│   4. 创建向量存储                                            │
│   5. 创建 RAGQASystem                                        │
│   6. 调用 qa_system.ask_with_sources(question)            │
│   7. 格式化返回结果                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 返回结果字符串                                               │
│                                                              │
│ "LangChain 是...\n\n[来源: documents/test.txt]"            │
└─────────────────────────────────────────────────────────────┘
```

### 代码对比

#### ❌ 错误写法（函数调用）
```python
# 这样会立即执行函数，而不是传递函数引用
rag_tool = Tool(
    func=query_documents("测试"),  # ❌ 错误：立即执行并传入返回值
    ...
)
```

#### ✅ 正确写法（函数引用）
```python
# 传入函数对象本身，不执行
rag_tool = Tool(
    func=query_documents,  # ✅ 正确：传入函数引用
    ...
)
```

### 等价写法

以下三种写法是等价的：

```python
# 方式 1：通过 Tool.run()
result = rag_tool.run("什么是 LangChain？")

# 方式 2：直接调用函数
result = query_documents("什么是 LangChain？")

# 方式 3：手动模拟 Tool.run() 的行为
result = rag_tool.func("什么是 LangChain？")
```

### Tool 类的简化实现

为了理解 Tool 的工作原理，这里是一个简化版实现：

```python
class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func  # 保存函数引用
        self.description = description
    
    def run(self, tool_input: str) -> str:
        """调用保存的函数"""
        return self.func(tool_input)  # 执行函数引用
```

### 实际执行流程

1. **第 84 行**：`rag_tool.run("什么是 LangChain？")`
2. **Tool.run() 内部**：执行 `self.func("什么是 LangChain？")`
3. **因为 self.func = query_documents**：所以实际调用 `query_documents("什么是 LangChain？")`
4. **query_documents 执行**：完成所有 RAG 查询逻辑
5. **返回结果**：字符串结果通过调用链返回

### 关键点总结

1. **函数引用 vs 函数调用**
   - `query_documents` → 函数引用（函数对象）
   - `query_documents()` → 函数调用（执行函数）

2. **Tool 类的作用**
   - 包装函数，提供统一的接口
   - 可以被 Agent 系统使用
   - 提供元数据（name, description）

3. **调用链**
   - `rag_tool.run()` → `Tool.run()` → `self.func()` → `query_documents()`

### 验证方法

运行 `tool_call_chain_demo.py` 可以看到详细的调用过程：

```bash
python tool_call_chain_demo.py
```
