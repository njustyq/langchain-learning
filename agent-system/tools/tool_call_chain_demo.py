"""
演示 Tool.run() 如何调用 func 参数的调用链
"""
from langchain_core.tools import Tool

def my_function(input_str: str) -> str:
    """自定义函数"""
    print(f"[my_function] 被调用，输入参数: {input_str}")
    return f"处理结果: {input_str.upper()}"

# 创建 Tool 对象，传入 func=my_function
my_tool = Tool(
    name="my_tool",
    func=my_function,  # 关键：这里传入函数引用，不是函数调用
    description="演示工具"
)

print("=" * 60)
print("调用链演示：")
print("=" * 60)

# 当调用 tool.run() 时，内部会调用传入的 func
print("\n1. 调用 my_tool.run('hello')")
result = my_tool.run("hello")
print(f"   返回结果: {result}")

print("\n" + "=" * 60)
print("等价写法（手动调用）：")
print("=" * 60)

# 实际上，Tool.run() 内部做的事情类似于：
print("\n2. 直接调用 my_function('hello')")
result2 = my_function("hello")
print(f"   返回结果: {result2}")

print("\n" + "=" * 60)
print("Tool 类的内部机制（简化版）：")
print("=" * 60)

class SimpleTool:
    """简化版的 Tool 类，演示内部机制"""
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func  # 保存函数引用
        self.description = description
    
    def run(self, input_str: str) -> str:
        """run 方法内部调用保存的 func"""
        print(f"[SimpleTool.run] 接收到输入: {input_str}")
        print(f"[SimpleTool.run] 准备调用 self.func (即 {self.func.__name__})")
        # 关键：这里调用保存的函数引用
        result = self.func(input_str)
        print(f"[SimpleTool.run] func 返回结果: {result}")
        return result

# 使用简化版 Tool
simple_tool = SimpleTool(
    name="simple_tool",
    func=my_function,
    description="演示"
)

print("\n3. 调用 simple_tool.run('world')")
result3 = simple_tool.run("world")
print(f"   最终返回: {result3}")

print("\n" + "=" * 60)
print("总结：")
print("=" * 60)
print("""
调用链：
  rag_tool.run("什么是 LangChain？")
    ↓
  Tool.run() 方法内部执行：
    self.func("什么是 LangChain？")  
    ↓
  因为 self.func = query_documents（在创建时传入）
    ↓
  query_documents("什么是 LangChain？")
    ↓
  执行 query_documents 函数内的所有逻辑
    ↓
  返回结果字符串
""")
