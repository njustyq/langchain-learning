from langchain_core.tools import Tool
import math
import re

def calculate(expression: str) -> str:
    """安全的数学计算器
    
    Args:
        expression: 数学表达式
    
    Returns:
        计算结果
    """
    try:
        # 移除空格
        expression = expression.strip()
        
        # 安全检查：只允许数字、运算符、括号、常见函数
        allowed_pattern = r'^[\d\+\-\*/\(\)\.\s\^sqrt,]+$'
        if not re.match(allowed_pattern, expression):
            return "错误：表达式包含不允许的字符"
        
        # 替换常见符号
        expression = expression.replace('^', '**')  # 幂运算
        expression = expression.replace('sqrt', 'math.sqrt')  # 平方根
        
        # 安全的命名空间
        safe_namespace = {
            'math': math,
            '__builtins__': {}
        }
        
        # 计算
        result = eval(expression, safe_namespace)
        
        return f"计算结果：{result}"
        
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"错误：{str(e)}"

# 创建 LangChain Tool
calculator_tool = Tool(
    name="calculator",
    func=calculate,
    description="""执行数学计算。

支持的运算：
- 基本运算：+、-、*、/
- 幂运算：^ 或 **（如：2^3 或 2**3）
- 平方根：sqrt(x)
- 括号：()

输入格式：数学表达式字符串
输出格式：计算结果

使用示例：
- "2 + 3 * 4" → 14
- "sqrt(16)" → 4.0
- "(10 + 5) * 2" → 30

注意：
1. 只输入数学表达式，不要包含其他文字
2. 使用标准数学符号
3. 确保表达式语法正确"""
)

# 测试
if __name__ == "__main__":
    print("测试计算器工具：")
    test_cases = [
        "2 + 3",
        "10 * 5 - 8",
        "sqrt(16)",
        "(100 - 50) / 2"
    ]
    
    for expr in test_cases:
        print(f"{expr} = {calculator_tool.run(expr)}")