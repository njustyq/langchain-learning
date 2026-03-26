from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# 定义状态
class State(TypedDict):
    count: int

# 定义节点函数
# 节点函数接收当前状态并返回更新后的状态

def my_node(state: State):
    # 读取当前状态
    current_count = state["count"]
    # 返回更新后的状态
    return {"count": current_count + 1}
def my_node_2(state: State):
    # 读取当前状态
    current_count = state["count"]
    # 返回更新后的状态
    return {"count": current_count + 10}

# 创建图
builder = StateGraph(State)
# 添加节点
builder.add_node("my_node", my_node)
builder.add_node("my_node_2", my_node_2)
# 添加边（定义控制流）
builder.add_edge(START, "my_node")
builder.add_edge("my_node", "my_node_2")
builder.add_edge("my_node_2", END)
# 编译图
graph = builder.compile()

# 执行图
result = graph.invoke({"count": 0})
print(result)
# {'count': 1}

# 绘制图并显示
image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(image)

display(Image(image))  # Notebook里继续显示
print("saved to graph.png")