
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Literal

class State(TypedDict):
    # operator.add reducer使此仅可追加
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'向 {state["aggregate"]} 添加 "A"')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'向 {state["aggregate"]} 添加 "B"')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'向 {state["aggregate"]} 添加 "C"')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'向 {state["aggregate"]} 添加 "D"')
    return {"aggregate": ["D"]}


def conditional_edge(state: State) -> Literal["b", "c"]:
    import time
    if int(time.time()) % 2 == 1:
        return "b"
    else:
        return "c"



builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_conditional_edges(
    # 第一个参数: 当前节点名。即从哪个节点出发设置条件边，这里是 "a"。
    "a",
    # 第二个参数: 判定函数。接收 state 返回目标分支名（如 "b" 或 "c"）。
    conditional_edge,
    # 第三个参数: 分支映射字典。key 是判定函数可能返回值，value 是对应目标节点名。
    {
        "b": "b",
        "c": "c",
    },
)
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

result = graph.invoke({"aggregate": ["0"]})
print(result)

# 绘制图并显示
image = graph.get_graph().draw_mermaid_png()
with open("condition_edge_graph.png", "wb") as f:
    f.write(image)
