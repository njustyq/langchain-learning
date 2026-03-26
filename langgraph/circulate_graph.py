import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # operator.add reducer使此仅可追加
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'节点A看到 {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'节点B看到 {state["aggregate"]}')
    return {"aggregate": ["B"]}

# 定义节点
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# 定义边
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
reslut = graph.invoke({"aggregate":["0"]})
image = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(image)
