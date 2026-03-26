import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

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

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()

result = graph.invoke({"aggregate": ["0"]})
print(result)