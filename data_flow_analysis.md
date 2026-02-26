# LangChain 数据流分析：prompt 和 llm 之间的数据流动

## 代码结构

```python
chain = prompt | llm | parser
result = chain.invoke({
    "text": "人工智能正在改变世界",
    "format_instructions": parser.get_format_instructions()
})
```

## 数据流动过程详解

### 1. 输入阶段（chain.invoke()）

**输入数据：**
```python
{
    "text": "人工智能正在改变世界",
    "format_instructions": parser.get_format_instructions()
}
```

### 2. Prompt 阶段（prompt | ...）

**数据转换过程：**

#### 2.1 Prompt 模板定义
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业的翻译专家，擅长将中文翻译成地道的英文。"),
    ("human", """请将以下中文翻译成英文，并评估翻译质量。

{format_instructions}

中文文本：
{text}
""")
])
```

#### 2.2 Prompt.format() 执行
当 `chain.invoke()` 被调用时，LangChain 内部会执行：

```python
# 伪代码展示内部流程
formatted_messages = prompt.format(
    text="人工智能正在改变世界",
    format_instructions=parser.get_format_instructions()
)
```

#### 2.3 格式化后的消息结构
`prompt.format()` 返回的是一个 `ChatPromptValue` 对象，包含格式化后的消息列表：

```python
# 内部数据结构（简化表示）
[
    SystemMessage(content="你是一位专业的翻译专家，擅长将中文翻译成地道的英文。"),
    HumanMessage(content="""请将以下中文翻译成英文，并评估翻译质量。

{format_instructions 被替换后的内容}

中文文本：
人工智能正在改变世界
""")
]
```

**关键点：**
- `{text}` 被替换为 "人工智能正在改变世界"
- `{format_instructions}` 被替换为 Pydantic 格式说明（JSON Schema）
- 结果是一个包含 SystemMessage 和 HumanMessage 的消息列表

### 3. LLM 阶段（... | llm | ...）

#### 3.1 数据传递
格式化后的消息列表（`ChatPromptValue`）被传递给 `llm`：

```python
# 伪代码
llm_input = formatted_messages.to_messages()  # 转换为消息对象列表
llm_response = llm.invoke(llm_input)
```

#### 3.2 LLM 处理
`ChatOpenAI` 接收消息列表后：

1. **序列化为 API 格式**：将消息列表转换为 OpenAI API 所需的格式
   ```json
   {
     "model": "MaaS 3.7 Sonnet",
     "messages": [
       {"role": "system", "content": "你是一位专业的翻译专家..."},
       {"role": "user", "content": "请将以下中文翻译成英文..."}
     ],
     "temperature": 0
   }
   ```

2. **发送 HTTP 请求**：调用 OpenAI 兼容的 API
   ```python
   # 内部执行
   response = openai_client.chat.completions.create(
       model="MaaS 3.7 Sonnet",
       messages=[...],
       temperature=0
   )
   ```

3. **接收响应**：返回 `AIMessage` 对象
   ```python
   # 返回类型：AIMessage
   AIMessage(
       content='{"translation": "Artificial intelligence is changing the world", "quality_score": 9, "notes": "..."}'
   )
   ```

### 4. Parser 阶段（... | parser）

#### 4.1 解析输入
`AIMessage.content`（字符串）被传递给 parser：

```python
# 伪代码
raw_text = llm_response.content  # 获取 AI 返回的文本
parsed_result = parser.parse(raw_text)  # 解析为 Pydantic 对象
```

#### 4.2 输出解析
`PydanticOutputParser` 执行：

1. **提取 JSON**：从文本中提取 JSON 部分
2. **验证结构**：根据 `TranslationResult` 模型验证
3. **创建对象**：返回 `TranslationResult` 实例

```python
# 最终返回
TranslationResult(
    translation="Artificial intelligence is changing the world",
    quality_score=9,
    notes="这是一个简洁明了的翻译..."
)
```

## 完整数据流图

```
输入字典
  ↓
{
  "text": "人工智能正在改变世界",
  "format_instructions": "..."
}
  ↓
[Prompt.format()]  ← 模板变量替换
  ↓
ChatPromptValue (消息列表)
  ↓
[
  SystemMessage("你是一位专业的翻译专家..."),
  HumanMessage("请将以下中文翻译成英文...\n{format_instructions}\n中文文本：人工智能正在改变世界")
]
  ↓
[LLM.invoke()]  ← HTTP API 调用
  ↓
AIMessage(content="{\"translation\": \"...\", \"quality_score\": 9, \"notes\": \"...\"}")
  ↓
[Parser.parse()]  ← JSON 解析和验证
  ↓
TranslationResult 对象
  ↓
{
  translation: "Artificial intelligence is changing the world",
  quality_score: 9,
  notes: "..."
}
```

## 关键技术点

### 1. 管道操作符（|）
```python
chain = prompt | llm | parser
```
这是 Python 3.9+ 的管道操作符，等价于：
```python
chain = RunnableSequence(prompt, llm, parser)
```

### 2. Runnable 接口
LangChain 中的所有组件（prompt、llm、parser）都实现了 `Runnable` 接口，支持：
- `invoke()`: 同步调用
- `ainvoke()`: 异步调用
- `stream()`: 流式处理
- `|` 操作符: 链式组合

### 3. 类型转换
- **输入**：`Dict[str, Any]` → **Prompt** → `ChatPromptValue` → **LLM** → `AIMessage` → **Parser** → `TranslationResult`

### 4. 异步支持
每个阶段都可以异步执行：
```python
result = await chain.ainvoke({...})
```

## 调试技巧

如果想查看中间数据，可以分步执行：

```python
# 1. 查看格式化后的 prompt
formatted = prompt.invoke({
    "text": "人工智能正在改变世界",
    "format_instructions": parser.get_format_instructions()
})
print("Formatted messages:", formatted.to_messages())

# 2. 查看 LLM 的原始响应
llm_response = llm.invoke(formatted)
print("LLM response:", llm_response.content)

# 3. 查看解析后的结果
parsed = parser.invoke(llm_response)
print("Parsed result:", parsed)
```
