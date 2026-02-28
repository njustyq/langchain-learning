# Embedding 模型选择指南

如果你没有 OpenAI API key，可以使用以下免费替代方案：

## 方案1：HuggingFace Embeddings（推荐⭐）

### 优点
- ✅ 完全免费
- ✅ 本地运行，不需要网络（首次下载后）
- ✅ 支持中文
- ✅ 使用简单

### 安装依赖

```bash
pip install langchain-huggingface sentence-transformers
```

### 使用方式

```python
vector_manager = VectorStoreManager(embedding_type="huggingface")
```

### 推荐模型

1. **paraphrase-multilingual-MiniLM-L12-v2**（默认）
   - 支持多语言（包括中文）
   - 模型较小（约 470MB）
   - 速度快

2. **其他中文模型**（可在代码中修改）：
   - `shibing624/text2vec-base-chinese` - 专门针对中文
   - `BAAI/bge-small-zh-v1.5` - 百度开源的中文模型

---

## 方案2：Ollama Embeddings

### 优点
- ✅ 免费
- ✅ 支持多种模型
- ✅ 性能较好

### 安装步骤

1. 安装 Ollama：https://ollama.ai/
2. 下载 embedding 模型：
   ```bash
   ollama pull nomic-embed-text
   ```
3. 安装 Python 包：
   ```bash
   pip install langchain-ollama
   ```

### 使用方式

```python
vector_manager = VectorStoreManager(embedding_type="ollama")
```

---

## 方案3：OpenAI Embeddings（需要付费）

### 使用方式

1. 安装依赖：
   ```bash
   pip install langchain-openai
   ```

2. 设置环境变量（在 `.env` 文件中）：
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. 使用：
   ```python
   vector_manager = VectorStoreManager(embedding_type="openai")
   ```

---

## 性能对比

| 方案 | 速度 | 质量 | 成本 | 中文支持 |
|------|------|------|------|----------|
| HuggingFace | 中等 | 良好 | 免费 | ✅ |
| Ollama | 快 | 很好 | 免费 | ✅ |
| OpenAI | 很快 | 优秀 | 付费 | ✅ |

## 快速开始

如果你是第一次使用，推荐使用 HuggingFace：

```python
# 1. 安装依赖
# pip install langchain-huggingface sentence-transformers

# 2. 使用默认设置（已配置为 huggingface）
from vectorizer import VectorStoreManager

vector_manager = VectorStoreManager()  # 默认使用 huggingface
```
