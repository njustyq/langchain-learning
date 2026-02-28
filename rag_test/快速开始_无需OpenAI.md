# 🚀 快速开始 - 无需 OpenAI API Key

## 步骤 1: 安装依赖

打开终端，运行以下命令：

```bash
pip install langchain-huggingface sentence-transformers langchain-chroma
```

## 步骤 2: 运行示例

```bash
cd rag_test
python example_no_openai.py
```

## 步骤 3: 使用你自己的代码

```python
from vectorizer import VectorStoreManager

# 创建向量存储管理器（使用免费的 HuggingFace）
vector_manager = VectorStoreManager(
    persist_directory="./vector_store",
    embedding_type="huggingface"  # 默认就是 huggingface，可以不写
)

# 其他操作和之前一样...
```

## ⚠️ 注意事项

1. **首次运行**：会自动下载模型（约 470MB），需要等待几分钟
2. **模型位置**：模型会缓存到 `~/.cache/huggingface/` 目录
3. **离线使用**：下载完成后，可以离线使用

## 🎯 三种 Embedding 方案对比

### 1️⃣ HuggingFace（推荐）✅

```python
vector_manager = VectorStoreManager(embedding_type="huggingface")
```

**优点**：
- ✅ 完全免费
- ✅ 无需 API key
- ✅ 支持中文
- ✅ 本地运行

**缺点**：
- ⚠️ 首次需要下载模型
- ⚠️ 速度比 OpenAI 稍慢

---

### 2️⃣ Ollama

```python
vector_manager = VectorStoreManager(embedding_type="ollama")
```

**前置步骤**：
1. 安装 Ollama: https://ollama.ai/
2. 下载模型: `ollama pull nomic-embed-text`
3. 安装包: `pip install langchain-ollama`

**优点**：
- ✅ 免费
- ✅ 性能好
- ✅ 支持多种模型

---

### 3️⃣ OpenAI

```python
vector_manager = VectorStoreManager(embedding_type="openai")
```

**前置步骤**：
1. 获取 OpenAI API key
2. 在 `.env` 文件中设置 `OPENAI_API_KEY=xxx`
3. 安装包: `pip install langchain-openai`

**优点**：
- ✅ 速度快
- ✅ 质量高

**缺点**：
- ❌ 需要付费
- ❌ 需要网络

---

## 💡 常见问题

### Q: 模型下载太慢怎么办？

可以设置国内镜像源：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者在 Python 代码中：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### Q: 可以使用其他中文模型吗？

可以！修改 `vectorizer.py` 中的 `model_name`：

```python
# 在 _create_embeddings 方法的 huggingface 部分
return HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",  # 改为中文模型
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

推荐的中文模型：
- `shibing624/text2vec-base-chinese`
- `BAAI/bge-small-zh-v1.5`
- `BAAI/bge-base-zh-v1.5`

### Q: 如何使用 GPU 加速？

修改 `device` 参数：

```python
model_kwargs={'device': 'cuda'}  # 使用 GPU
```

前提是你已经安装了支持 CUDA 的 PyTorch。

---

## 🎉 完成！

现在你可以完全免费地使用 RAG 系统，无需任何 API key！
