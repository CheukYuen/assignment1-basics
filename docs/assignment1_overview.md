# CS336 作业1（基础篇）：从零构建 Transformer 语言模型

版本 1.0.6 | CS336 课程团队 | Spring 2025

---

## 1 作业概述

本作业要求你从零开始构建训练标准 Transformer 语言模型（LM）所需的全部组件，并实际训练一些模型。

---

### 你需要实现的内容

1. **字节对编码（BPE）分词器**（§2）
2. **Transformer 语言模型**（§3）
3. **交叉熵损失函数与 AdamW 优化器**（§4）
4. **训练循环**，支持模型与优化器状态的序列化与加载（§5）

---

### 你需要运行的内容

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 用训练好的分词器对数据集进行处理，转换为整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer LM。
4. 用训练好的 Transformer LM 生成样本并评估困惑度（perplexity）。
5. 在 OpenWebText 上训练模型，将所得困惑度提交到排行榜。

---

### 允许使用的工具

**禁止**使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的定义，以下例外除外：
- `torch.nn.Parameter`
- `torch.nn` 中的容器类（如 `Module`、`ModuleList`、`Sequential` 等）
- `torch.optim.Optimizer` 基类

---

### 关于 AI 工具的声明

允许使用 ChatGPT 等 LLM 解答底层编程问题或高层概念问题，但**禁止**直接用其解题。
**强烈不建议**使用 AI 自动补全（如 Cursor Tab、GitHub Copilot）。

---

### 代码结构

| 路径 | 说明 |
|------|------|
| `cs336_basics/` | **你写代码的地方** |
| `tests/adapters.py` | 胶水代码；通过调用你的实现来填充 |
| `tests/test_*.py` | 必须通过的测试；**不要修改** |

---

### 提交方式

- **Gradescope**：提交 `writeup.pdf`（书面回答）和 `code.zip`（代码压缩包）
- **排行榜**：向 `github.com/stanford-cs336/assignment1-basics-leaderboard` 提交 PR

---

### 数据集

- **TinyStories** 和 **OpenWebText**
- 课程机器上位于 `/data` 目录
- 也可通过 `README.md` 中的说明下载
