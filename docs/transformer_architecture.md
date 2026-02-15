# 第3章：Transformer 语言模型架构（教学翻译）

---

## 3.1 Transformer LM 总览

```
┌──────────────────────┐
│   输出概率 (Output     │
│   Probabilities)      │  ← 前向传播的终点
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│     Softmax           │
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│   线性层 (输出嵌入)     │  ← 将 d_model 映射到 vocab_size
│   Linear / LM Head    │
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│      RMSNorm          │  ← 最终归一化
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│  Transformer Block    │
│       ...             │  ← 共 num_layers 层
│  Transformer Block    │
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│   Token Embedding     │  ← 将整数 token ID 映射为向量
└──────────┬───────────┘
           ▲
┌──────────┴───────────┐
│    输入 (Inputs)       │  ← (batch_size, seq_len) 的整数张量，前向传播的起点
└──────────────────────┘
```

语言模型接收一个**批量的整数 token ID 序列**（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），输出一个**归一化的概率分布**（形状为 `(batch_size, sequence_length, vocab_size)` 的张量），其中预测的分布是**对每个输入 token 的下一个词的预测**。

**训练时**，我们用这些下一词预测来计算**交叉熵损失**（实际下一个词 vs 预测的下一个词）。

**推理/生成时**，我们取最后一个时间步的预测分布，从中采样下一个 token（比如取概率最高的 token、从分布中随机采样等），将生成的 token 追加到输入序列，然后重复。

> **目的与价值**：这一段建立了整体心智模型——Transformer LM 本质上就是一个函数：`整数序列 → 下一词概率分布`。理解这一点是理解后续所有组件的前提。

---

在本部分中，你将从零构建这个 Transformer 语言模型。我们先从高层描述开始，然后逐步细化各个组件。

### 3.1.1 Token 嵌入（Token Embeddings）

在最开始的一步，Transformer 将（批量的）token ID 序列**嵌入**为包含 token 身份信息的向量序列（即图中的红色方块）。

具体来说，给定一组 token ID，Transformer 语言模型使用一个 **token 嵌入层**来生成向量序列。每个嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量，输出形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

> **目的与价值**：神经网络无法直接处理离散整数。嵌入层将每个 token ID 映射到一个连续的、可学习的高维向量。这是模型的"入口"——把离散的语言符号转换为连续的数学对象，才能进行梯度下降训练。`d_model` 是模型的核心维度，贯穿整个 Transformer。

### 3.1.2 Pre-norm Transformer Block（预归一化 Transformer 块）

嵌入之后，激活值通过若干结构相同的神经网络层进行处理。标准的仅解码器（decoder-only）Transformer 语言模型由 `num_layers` 个相同的层（通常称为 Transformer "块"）组成。每个 Transformer 块接收形状为 `(batch_size, sequence_length, d_model)` 的输入，返回相同形状的输出。每个块通过**自注意力**（self-attention）在序列维度上聚合信息，并通过**前馈层**（feed-forward layers）进行非线性变换。

```
┌─────────────────────────────────────────┐
│ 输出张量 (batch_size, seq_len, d_model)   │
└───────────────────┬─────────────────────┘
                    │
              ┌─────┴─────┐
              │    Add     │◄──────────────────┐
              └─────┬─────┘                    │
                    │                          │
         ┌──────────▼──────────┐               │
         │   Position-Wise     │               │
         │   Feed-Forward      │               │
         └──────────┬──────────┘               │
                    │                          │
         ┌──────────▼──────────┐               │
         │      RMSNorm        │               │
         └──────────┬──────────┘               │
                    │                          │
              ┌─────┴─────┐                    │
              │    Add     │◄─────────┐        │
              └─────┬─────┘          │        │
                    │                │        │
         ┌──────────▼──────────┐     │        │
         │  Causal Multi-Head  │     │        │
         │  Self-Attention     │     │        │
         │     w/ RoPE         │     │        │
         └──────────┬──────────┘     │        │
                    │                │        │
         ┌──────────▼──────────┐     │        │
         │      RMSNorm        │     │        │
         └──────────┬──────────┘     │        │
                    │                │        │
                    ├────────────────┘        │
                    ├─────────────────────────┘
                    │
┌───────────────────▼─────────────────────┐
│ 输入张量 (batch_size, seq_len, d_model)   │
└─────────────────────────────────────────┘
```

> **目的与价值**：Transformer 块是整个模型的核心重复单元。每一层做两件事：(1) 注意力——让每个位置"看到"其他位置的信息；(2) 前馈网络——对每个位置独立做非线性变换。通过堆叠多层，模型逐步构建出越来越抽象的表示。

---

## 3.2 输出归一化与嵌入（Output Normalization and Embedding）

经过 `num_layers` 个 Transformer 块之后，我们将最终的激活值转化为词汇表上的分布。

我们将实现 "pre-norm"（预归一化）Transformer 块（详见 §3.5），它额外要求在最后一个 Transformer 块之后使用**层归一化**，以确保输出被正确缩放。

在这个归一化之后，我们使用一个标准的**可学习线性变换**将 Transformer 块的输出转换为预测的下一词 logits（参见 Radford et al. [2018] 公式 2）。

> **目的与价值**：pre-norm 架构有一个干净的"残差流"（residual stream）——从输入嵌入到最终输出之间没有归一化打断。但这意味着最后一层的输出可能尺度不对，所以需要额外的最终归一化。线性投影（LM head）将 `d_model` 维的表示映射到 `vocab_size` 维的 logits，这就是模型的"出口"。

---

## 3.3 备注：批处理、Einsum 与高效计算

在整个 Transformer 中，我们将对许多"类批处理"（batch-like）的输入执行相同的计算。以下是一些例子：

- **批次中的元素**：我们对每个批次元素应用相同的 Transformer 前向操作。
- **序列长度**：RMSNorm 和前馈网络等"逐位置"（position-wise）的操作在序列的每个位置上独立运行。
- **注意力头**：注意力操作在"多头"注意力操作中跨注意力头进行批处理。

拥有一种符合人体工学的方式来执行这些操作是很有用的——既能充分利用 GPU，又易于阅读和理解。许多 PyTorch 操作可以在张量开头接受额外的"类批处理"维度，并在这些维度上高效地重复/广播操作。

例如，假设我们在做逐位置的批量操作。我们有一个"数据张量" D，形状为 `(batch_size, sequence_length, d_model)`，我们想对矩阵 A（形状为 `(d_model, d_model)`）做批量向量-矩阵乘法。在这种情况下，`D @ A` 会做批量矩阵乘法，这是 PyTorch 中的高效原语，其中 `(batch_size, sequence_length)` 维度被批处理。

因此，假设你的函数可能会接收额外的类批处理维度，并将这些维度保持在 PyTorch shape 的开头，这是很有帮助的。为了将张量组织成这种可批处理的形式，可能需要多步 `view`、`reshape` 和 `transpose`。这可能有些麻烦，而且代码往往变得难以阅读。

一个更符合人体工学的选择是使用 **einsum 表示法**（通过 `torch.einsum`），或者使用框架无关的库如 **einops** 或 **einx**。两个关键操作是 `einsum`（可以对任意维度的输入张量进行张量收缩）和 `rearrange`（可以重排、拼接和拆分任意维度）。事实证明，机器学习中的几乎所有操作都是维度调整和张量收缩的某种组合，再加上偶尔的（通常是逐点的）非线性函数。这意味着使用 einsum 表示法可以让大量代码变得更可读、更灵活。

**我们强烈建议在课程中学习和使用 einsum 表示法。** 之前没有接触过 einsum 表示法的同学应该使用 **einops**，已经熟悉 einops 的同学应该学习更通用的 **einx**。两个包都已经安装在我们提供的环境中。

> **目的与价值**：这一整节解决的是"工程实践"问题。Transformer 的代码充满了维度操作（reshape、transpose），如果用原始 PyTorch 写，代码既难读又容易出 bug。einsum 表示法让"文档就是实现"——你在代码中直接写清楚每个维度的名字和含义。这是一个改变工作效率的工具建议。

以下是一些 einsum 表示法的使用示例。这些是对 einops 文档的补充，你应该先阅读 einops 的文档。

### 示例 1：使用 einops.einsum 的批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

## 基本实现
Y = D @ A.T
# 难以看出输入输出的形状和含义。
# D 和 A 可以有什么形状，会不会有意外行为？

## Einsum 是自文档化的、健壮的
#             D                     A             ->    Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## 或者，一个批量版本，D 可以有任意前导维度但 A 受限。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

### 示例 2：使用 einops.rearrange 的广播操作

我们有一批图像，对每张图像我们想根据某个缩放因子生成 10 个暗化版本：

```python
images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape 并乘法
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## 或者一步到位：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### 示例 3：使用 einops.rearrange 的像素混合

假设我们有一批图像，表示为形状 `(batch, height, width, channel)` 的张量，我们想对图像的所有像素执行线性变换，但这个变换应该对每个通道独立进行。线性变换表示为矩阵 B，形状为 `(height * width, height * width)`。

```python
channels_last = torch.randn(64, 32, 32, 3)  # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## 原始方式：用 view + transpose 重排图像张量进行跨像素混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

改用 einops：

```python
height = width = 32
## rearrange 替代笨拙的 torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

或者，如果你胆子大：用 einx.dot 一步完成（einx 等价于 einops.einsum）

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

第一种实现可以通过在前后添加注释来说明输入输出形状来改进，但这很笨拙且容易出 bug。**使用 einsum 表示法，文档就是实现！**

Einsum 表示法可以处理任意输入批处理维度，但也有一个关键好处：**自文档化**。在使用 einsum 表示法的代码中，输入和输出张量的相关形状更加清晰。对于其余张量，你可以考虑使用 Tensor 类型提示，例如使用 `jaxtyping` 库（不特定于 Jax）。

我们将在作业 2 中更多地讨论使用 einsum 表示法的性能影响，但现在你只需要知道它们几乎总是比替代方案更好！

### 3.3.1 数学表示与内存排列

许多机器学习论文在其表示中使用**行向量**，这与 NumPy 和 PyTorch 默认使用的**行优先内存排列**（row-major）一致。使用行向量时，线性变换看起来像：

$$y = x W^{\top}$$

其中行优先的 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，行向量 $x \in \mathbb{R}^{1 \times d_{\text{in}}}$。

在线性代数中，更常见的是使用**列向量**，线性变换看起来像：

$$y = W x$$

其中行优先的 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，列向量 $x \in \mathbb{R}^{d_{\text{in}}}$。

**我们将在本作业中使用列向量进行数学表示**，因为这样通常更容易理解数学。你应该记住，如果你想使用普通的矩阵乘法表示法，由于 PyTorch 使用行优先内存排列，你需要使用行向量惯例来应用矩阵。如果你使用 einsum 进行矩阵运算，这就不是问题了。

> **目的与价值**：这解决了一个常见的困惑源——论文中的数学公式用列向量写 `y = Wx`，但代码中 PyTorch 用行向量做 `y = x @ W.T`。理解这个转换关系是正确实现的前提。用 einsum 可以完全回避这个问题。

---

## 3.4 基本构件：Linear 和 Embedding 模块

### 3.4.1 参数初始化

有效训练神经网络往往需要仔细初始化模型参数——糟糕的初始化会导致不良行为，如**梯度消失或梯度爆炸**。Pre-norm Transformer 对初始化异常鲁棒，但初始化仍然会对训练速度和收敛产生显著影响。由于本作业已经很长，我们将把细节留到作业 3，这里只给出一些近似的初始化设置，应该在大多数情况下都能工作。目前使用：

- **Linear 权重**：$\mathcal{N}\left(\mu=0,\ \sigma^2 = \frac{2}{d_{\text{in}}+d_{\text{out}}}\right)$，在 $[-3\sigma, 3\sigma]$ 处截断。
- **Embedding**：$\mathcal{N}(\mu=0,\ \sigma^2=1)$，在 $[-3, 3]$ 处截断。
- **RMSNorm**：增益初始化为 1。

你应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

> **目的与价值**：初始化看似琐碎，实际上极其重要。如果权重太大，前向传播中激活值会爆炸；如果太小，梯度会消失。截断正态分布避免了极端值。这里给出的公式是 "Xavier/Glorot 风格"的变体，使用 $\frac{2}{d_{\text{in}}+d_{\text{out}}}$ 的方差来平衡输入输出维度。

### 3.4.2 Linear 模块

线性层是 Transformer 和一般神经网络的基本构件。首先，你将实现自己的 `Linear` 类，继承自 `torch.nn.Module`，执行线性变换：

$$y = W x$$

注意我们**不包含偏置项**，遵循大多数现代 LLM 的做法。

> **目的与价值**：从零实现 Linear 层迫使你理解最基本的操作。存储 $W$（而非 $W^{\top}$）是出于内存排列的考虑——参数的物理布局会影响 GPU 计算效率。不加偏置是现代 LLM 的标准做法，因为偏置对模型性能贡献极小但增加了参数量。

#### 问题 (linear)：实现线性模块（1 分）

**可交付物**：实现一个继承自 `torch.nn.Module` 的 `Linear` 类，执行线性变换。你的实现应遵循 PyTorch 内置 `nn.Linear` 模块的接口，但没有 bias 参数。推荐接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

构造线性变换模块。参数：

- `in_features: int` — 输入的最后一维
- `out_features: int` — 输出的最后一维
- `device: torch.device | None = None` — 参数存储的设备
- `dtype: torch.dtype | None = None` — 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

对输入应用线性变换。

确保：

- 继承 `nn.Module`
- 调用父类构造函数
- 将参数存储为 $W$（不是 $W^{\top}$），出于内存排列原因，放在 `nn.Parameter` 中
- 当然，不要使用 `nn.Linear` 或 `nn.functional.linear`

使用上述初始化设置和 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试你的 Linear 模块，在 `adapters.run_linear` 实现测试适配器。适配器应该将给定的权重加载到你的 Linear 模块中。你可以使用 `Module.load_state_dict` 来实现。然后运行 `uv run pytest -k test_linear`。

### 3.4.3 Embedding 模块

如上所述，Transformer 的第一层是一个嵌入层，将整数 token ID 映射到 `d_model` 维的向量空间。我们将实现一个自定义的 `Embedding` 类，继承自 `torch.nn.Module`（因此你不应使用 `nn.Embedding`）。`forward` 方法应通过使用形状为 `(batch_size, sequence_length)` 的 `torch.LongTensor` token ID，**索引**到形状为 `(vocab_size, d_model)` 的嵌入矩阵中，来选择每个 token ID 的嵌入向量。

> **目的与价值**：Embedding 本质上就是一个查找表——给一个整数 ID，返回对应的行向量。实现起来极其简单（就是数组索引），但理解它是可学习参数、参与梯度更新，这一点很重要。

#### 问题 (embedding)：实现嵌入模块（1 分）

**可交付物**：实现继承自 `torch.nn.Module` 的 `Embedding` 类，执行嵌入查找。你的实现应遵循 PyTorch 内置 `nn.Embedding` 模块的接口。推荐接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
```

参数：

- `num_embeddings: int` — 词汇表大小
- `embedding_dim: int` — 嵌入向量的维度，即 $d_{\text{model}}$
- `device: torch.device | None = None`
- `dtype: torch.dtype | None = None`

```python
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
```

查找给定 token ID 的嵌入向量。

确保：

- 继承 `nn.Module`
- 调用父类构造函数
- 将嵌入矩阵初始化为 `nn.Parameter`
- 存储嵌入矩阵时 `d_model` 作为最后一维
- 不要使用 `nn.Embedding` 或 `nn.functional.embedding`

同样使用上述初始化设置，使用 `torch.nn.init.trunc_normal_` 初始化权重。

在 `adapters.run_embedding` 实现测试适配器，然后运行 `uv run pytest -k test_embedding`。

---

## 3.5 Pre-Norm Transformer Block

每个 Transformer 块有两个子层：**多头自注意力机制**和**逐位置前馈网络**（Vaswani et al., 2017, 第 3.1 节）。

在原始 Transformer 论文中，模型在每个子层周围使用残差连接，然后接层归一化。这种架构通常被称为 "**post-norm**"（后归一化）Transformer，因为层归一化应用在子层输出之后。然而，多项研究发现将层归一化从每个子层的**输出**移到**输入**（在最后一个 Transformer 块之后额外加一层归一化）可以改善 Transformer 的训练稳定性 [Nguyen and Salazar, 2019, Xiong et al., 2020]——参见图 2 中 "pre-norm" Transformer 块的可视化表示。每个 Transformer 块子层的输出随后通过残差连接加到子层输入上（Vaswani et al., 2017, 第 5.4 节）。

**Pre-norm 的直觉**是：存在一条干净的"残差流"，没有任何归一化，从输入嵌入直接到 Transformer 的最终输出，据说这改善了梯度流。这种 pre-norm Transformer 现在是语言模型的标准用法（如 GPT-3、LLaMA、PaLM 等），所以我们将实现这个变体。

我们将依次走过 pre-norm Transformer 块的每个组件并实现它们。

> **目的与价值**：Pre-norm vs Post-norm 是 Transformer 架构最重要的设计选择之一。Post-norm 更难训练（需要更仔细的学习率调整和 warmup），Pre-norm 更稳定。后面的消融实验会让你亲身体会这个差异。残差连接保证信息可以"跳过"子层直接传递，这是深度网络可训练的关键。

---

### 3.5.1 根均方层归一化（RMSNorm）

原始 Transformer 实现 [Vaswani et al., 2017] 使用层归一化 [Ba et al., 2016] 来归一化激活值。遵循 [Touvron et al., 2023]，我们将使用**根均方层归一化**（RMSNorm; Zhang and Sennrich, 2019, 公式 4）。给定一个激活向量 $a \in \mathbb{R}^{d_{\text{model}}}$，RMSNorm 将每个激活 $a_i$ 重新缩放如下：

$$\text{RMSNorm}(a_i) = \frac{a_i}{\text{RMS}(a)} \cdot g_i$$

其中：

$$\text{RMS}(a) = \sqrt{\frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} a_i^2 + \varepsilon}$$

这里 $g_i$ 是一个**可学习的"增益"参数**（总共有 `d_model` 个这样的参数），$\varepsilon$ 是一个超参数，通常固定为 `1e-5`。

> **目的与价值**：RMSNorm 相比完整的 LayerNorm 更简单、更快——它省略了均值中心化步骤，只做方差归一化。直觉上，它确保每一层的输入都有一致的"尺度"，防止激活值在层间累积增大或缩小。可学习的增益参数 $g_i$ 让模型在需要时可以恢复原始尺度。

你应该将输入**向上转型**到 `torch.float32` 以防止在对输入求平方时溢出。总体上，你的 forward 方法应该像这样：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# 你的 RMSNorm 代码
...
result = ...

# 以原始 dtype 返回结果
return result.to(in_dtype)
```

#### 问题 (rmsnorm)：根均方层归一化（1 分）

**可交付物**：将 RMSNorm 实现为 `torch.nn.Module`。推荐接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

参数：

- `d_model: int` — 模型的隐藏维度
- `eps: float = 1e-5` — 数值稳定性的 epsilon 值
- `device, dtype` — 同上

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

处理形状为 `(batch_size, sequence_length, d_model)` 的输入张量，返回相同形状的张量。

**注意**：记得在执行归一化之前将输入向上转型为 `torch.float32`（之后再向下转型回原始 dtype）。

实现 `adapters.run_rmsnorm`，然后运行 `uv run pytest -k test_rmsnorm`。

---

### 3.5.2 逐位置前馈网络（Position-Wise Feed-Forward Network）

```
SiLU 激活函数图示：

SiLU(x) = x * sigma(x)

     4 |                          ___/
     3 |                       __/
     2 |                    __/
     1 |                 __/
─────0─┼────────────────/──────────── x
    -1 |            __/
    -2 |        __/
    -3 |    ___/
    -4 |___/
       -4  -3  -2  -1   0   1   2   3   4

对比：
- SiLU:     f(x) = x * sigma(x)  ← 平滑的，在0附近非零
- ReLU:     f(x) = max(0, x)     ← 在0处有拐点
- Identity: f(x) = x             ← 直线
```

在原始 Transformer 论文（Vaswani et al. [2017] 第 3.3 节）中，Transformer 前馈网络由两个线性变换和中间的 ReLU 激活（$\text{ReLU}(x) = \max(0, x)$）组成。内部前馈层的维度通常是输入维度的 4 倍。

然而，现代语言模型相比原始设计通常有两个主要变化：使用**另一种激活函数**并采用**门控机制**。具体来说，我们将实现在 Llama 3 [Grattafiori et al., 2024] 和 Qwen 2.5 [Yang et al., 2024] 等 LLM 中采用的 "**SwiGLU**" 激活函数，它将 SiLU（常称为 Swish）激活与称为**门控线性单元**（GLU）的门控机制结合。我们也将省略线性层中有时使用的偏置项，遵循自 PaLM [Chowdhery et al., 2022] 和 LLaMA [Touvron et al., 2023] 以来的大多数现代 LLM。

**SiLU 或 Swish 激活函数** [Hendrycks and Gimpel, 2016, Elfwing et al., 2017] 定义如下：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

如图所示，SiLU 激活函数类似于 ReLU 激活函数，但**在零处是平滑的**。

**门控线性单元（GLU）** 最初由 Dauphin et al. [2017] 定义为一个线性变换通过 sigmoid 函数后与另一个线性变换的逐元素乘积：

$$\text{GLU}(x, W_1, W_2) = \sigma(W_1 x) \odot W_2 x$$

其中 $\odot$ 表示逐元素乘法。门控线性单元被认为"通过为梯度提供线性路径同时保留非线性能力，减少了深度架构的梯度消失问题。"

将 SiLU/Swish 和 GLU 放在一起，我们得到 **SwiGLU**，这将用于我们的前馈网络：

$$\text{FFN}(x) = \text{SwiGLU}(x, W_1, W_2, W_3) = W_2 \left( \text{SiLU}(W_1 x) \odot W_3 x \right)$$

其中 $x \in \mathbb{R}^{d_{\text{model}}}$，$W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$，标准设置 $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$。

> **目的与价值**：SwiGLU 是现代 LLM 中最广泛使用的前馈网络变体。它的关键设计：
> 1. **SiLU 替代 ReLU**：ReLU 在 0 处不可导且"死神经元"问题，SiLU 平滑且允许小的负值通过
> 2. **门控机制**：$W_3 x$ 作为"门"控制哪些特征通过，增加了模型的表达能力
> 3. **$d_{\text{ff}} = 8/3 \cdot d_{\text{model}}$** 而非 $4 \cdot d_{\text{model}}$：因为 SwiGLU 有 3 个权重矩阵而非 2 个，$8/3$ 让总参数量与传统 FFN 大致匹配
>
> Shazeer [2020] 的名言很有趣："我们无法解释为什么这些架构似乎有效；我们将它们的成功归因于一切之源——神圣的恩典。"

#### 问题 (positionwise_feedforward)：实现逐位置前馈网络（2 分）

**可交付物**：实现 SwiGLU 前馈网络，由 SiLU 激活函数和 GLU 组成。

**注意**：在这种特殊情况下，你可以在实现中自由使用 `torch.sigmoid` 以确保数值稳定性。

你应该将 $d_{\text{ff}}$ 设置为大约 $\frac{8}{3} \times d_{\text{model}}$，同时确保内部前馈层的维度是 64 的倍数，以充分利用硬件。

实现 `adapters.run_swiglu`，然后运行 `uv run pytest -k test_swiglu`。

---

### 3.5.3 相对位置嵌入（RoPE）

为了向模型注入**位置信息**，我们将实现**旋转位置嵌入**（Rotary Position Embeddings）[Su et al., 2021]，通常称为 RoPE。

对于在 token 位置 $i$ 的给定查询 token $q^{(i)} = W_q x^{(i)} \in \mathbb{R}^d$，我们将应用一个成对旋转矩阵 $R^i$，得到 $q'^{(i)} = R^i q^{(i)} = R^i W_q x^{(i)}$。

这里，$R^i$ 将嵌入元素对 $q^{(i)}_{2k-1:2k}$ 作为 2D 向量旋转角度：

$$\theta_{i,k} = \frac{i}{\Theta^{(2k-2)/d}}$$

其中 $k \in \{1, \ldots, d/2\}$，$\Theta$ 是某个常数。

因此，我们可以将 $R^i$ 视为大小为 $d \times d$ 的**分块对角矩阵**，对于 $k \in \{1, \ldots, d/2\}$，块 $R^i_k$ 为：

$$R^i_k = \begin{bmatrix} \cos(\theta_{i,k}) & -\sin(\theta_{i,k}) \\ \sin(\theta_{i,k}) & \cos(\theta_{i,k}) \end{bmatrix}$$

完整的旋转矩阵为：

```
     ┌                                           ┐
     │  R¹ᵢ    0      0     ...     0            │
     │   0    R²ᵢ     0     ...     0            │
Rⁱ = │   0     0     R³ᵢ    ...     0            │
     │   .     .      .      .      .            │
     │   0     0      0     ...   R^(d/2)ᵢ       │
     └                                           ┘

其中每个 Rᵏᵢ 是 2x2 旋转矩阵，0 是 2x2 零矩阵
```

虽然可以构造完整的 $d \times d$ 矩阵，但**好的解决方案应该利用这个矩阵的性质来更高效地实现变换**。

由于我们只关心给定序列内 token 的相对旋转，我们可以在层间和不同批次间**重用**计算的 $\cos(\theta_{i,k})$ 和 $\sin(\theta_{i,k})$ 值。如果你想优化，可以使用一个单独的 RoPE 模块被所有层引用，它可以在 init 时用 `self.register_buffer(persistent=False)` 创建一个 2D 的预计算 sin 和 cos 缓冲区，而不是 `nn.Parameter`（因为我们不想学习这些固定的正弦和余弦值）。

对 $q^{(i)}$ 做的完全相同的旋转过程，然后对 $k^{(j)}$ 也做，旋转使用对应的 $R^j$。注意这一层**没有可学习参数**。

> **目的与价值**：
> 1. **为什么需要位置信息？** 自注意力本身是"位置无关"的——它只看内容，不知道 token 的顺序。没有位置信息，"猫吃鱼"和"鱼吃猫"对模型来说完全一样。
> 2. **为什么用 RoPE 而非绝对位置嵌入？** RoPE 编码的是**相对位置**：两个 token 之间的注意力分数只取决于它们的**相对距离**，而非绝对位置。这在理论上更优雅，实践中也表现更好。
> 3. **旋转的直觉**：把每对维度看作 2D 平面中的一个向量。位置 $i$ 的 token 被旋转 $i$ 倍的基础角度。当计算位置 $i$ 的 query 和位置 $j$ 的 key 的点积时，旋转的差值只取决于 $i-j$，自然编码了相对位置。
> 4. **高效实现**：不要真的构建完整的 $d \times d$ 矩阵。利用分块对角结构，只需对每对维度做 2D 旋转即可。

#### 问题 (rope)：实现 RoPE（2 分）

**可交付物**：实现一个 `RotaryPositionalEmbedding` 类，将 RoPE 应用到输入张量。推荐接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None)
```

参数：

- `theta: float` — RoPE 的 $\Theta$ 值
- `d_k: int` — query 和 key 向量的维度
- `max_seq_len: int` — 将输入的最大序列长度
- `device` — 缓冲区存储设备

```python
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
```

处理形状为 `(..., seq_len, d_k)` 的输入张量，返回相同形状的张量。注意你应该容忍 x 有**任意数量的批处理维度**。假设 token_positions 是形状为 `(..., seq_len)` 的张量，指定 x 在序列维度上的 token 位置。

你应该使用 token positions 来沿序列维度切片你的（可能预计算的）cos 和 sin 张量。

实现 `adapters.run_rope`，确保通过 `uv run pytest -k test_rope`。

---

### 3.5.4 缩放点积注意力（Scaled Dot-Product Attention）

作为前置步骤，注意力操作的定义将使用 **softmax**——一种将未归一化的分数向量转化为归一化分布的操作：

$$\text{softmax}(v)_i = \frac{\exp(v_i)}{\sum_{j=1}^{n} \exp(v_j)}$$

注意，$\exp(v_i)$ 对于大的值可能变成 `inf`（然后 `inf/inf = NaN`）。我们可以通过注意到 softmax 操作对所有输入加任何常数 $c$ 是**不变的**来避免这个问题。我们可以利用这个性质来确保数值稳定性——通常，我们从所有元素中减去最大元素，使新的最大元素为 0。

> **目的与价值**：Softmax 是注意力机制的核心数学操作，也是交叉熵损失的核心。max-subtraction 技巧是深度学习中最基本的数值稳定性技巧之一——几乎所有 softmax 实现都用它。不用它，即使是中等大小的 logits 也会导致数值溢出。

#### 问题 (softmax)：实现 softmax（1 分）

**可交付物**：编写一个函数，对张量应用 softmax 操作。你的函数应接受两个参数：一个张量和一个维度 $i$，并对输入张量的第 $i$ 维应用 softmax。输出张量应与输入张量形状相同，但其第 $i$ 维现在将有归一化的概率分布。使用从第 $i$ 维所有元素中减去最大值的技巧来避免数值稳定性问题。

实现 `adapters.run_softmax`，确保通过 `uv run pytest -k test_softmax_matches_pytorch`。

---

我们现在可以如下数学定义**注意力操作**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q^{\top} K}{\sqrt{d_k}}\right) V$$

其中 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{m \times d_k}$，$V \in \mathbb{R}^{m \times d_v}$。这里 Q、K、V 都是这个操作的**输入**——注意这些不是可学习参数。如果你好奇为什么这不是 $QK^{\top}$，参见 3.3.1 节。

> **目的与价值**：
> - $Q^{\top} K$ 计算每个 query 和每个 key 之间的"相似度"——点积越大，两个向量越相似
> - 除以 $\sqrt{d_k}$ 是**缩放**——没有缩放，点积的方差会随 $d_k$ 增长，导致 softmax 输出接近 one-hot，梯度几乎为 0
> - Softmax 把相似度分数转化为概率分布——告诉我们每个位置应该"看"其他位置多少
> - 乘以 $V$ 是加权求和——用注意力权重对 value 向量求加权平均

**掩码（Masking）**：有时候我们需要遮盖注意力操作的输出。掩码形状为 $M \in \{\text{True}, \text{False}\}^{n \times m}$，每一行 $i$ 指示查询 $i$ 应该关注哪些键。按惯例（稍有些令人困惑），位置 $(i, j)$ 处的 `True` 值表示查询 $i$ **确实关注**键 $j$，`False` 表示**不关注**。换句话说，"信息流通"发生在值为 `True` 的 $(i, j)$ 对。例如，一个 $1 \times 3$ 的掩码矩阵 `[[True, True, False]]` 表示单个查询向量只关注前两个键。

在计算上，使用掩码比在子序列上计算注意力**高效得多**，我们可以通过取 softmax 前的值 $\frac{Q^{\top} K}{\sqrt{d_k}}$ 并在掩码矩阵中为 `False` 的位置加上 $-\infty$ 来实现。

#### 问题 (scaled_dot_product_attention)：实现缩放点积注意力（5 分）

**可交付物**：实现缩放点积注意力函数。你的实现应处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询，以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中 `...` 代表任意数量的其他类批处理维度（如果提供的话）。实现应返回形状为 `(batch_size, ..., d_v)` 的输出。参见第 3.3 节关于类批处理维度的讨论。

你的实现还应支持可选的用户提供的布尔掩码，形状为 `(seq_len, seq_len)`。掩码值为 `True` 的位置的注意力概率之和应为 1，掩码值为 `False` 的位置的注意力概率应为零。

实现 `adapters.run_scaled_dot_product_attention`。

- `uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试
- `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试

---

### 3.5.5 因果多头自注意力（Causal Multi-Head Self-Attention）

我们将实现 Vaswani et al. [2017] 第 3.2.2 节描述的多头自注意力。回忆一下，数学上应用多头注意力的操作定义如下：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i)$$

其中 $Q_i$、$K_i$、$V_i$ 分别是 Q、K、V 嵌入维度的第 $i \in \{1, \ldots, h\}$ 个大小为 $d_k$ 或 $d_v$ 的切片。Attention 就是 §3.5.4 中定义的缩放点积注意力操作。

由此我们可以形成多头**自**注意力操作：

$$\text{MultiHeadSelfAttention}(x) = W_O \cdot \text{MultiHead}(W_Q x,\ W_K x,\ W_V x)$$

这里，可学习参数为：

- $W_Q \in \mathbb{R}^{h d_k \times d_{\text{model}}}$
- $W_K \in \mathbb{R}^{h d_k \times d_{\text{model}}}$
- $W_V \in \mathbb{R}^{h d_v \times d_{\text{model}}}$
- $W_O \in \mathbb{R}^{d_{\text{model}} \times h d_v}$

由于 Q、K、V 在多头注意力操作中被切片，我们可以将 $W_Q$、$W_K$ 和 $W_V$ 视为沿输出维度为每个头分离。当你实现好之后，你应该用**总共三次矩阵乘法**来计算键、值和查询投影。

> **目的与价值**：
> 1. **为什么要"多头"？** 单头注意力只能学习一种"关注模式"。多头让模型同时学习多种关注模式——比如一个头关注语法关系，另一个头关注语义相似性，又一个头关注位置邻近性。
> 2. **$d_k = d_v = d_{\text{model}} / h$**：这确保总参数量和计算量与头数无关——我们只是把 $d_{\text{model}}$ 维分成了 $h$ 份。
> 3. **"自"注意力**：Q、K、V 都来自同一个输入 $x$，只是经过不同的线性投影。这让序列中的每个位置都能关注所有其他位置。
> 4. **输出投影 $W_O$**：将所有头的输出拼接后投影回 $d_{\text{model}}$ 维度。

**因果掩码（Causal masking）**。你的实现应防止模型关注序列中的**未来** token。换句话说，如果模型被给定 token 序列 $t_1, \ldots, t_n$，当我们想计算前缀 $t_1, \ldots, t_i$（其中 $i < n$）的下一词预测时，模型不应该能够访问（关注）位置 $t_{i+1}, \ldots, t_n$ 处的 token 表示，因为在推理时生成文本时它不会有这些 token（而且这些未来 token 泄漏了关于真实下一词身份的信息，使语言建模预训练目标变得无意义）。

对于输入 token 序列 $t_1, \ldots, t_n$，我们可以天真地通过运行 $n$ 次多头自注意力（对序列中的 $n$ 个唯一前缀）来防止访问未来 token。相反，我们使用**因果注意力掩码**，允许 token $i$ 关注序列中所有位置 $j \leq i$。你可以使用 `torch.triu` 或广播的索引比较来构造这个掩码，你应该利用 §3.5.4 中你的缩放点积注意力实现已经支持注意力掩码这一事实。

```
因果掩码示例（4个token）：

Query\Key  t1    t2    t3    t4
  t1      True  False False False
  t2      True  True  False False
  t3      True  True  True  False
  t4      True  True  True  True

每行 i 只能看到列 j <= i 的位置
```

> **目的与价值**：因果掩码是使 Transformer 成为**自回归语言模型**的关键。它确保模型在预测位置 $i+1$ 的词时，只使用位置 $1$ 到 $i$ 的信息——这正是生成时的情况。没有因果掩码，模型在训练时会"作弊"（偷看未来的答案）。

**应用 RoPE**。RoPE 应该应用到**查询和键向量**上，但**不应用到值向量**上。同时，头维度应该作为**批处理维度**处理，因为在多头注意力中，注意力是对每个头独立应用的。这意味着完全相同的 RoPE 旋转应该应用到每个头的查询和键向量上。

> **目的与价值**：RoPE 只作用于 Q 和 K，因为位置信息只需要影响"谁关注谁"（通过 Q·K 的点积），而不需要影响被关注的内容（V）。

#### 问题 (multihead_self_attention)：实现因果多头自注意力（5 分）

**可交付物**：将因果多头自注意力实现为 `torch.nn.Module`。你的实现应接受（至少）以下参数：

- `d_model: int` — Transformer 块输入的维度
- `num_heads: int` — 多头自注意力中使用的头数

遵循 Vaswani et al. [2017]，设 $d_k = d_v = d_{\text{model}} / h$。

实现 `adapters.run_multihead_self_attention`，然后运行 `uv run pytest -k test_multihead_self_attention`。

---

## 3.6 完整的 Transformer LM

我们先来组装 Transformer 块（参考图 2 会很有帮助）。一个 Transformer 块包含两个"子层"，一个用于多头自注意力，另一个用于前馈网络。在每个子层中，我们首先执行 RMSNorm，然后执行主操作（MHA/FF），最后加入残差连接。

具体来说，Transformer 块的前半部分（第一个"子层"）应该实现以下更新，从输入 $x$ 产生输出 $y$：

$$y = x + \text{MultiHeadSelfAttention}(\text{RMSNorm}(x))$$

> **目的与价值**：这就是 pre-norm 的具体公式。注意顺序：先 Norm → 再注意力/FFN → 再残差加法。残差连接让原始信号 $x$ 可以不经修改地通过，子层只需要学习"需要添加什么修正"。这使得深度网络更容易训练。

#### 问题 (transformer_block)：实现 Transformer 块（3 分）

实现 §3.5 中描述的、图 2 中展示的 pre-norm Transformer 块。你的 Transformer 块应接受（至少）以下参数：

- `d_model: int` — Transformer 块输入的维度
- `num_heads: int` — 多头自注意力中使用的头数
- `d_ff: int` — 逐位置前馈内层的维度

实现 `adapters.run_transformer_block`，然后运行 `uv run pytest -k test_transformer_block`。

**可交付物**：通过提供测试的 Transformer 块代码。

---

现在我们把所有块组合在一起，遵循图 1 的高层图示。按照第 3.1.1 节的嵌入描述，将其输入到 `num_layers` 个 Transformer 块中，然后传入三个输出层以获得词汇表上的分布。

#### 问题 (transformer_lm)：实现 Transformer LM（3 分）

是时候把一切组合在一起了！实现 §3.1 中描述的、图 1 中展示的 Transformer 语言模型。你的实现至少应接受 Transformer 块的所有前述构造参数，加上这些额外参数：

- `vocab_size: int` — 词汇表大小，用于确定 token 嵌入矩阵的维度
- `context_length: int` — 最大上下文长度，用于确定位置嵌入矩阵的维度
- `num_layers: int` — 使用的 Transformer 块数量

实现 `adapters.run_transformer_lm`，然后运行 `uv run pytest -k test_transformer_lm`。

**可交付物**：通过上述测试的 Transformer LM 模块。

> **目的与价值**：这是整个模型架构的"集大成"。完整的数据流是：
>
> ```
> 整数 token IDs → Embedding → [Transformer Block x num_layers] → RMSNorm → Linear → logits
> ```
>
> 每一步都有明确的作用：Embedding 把离散 ID 变成连续向量，Transformer 块逐层提取特征，最终的 Linear 把特征投影到词汇表维度产生预测。

---

## 资源核算（Resource Accounting）

能够理解 Transformer 各部分如何消耗计算和内存是很有用的。我们将走过做一些基本 "FLOPs 核算"的步骤。Transformer 中**绝大多数 FLOPS 都是矩阵乘法**，所以我们的核心方法很简单：

1. 列出 Transformer 前向传播中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的 FLOPs。

对于第二步，以下事实很有用：

> **规则**：给定 $A \in \mathbb{R}^{m \times n}$ 和 $B \in \mathbb{R}^{n \times p}$，矩阵乘积 $AB$ 需要 $2mnp$ FLOPs。

要看到这一点，注意 $(AB)[i, j] = A[i, :] \cdot B[:, j]$，这个点积需要 $n$ 次加法和 $n$ 次乘法（$2n$ FLOPs）。然后，由于矩阵乘积 $AB$ 有 $m \times p$ 个元素，总 FLOPS 为 $(2n)(mp) = 2mnp$。

> **目的与价值**：FLOPs 核算是理解模型效率的基础。它告诉你：
> - 模型的哪些部分最"昂贵"
> - 增大哪个超参数对计算量影响最大
> - 训练一个模型需要多少 GPU 时间
>
> 这是做系统级优化决策的前提——比如该不该增大模型、该用多大的上下文长度等。

#### 问题 (transformer_accounting)：Transformer LM 资源核算（5 分）

**(a)** 考虑 GPT-2 XL，配置如下：

| 参数 | 值 |
|------|------|
| vocab_size | 50,257 |
| context_length | 1,024 |
| num_layers | 48 |
| d_model | 1,600 |
| num_heads | 25 |
| d_ff | 6,400 |

假设我们用这个配置构建模型。模型有多少**可训练参数**？假设每个参数用单精度浮点表示，仅加载模型需要多少**内存**？

**可交付物**：一到两句话的回答。

**(b)** 识别完成 GPT-2 XL 模型一次前向传播所需的矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列有 `context_length` 个 token。

**可交付物**：矩阵乘法列表（带描述），以及所需的总 FLOPs 数。

**(c)** 根据上面的分析，模型的哪些部分需要最多的 FLOPs？

**可交付物**：一到两句话的回答。

**(d)** 用 GPT-2 small（12 层，768 d_model，12 头）、GPT-2 medium（24 层，1024 d_model，16 头）和 GPT-2 large（36 层，1280 d_model，20 头）重复你的分析。随着模型大小增加，Transformer LM 的哪些部分占总 FLOPs 的比例变大或变小？

**可交付物**：每个模型的组件 FLOPs 分解（占前向传播总 FLOPs 的比例），以及一到两句话描述模型大小变化如何改变各组件的 FLOPs 比例。

**(e)** 取 GPT-2 XL 并将上下文长度增加到 16,384。一次前向传播的总 FLOPs 如何变化？模型各组件的 FLOPs 相对贡献如何变化？

**可交付物**：一到两句话的回答。

> **目的与价值**：这些计算题让你建立对模型规模的直觉。关键发现通常是：
> - 注意力的 FLOPs 与序列长度**平方**成正比（$Q^{\top}K$ 操作），而 FFN 的 FLOPs 与序列长度线性成正比
> - 对于短序列，FFN 占主导；对于长序列，注意力占主导
> - 这解释了为什么长上下文模型是一个活跃的研究方向——注意力的二次复杂度是主要瓶颈

---

## 全章总结

第 3 章从零构建了完整的 Transformer 语言模型。核心组件链条是：

```
Embedding → [RMSNorm → MHA(+RoPE) → Add → RMSNorm → SwiGLU FFN → Add] x N → RMSNorm → Linear
```

每个组件都有明确的设计动机：

| 组件 | 作用 |
|------|------|
| **Embedding** | 离散 → 连续 |
| **RMSNorm** | 稳定激活值尺度 |
| **Multi-Head Attention + RoPE** | 建模 token 间关系 + 位置感知 |
| **SwiGLU FFN** | 非线性特征变换 |
| **残差连接** | 确保梯度流通 |
| **因果掩码** | 保证自回归性质 |
| **最终 Linear** | 连续 → 词汇表分布 |
