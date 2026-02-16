# 第4-7章：训练与实验（教学翻译）

---

## 4 训练 Transformer 语言模型

我们现在已经有了数据预处理（分词器）和模型（Transformer）的步骤。剩下的工作是构建所有支持训练的代码。这包括以下部分：

- **损失函数**：需要定义损失函数（交叉熵）
- **优化器**：需要定义最小化损失的优化器（AdamW）
- **训练循环**：需要所有支持基础设施，包括数据加载、保存检查点和管理训练过程

---

## 4.1 交叉熵损失（Cross-Entropy Loss）

回顾一下，Transformer 语言模型对长度为 $m+1$ 的序列 $x$ 中的每个 $i = 1, \ldots, m$，定义了分布 $p_\theta(x_{i+1} | x_{1:i})$。给定由长度为 $m$ 的序列组成的训练集 $\mathcal{D}$，我们定义标准交叉熵（负对数似然）损失函数：

$$\ell(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|m} \sum_{x \in \mathcal{D}} \sum_{i=1}^{m} -\log p_\theta(x_{i+1} | x_{1:i}) \tag{16}$$

（注意，Transformer 的一次前向传播即可得到所有 $i = 1, \ldots, m$ 的 $p_\theta(x_{i+1} | x_{1:i})$。）

具体来说，Transformer 在每个位置 $i$ 处计算 logits $o_i \in \mathbb{R}^{\text{vocab\_size}}$，从而得到：

$$p(x_{i+1} | x_{1:i}) = \text{softmax}(o_i)[x_{i+1}] = \frac{\exp(o_i[x_{i+1}])}{\sum_{a=1}^{\text{vocab\_size}} \exp(o_i[a])} \tag{17}$$

交叉熵损失通常以 logits 向量 $o_i \in \mathbb{R}^{\text{vocab\_size}}$ 和目标 $x_{i+1}$ 为参数来定义。

实现交叉熵损失需要像处理 softmax 一样注意数值问题。

> **目的与价值**：交叉熵是语言模型最自然的训练目标——它衡量模型对真实下一词的惊讶程度（surprisal）。通过最小化交叉熵，模型被迫学习对语言分布的良好概率估计。max-subtraction 技巧（同 softmax）在实现中同样关键，且还可以利用 log-exp 相消来进一步简化计算。

### 问题 (cross_entropy)：实现交叉熵损失

**可交付物**：编写一个函数计算交叉熵损失，接受预测 logits（$o_i$）和目标（$x_{i+1}$），计算交叉熵 $\ell_i = -\log \text{softmax}(o_i)[x_{i+1}]$。函数应处理以下情况：

- 减去最大元素以确保数值稳定性
- 尽可能消去 log 和 exp
- 处理任意额外的批处理维度，并返回批次上的平均值（类批处理维度始终在词汇表大小维度之前）

实现 `adapters.run_cross_entropy`，然后运行 `uv run pytest -k test_cross_entropy`。

**困惑度（Perplexity）**。交叉熵足以用于训练，但在评估模型时，我们还需要报告困惑度。对于长度为 $m$ 的序列，承受交叉熵损失 $\ell_1, \ldots, \ell_m$，则：

$$\text{perplexity} = \exp\left(\frac{1}{m} \sum_{i=1}^{m} \ell_i\right) \tag{18}$$

> **目的与价值**：困惑度（perplexity）是交叉熵损失的指数变换，更直观地衡量"模型平均有多少个词需要猜测"。困惑度为 100 意味着模型在每个位置有效地从 100 个等概率词中选择。相比原始 bits/token，困惑度更容易与人类直觉对比。

---

## 4.2 SGD 优化器

现在有了损失函数，我们开始探索优化器。最简单的基于梯度的优化器是**随机梯度下降**（SGD）。我们从随机初始化的参数 $\theta_0$ 开始，然后对每一步 $t = 0, \ldots, T-1$，执行以下更新：

$$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla L(\theta_t; \mathcal{B}_t) \tag{19}$$

其中 $\mathcal{B}_t$ 是从数据集 $\mathcal{D}$ 中随机采样的一批数据，学习率 $\alpha_t$ 和批次大小 $|\mathcal{B}_t|$ 是超参数。

### 4.2.1 在 PyTorch 中实现 SGD

要实现我们的优化器，需要子类化 PyTorch 的 `torch.optim.Optimizer` 类。`Optimizer` 子类必须实现两个方法：

`def __init__(self, params, ...)` — 初始化优化器。`params` 是要优化的参数集合（或参数组，如果用户想对模型的不同部分使用不同超参数）。确保将 `params` 传递给基类的 `__init__` 方法，它会存储这些参数供 `step` 使用。

`def step(self)` — 执行一次参数更新。在训练循环中，这将在反向传播之后调用，因此你可以访问最后一批次上的梯度。此方法应遍历每个参数张量 `p` 并**原地修改**它们，即根据梯度 `p.grad`（如果存在）设置 `p.data`。

以下是一个实现了带学习率衰减的 SGD 变体的示例：

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{t+1}} \nabla L(\theta_t; \mathcal{B}_t) \tag{20}$$

```python
from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # 获取学习率
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # 获取与 p 关联的状态
                t = state.get("t", 0)  # 获取迭代次数，或初始值
                grad = p.grad.data  # 获取损失对 p 的梯度
                p.data -= lr / math.sqrt(t + 1) * grad  # 原地更新权重张量
                state["t"] = t + 1  # 递增迭代次数
        return loss
```

训练循环的典型结构如下：

```python
weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1)
for t in range(100):
    opt.zero_grad()   # 重置所有可学习参数的梯度
    loss = (weights**2).mean()  # 计算标量损失值
    print(loss.cpu().item())
    loss.backward()   # 运行反向传播，计算梯度
    opt.step()        # 运行优化器步骤
```

> **目的与价值**：这段代码展示了 PyTorch 优化器 API 的标准模式。`param_groups` 允许对不同参数使用不同超参数（如不同学习率），`self.state[p]` 是每个参数的持久状态存储（对 AdamW 来说就是动量估计）。这个 SGD 变体展示了带衰减调度的基本思路，但 SGD 在训练大语言模型时通常不如 Adam 系列优化器。

### 问题 (learning_rate_tuning)：调整学习率（1 分）

用三个其他学习率值（`1e1`、`1e2`、`1e3`）运行上面的 SGD 示例，只训练 10 次迭代。每个学习率下损失有什么变化？是衰减更快、更慢，还是发散（即在训练过程中增大）？

**可交付物**：一到两句话描述你观察到的行为。

---

## 4.3 AdamW 优化器

现代语言模型通常使用比 SGD 更复杂的优化器来训练。最近使用的大多数优化器都是 Adam 优化器 [Kingma and Ba, 2015] 的变体。我们将使用 **AdamW** [Loshchilov and Hutter, 2019]，它在近期工作中被广泛使用。AdamW 提出了对 Adam 的修改，通过添加**权重衰减**（每次迭代将参数拉向 0）来改善正则化，且权重衰减与梯度更新解耦。我们将按照 Loshchilov and Hutter [2019] 算法 2 来实现 AdamW。

AdamW 是**有状态的**：对每个参数，它跟踪其一阶矩和二阶矩的滑动估计。因此，AdamW 使用额外内存换取更好的稳定性和收敛性。

除学习率 $\alpha$ 外，AdamW 还有控制动量估计更新的超参数 $(\beta_1, \beta_2)$ 和权重衰减率 $\lambda$。典型设置中 $(\beta_1, \beta_2) = (0.9, 0.999)$，但大型语言模型（如 LLaMA、GPT-3）通常使用 $(0.9, 0.95)$。算法如下：

**算法 1：AdamW 优化器**

```
初始化 θ（可学习参数）
m ← 0         （一阶矩向量的初始值；与 θ 形状相同）
v ← 0         （二阶矩向量的初始值；与 θ 形状相同）

for t = 1, ..., T do
    采样批次数据 B_t
    g ← ∇_θ ℓ(θ; B_t)                （计算当前步骤的损失梯度）
    m ← β₁m + (1 - β₁)g              （更新一阶矩估计）
    v ← β₂v + (1 - β₂)g²             （更新二阶矩估计）
    α_t ← α · √(1-(β₂)^t) / (1-(β₁)^t)  （计算第 t 次迭代的调整 α）
    θ ← θ - α_t · m / (√v + ε)       （更新参数）
    θ ← θ - αλθ                       （应用权重衰减）
end for
```

注意 $t$ 从 1 开始。

> **目的与价值**：AdamW 的核心创新是：
> 1. **自适应学习率**：每个参数有自己的有效学习率，由梯度的一阶矩（方向）和二阶矩（幅度）共同决定。高频梯度的参数获得更小的更新步长。
> 2. **偏差修正**（$\sqrt{1-\beta_2^t} / (1-\beta_1^t)$）：初期动量估计偏向 0，偏差修正让早期步骤也能正确运作。
> 3. **解耦权重衰减**：原版 Adam 的 L2 正则化与自适应学习率交互（效果被除以 $\sqrt{v}$），而 AdamW 直接在参数上应用权重衰减，效果更纯粹。

### 问题 (adamw)：实现 AdamW（2 分）

**可交付物**：将 AdamW 优化器实现为 `torch.optim.Optimizer` 的子类。类应在 `__init__` 中接受学习率 $\alpha$ 以及 $\beta$、$\varepsilon$ 和 $\lambda$ 超参数。实现 `adapters.get_adamw_cls`，确保通过 `uv run pytest -k test_adamw`。

### 问题 (adamwAccounting)：AdamW 训练的资源核算（2 分）

计算运行 AdamW 所需的内存和计算量。假设所有张量使用 `float32`。

**(a)** 运行 AdamW 需要多少峰值内存？按参数、激活值、梯度和优化器状态分解你的答案。用 `batch_size` 和模型超参数（`vocab_size`、`context_length`、`num_layers`、`d_model`、`num_heads`）表示答案。假设 `d_ff = 4 × d_model`。

计算激活值内存时，只考虑以下组件：
- Transformer 块：RMSNorm，多头自注意力子层（QKV 投影、$Q^T K$ 矩阵乘法、softmax、值的加权求和、输出投影），逐位置前馈（$W_1$ 矩阵乘法、SiLU、$W_2$ 矩阵乘法）
- 最终 RMSNorm
- 输出嵌入
- logits 上的交叉熵

**可交付物**：参数、激活值、梯度和优化器状态各自的代数表达式，以及总量。

**(b)** 将你的答案代入 GPT-2 XL 形状的模型，得到一个只依赖 `batch_size` 的表达式。在 80GB 内存限制内，最大批次大小是多少？

**可交付物**：形如 $a \cdot \text{batch\_size} + b$ 的表达式（给出数值 $a, b$）以及最大批次大小的数值。

**(c)** 运行一步 AdamW 需要多少 FLOPs？

**可交付物**：代数表达式，附简短说明。

**(d)** **模型 FLOPs 利用率**（MFU）定义为观测到的吞吐量（token/秒）与硬件理论峰值 FLOP 吞吐量的比值。NVIDIA A100 GPU 的 `float32` 运算理论峰值为 19.5 teraFLOP/s。假设你能达到 50% MFU，在单张 A100 上以批次大小 1024 训练 GPT-2 XL 400K 步需要多长时间？（假设反向传播的 FLOPs 是前向传播的两倍。）

**可交付物**：训练所需天数，附简短说明。

---

## 4.4 学习率调度（Learning Rate Scheduling）

在训练过程中，导致损失最快下降的学习率值往往会变化。在训练 Transformer 时，通常使用**学习率调度**——以较大的学习率开始，在初期快速更新，然后随着模型训练逐渐衰减到较小的值。

在本作业中，我们将实现训练 LLaMA 所用的**余弦退火调度**（cosine annealing schedule）[Touvron et al., 2023]。

余弦退火学习率调度接受以下参数：(i) 当前迭代步 $t$，(ii) 最大学习率 $\alpha_{\max}$，(iii) 最小（最终）学习率 $\alpha_{\min}$，(iv) 预热迭代次数 $T_w$，(v) 余弦退火迭代次数 $T_c$。第 $t$ 步的学习率定义为：

$$\text{（预热阶段）} \quad \text{若 } t < T_w, \quad \alpha_t = \frac{t}{T_w} \alpha_{\max}$$

$$\text{（余弦退火）} \quad \text{若 } T_w \leq t \leq T_c, \quad \alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{\max} - \alpha_{\min})$$

$$\text{（退火后）} \quad \text{若 } t > T_c, \quad \alpha_t = \alpha_{\min}$$

> **目的与价值**：学习率调度是现代深度学习训练的标配。
> - **预热阶段**：训练初期参数随机且动量估计不可靠，大学习率可能导致不稳定。线性预热让优化器先"热身"。
> - **余弦退火**：余弦曲线比线性衰减更平滑——在训练中期衰减较慢（还有很多学习空间），在训练末期衰减更快（微调到最终值）。
> - **重要细节**：余弦调度应该恰好在最后一个训练步衰减到 $\alpha_{\min}$（即 $T_c$ = 总训练步数），这样才能充分利用训练预算。

### 问题 (learning_rate_schedule)：实现带预热的余弦学习率调度

编写一个函数，接受 $t$、$\alpha_{\max}$、$\alpha_{\min}$、$T_w$ 和 $T_c$，根据上述调度器定义返回学习率 $\alpha_t$。然后实现 `adapters.get_lr_cosine_schedule`，确保通过 `uv run pytest -k test_get_lr_cosine_schedule`。

---

## 4.5 梯度裁剪（Gradient Clipping）

在训练过程中，有时会遇到产生大梯度的训练样本，这会使训练不稳定。为了缓解这个问题，实践中经常使用**梯度裁剪**技术：在每次反向传播之后、执行优化器步骤之前，对梯度的范数施加限制。

给定（所有参数的）梯度 $g$，计算其 $\ell_2$ 范数 $\|g\|_2$。如果该范数小于最大值 $M$，则保持 $g$ 不变；否则，将 $g$ 缩小 $\frac{M}{\|g\|_2 + \varepsilon}$ 倍（其中加入小的 $\varepsilon$（如 $10^{-6}$）以保证数值稳定性）。注意，裁剪后的范数将略小于 $M$。

> **目的与价值**：梯度裁剪是防止"梯度爆炸"的实用工具。对于 Transformer 来说，长序列上的注意力机制有时会产生数值异常大的梯度，尤其是在训练初期。裁剪确保了单次更新步骤不会把参数推到太远的位置，提高了训练的鲁棒性。

### 问题 (gradient_clipping)：实现梯度裁剪（1 分）

编写一个实现梯度裁剪的函数。函数应接受参数列表和最大 $\ell_2$ 范数，原地修改每个参数的梯度。使用 $\varepsilon = 10^{-6}$（PyTorch 默认值）。然后实现适配器 `adapters.run_gradient_clipping`，确保通过 `uv run pytest -k test_gradient_clipping`。

---

## 5 训练循环

我们现在终于可以将到目前为止实现的主要组件放在一起：分词后的数据、模型和优化器。

---

## 5.1 数据加载器（Data Loader）

分词后的数据（如你在 `tokenizer_experiments` 中准备的）是单个 token 序列 $x = (x_1, \ldots, x_n)$。即使源数据可能由独立的文档组成（如不同的网页或源代码文件），一种常见做法是将所有内容拼接成单个 token 序列，在它们之间添加分隔符（如 `<|endoftext|>` token）。

**数据加载器**将这个序列转换为批次流，每个批次由 $B$ 个长度为 $m$ 的序列组成，并与对应的下一个 token（同样长度为 $m$）配对。例如，对于 $B=1, m=3$，`([x₂, x₃, x₄], [x₃, x₄, x₅])` 是一个潜在的批次。

这种数据加载方式简化了训练，原因如下：第一，任何 $1 \leq i < n-m$ 都是有效的训练序列起始位置，因此序列采样非常简单；第二，所有训练序列长度相同，无需填充，提高了硬件利用率；最后，我们不需要将完整数据集加载到内存中才能采样训练数据，便于处理可能无法装入内存的大型数据集。

> **目的与价值**：这种"滑动窗口"数据加载方式是训练大型语言模型的标准做法。它的优雅之处在于：不需要显式管理文档边界（`<|endoftext|>` 已处理），所有批次格式统一（无需 padding），且天然适配自回归训练目标（输入序列与目标序列只差一个位置）。

**内存映射（Memory Mapping）**。如果数据集太大，无法加载到内存，可以使用 Unix 系统调用 `mmap`，将磁盘上的文件映射到虚拟内存，并在访问该内存位置时懒惰地加载文件内容。NumPy 通过 `np.memmap`（或保存时用 `np.save`、加载时用 `np.load` 的 `mmap_mode='r'` 标志）来实现这一点，返回一个按需加载条目的类数组对象。训练期间从数据集（即 NumPy 数组）采样时，务必以内存映射模式加载数据集，并指定与加载数组匹配的 `dtype`。

### 问题 (data_loading)：实现数据加载（2 分）

**可交付物**：编写一个函数，接受 NumPy 数组 `x`（带有 token ID 的整数数组）、`batch_size`、`context_length` 和 PyTorch 设备字符串（如 `'cpu'` 或 `'cuda:0'`），返回一对张量：采样的输入序列和对应的下一 token 目标。两个张量的形状均为 `(batch_size, context_length)`，包含 token ID，并置于请求的设备上。

实现 `adapters.run_get_batch`，然后运行 `uv run pytest -k test_get_batch`。

**低资源提示：CPU 或 Apple Silicon 上的数据加载**

若你计划在 CPU 或 Apple Silicon 上训练，需要将数据移动到正确的设备。CPU 使用 `'cpu'` 设备字符串，Apple Silicon（M 系列芯片）使用 `'mps'` 设备字符串。

---

## 5.2 检查点保存（Checkpointing）

除了加载数据，我们还需要在训练过程中保存模型。运行作业时，通常希望能够恢复因某种原因中途停止的训练运行（如作业超时、机器故障等）。即使一切顺利，我们也可能想在事后访问中间模型（如研究训练动态、从训练不同阶段的模型生成样本等）。

检查点应包含恢复训练所需的所有状态：最低限度需要恢复模型权重；使用有状态优化器（如 AdamW）时，还需要保存优化器状态（如 AdamW 的动量估计）；为了恢复学习率调度，还需要知道停止时的迭代次数。

PyTorch 使保存这些内容变得容易：每个 `nn.Module` 都有 `state_dict()` 方法，返回包含所有可学习权重的字典；可以用 `load_state_dict()` 方法恢复这些权重。`torch.optim.Optimizer` 同样如此。此外，`torch.save(obj, dest)` 可以将对象转储到文件（路径）或类文件对象，之后可以用 `torch.load(src)` 加载回内存。

### 问题 (checkpointing)：实现模型检查点保存（1 分）

实现以下两个用于加载和保存检查点的函数：

`def save_checkpoint(model, optimizer, iteration, out)` — 将前三个参数的所有状态转储到类文件对象 `out` 中。可以使用模型和优化器的 `state_dict` 方法获取其相关状态，并使用 `torch.save(obj, out)` 将 `obj` 转储到 `out`。典型选择是将 `obj` 作为字典。参数：
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`
- `iteration: int`
- `out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`

`def load_checkpoint(src, model, optimizer)` — 从 `src`（路径或类文件对象）加载检查点，从中恢复模型和优化器状态。函数应返回保存到检查点的迭代次数。参数：
- `src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]`
- `model: torch.nn.Module`
- `optimizer: torch.optim.Optimizer`

实现 `adapters.run_save_checkpoint` 和 `adapters.run_load_checkpoint`，确保通过 `uv run pytest -k test_checkpointing`。

> **目的与价值**：检查点是生产级训练的必备功能。大型模型训练可能持续数天甚至数周，任何意外中断（机器故障、时间限制、抢占式作业调度）都不能导致之前的计算白费。保存完整的优化器状态（而非只保存模型权重）确保了训练可以**无缝续训**——从数值上完全等价于从未中断过的训练。

---

## 5.3 训练循环（Training Loop）

现在，终于可以将实现的所有组件组合成主训练脚本了。投入精力让训练运行易于以不同超参数启动（如通过命令行参数）是值得的，因为之后你将多次运行这些实验来研究不同选择的影响。

### 问题 (training_together)：整合（4 分）

**可交付物**：编写一个脚本，运行训练循环来在用户提供的输入上训练模型。特别是，建议你的训练脚本至少支持以下功能：

- 能够配置和控制各种模型和优化器超参数
- 使用 `np.memmap` 高效加载大型训练和验证数据集
- 将检查点序列化到用户提供的路径
- 定期记录训练和验证性能（如到控制台和/或外部服务，例如 [Weights and Biases](https://wandb.ai)）

> **目的与价值**：良好的训练脚本是提高实验效率的关键。命令行参数配置让你可以轻松提交带不同超参数的批量实验；`np.memmap` 避免了大型数据集的内存溢出；定期验证损失记录让你能够监控过拟合、选择最佳检查点，并绘制学习曲线（这是后面实验部分的必要条件）。

---

## 6 文本生成

现在我们可以训练模型了，最后需要的一个组件是从模型中**生成文本**的能力。

回顾一下，语言模型接受一个（可能是批量的）长度为 `sequence_length` 的整数序列，输出大小为 `(sequence_length × vocab_size)` 的矩阵，其中每个序列元素都是预测该位置之后下一个词的概率分布。我们现在将编写一些函数，将这个过程转化为生成新序列的采样方案。

**Softmax**。按照标准惯例，语言模型输出的是最终线性层的输出（"logits"），因此需要通过 softmax 操作将其转化为归一化概率。

**解码（Decoding）**。为了从模型中生成文本，我们向模型提供一组前缀 token（"提示"），并要求它生成预测序列中下一个词的词汇表概率分布。然后，我们从该词汇表项的分布中采样，确定下一个输出 token。

具体来说，解码过程的一个步骤接受序列 $x_{1\ldots t}$，通过以下公式返回 token $x_{t+1}$：

$$P(x_{t+1} = i | x_{1\ldots t}) = \frac{\exp(v_i)}{\sum_j \exp(v_j)}$$

$$v = \text{TransformerLM}(x_{1\ldots t})_t \in \mathbb{R}^{\text{vocab\_size}}$$

其中 TransformerLM 是我们的模型，它接受 `sequence_length` 长度的序列，输出大小为 `(sequence_length × vocab_size)` 的矩阵，我们取该矩阵的最后一个元素，因为我们在寻找第 $t$ 个位置的下一词预测。

通过反复从这些单步条件概率中采样（将之前生成的输出 token 追加到下一次解码步骤的输入中），直到生成序列结束 token `<|endoftext|>`（或用户指定的最大生成 token 数），我们得到了一个基本的解码器。

> **目的与价值**：这就是**自回归生成**的本质——每一步只预测下一个词，然后把它追加到上下文里再预测下一个词。这种方式与训练完全一致（训练时也是逐位置预测下一词），但推理时是**串行**的（不能并行化）。因果掩码在此发挥关键作用：确保每一步的预测只用到了已经生成的内容。

**解码技巧**。我们将实验小型模型，小模型有时会生成质量较低的文本。两个简单的解码技巧有助于解决这些问题。

**温度缩放（Temperature Scaling）**。第一种技巧是温度缩放，我们用温度参数 $\tau$ 修改 softmax：

$$\text{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\text{vocab\_size}|} \exp(v_j / \tau)} \tag{24}$$

注意，当 $\tau \to 0$ 时，$v$ 的最大元素占主导，softmax 的输出变成集中在该最大元素处的 one-hot 向量（确定性解码）；当 $\tau > 1$ 时，分布变得更平坦（更随机）。

**Nucleus / Top-p 采样（Nucleus Sampling）**。第二种技巧是核采样或 Top-p 采样，通过截断低概率词来修改采样分布。设 $q$ 是从（温度缩放后的）softmax 得到的大小为 `vocab_size` 的概率分布。具有超参数 $p$ 的核采样按以下公式生成下一个 token：

$$P(x_{t+1} = i | q) = \begin{cases} \frac{q_i}{\sum_{j \in V(p)} q_j} & \text{若 } i \in V(p) \\ 0 & \text{否则} \end{cases}$$

其中 $V(p)$ 是满足 $\sum_{j \in V(p)} q_j \geq p$ 的最小索引集合。可以通过先按大小对概率分布 $q$ 排序，然后选择最大的词汇表元素直到达到目标概率 $p$ 来计算这个量。

> **目的与价值**：
> - **温度**控制的是随机性与确定性的权衡。低温（$\tau < 1$）使分布更尖锐，生成更保守但可能重复；高温（$\tau > 1$）增加多样性但可能生成无意义内容。
> - **Top-p 采样**解决了这样一个问题：有时前几个 token 的概率极高，剩余 token 的概率极低但数量庞大。Top-p 动态地确定"有意义的"候选集合大小（$p=0.9$ 意味着只从累计概率达到 90% 的最高概率 token 中采样）。

### 问题 (decoding)：解码（3 分）

**可交付物**：实现一个从语言模型解码的函数。建议支持以下功能：

- 为用户提供的提示生成补全（即接受 $x_{1\ldots t}$ 并采样补全，直到遇到 `<|endoftext|>` token）
- 允许用户控制生成 token 的最大数量
- 给定温度值，在采样前对预测的下一词分布应用 softmax 温度缩放
- Top-p 采样（Holtzman et al., 2020；也称为核采样），给定用户指定的阈值

---

## 7 实验

现在是时候将所有内容整合起来，在预训练数据集上训练（小型）语言模型了。

---

## 7.1 如何运行实验与可交付物

理解 Transformer 架构组件背后原理的最佳方式是亲自修改并运行它。没有什么能替代实践经验。

为此，重要的是能够**快速、一致地进行实验并保留记录**。为了快速实验，我们将在小规模模型（17M 参数）和简单数据集（TinyStories）上运行许多实验。为了一致性，你将以系统化的方式消融组件和调整超参数，为了保留记录，我们要求你提交实验日志和每个实验对应的学习曲线。

为了能够提交损失曲线，确保定期评估验证损失并**同时记录步骤数和墙上时钟时间**。你可能会发现日志基础设施（如 Weights and Biases）很有帮助。

### 问题 (experiment_log)：实验记录（3 分）

为你的训练和评估代码创建实验追踪基础设施，允许你追踪相对于梯度步数和墙上时钟时间的实验和损失曲线。

**可交付物**：用于实验的日志基础设施代码，以及本节中所有作业问题的实验日志（记录你尝试过的所有内容）。

---

## 7.2 TinyStories

我们从一个非常简单的数据集（TinyStories；Eldan and Li, 2023）开始，模型将训练得很快，我们可以观察到一些有趣的行为。下面是该数据集的一个示例：

> **TinyStories 示例**：Once upon a time there was a little boy named Ben...（略，见原文）

### 超参数调整

以下是一些基本超参数，以及你需要寻找的其他设置：

| 超参数 | 值 |
|--------|-----|
| vocab_size | 10,000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1,344（约 $\frac{8}{3} \times d_{\text{model}}$，且为 64 的倍数） |
| RoPE theta $\Theta$ | 10,000 |
| 层数 / 头数 | 4 层，16 头（约 17M 非嵌入参数） |
| 总训练 token 数 | 327,680,000（batch_size × 总步数 × context_length ≈ 此值） |

你需要通过试验找到以下超参数的良好默认值：**学习率**、**学习率预热**、其他 AdamW 超参数（$\beta_1, \beta_2, \varepsilon$）和**权重衰减**。可以在 Kingma and Ba [2015] 中找到一些典型选择。

**调试模型架构的技巧**：强烈建议你熟悉 IDE 的内置调试器（如 VSCode/PyCharm）。调试神经网络架构的一些好做法是：

- 一个常见的第一步是**在单个小批次上过拟合**。如果你的实现是正确的，应该能迅速将训练损失驱动到接近零。
- 在各个模型组件中设置调试断点，检查中间张量的形状是否符合预期。
- 监控激活值、模型权重和梯度的范数，确保它们没有爆炸或消失。

### 问题 (learning_rate)：调整学习率（3 分）（4 个 H100 小时）

**(a)** 对学习率进行超参数搜索，报告最终损失（或在优化器发散时注明）。

**可交付物**：与多个学习率关联的学习曲线。解释你的超参数搜索策略。

**可交付物**：一个在 TinyStories 上验证损失（每 token）至多为 **1.45** 的模型。

**低资源提示（CPU/Apple Silicon）**：将总 token 数减少到 40,000,000，目标验证损失可放宽到 **2.00**。

**(b)** 民间智慧是最佳学习率"在稳定边缘"。研究学习率发散的临界点与最佳学习率的关系。

**可交付物**：不断增大学习率的学习曲线，至少包含一条发散运行，以及关于这与收敛率关系的分析。

### 问题 (batch_size_experiment)：批次大小变化（1 分）（2 个 H100 小时）

从 1 到 GPU 内存上限不断调整批次大小。至少尝试几个中间值，包括 64 和 128 等典型大小。

**可交付物**：不同批次大小运行的学习曲线。如有必要，重新优化学习率。

**可交付物**：关于批次大小发现及其对训练影响的几句话讨论。

### 问题 (generate)：生成文本（1 分）

使用你的解码器和训练好的检查点，报告模型生成的文本。可能需要调整解码参数（温度、top-p 等）来获得流畅的输出。

**可交付物**：至少 256 个 token 的文本转储（或直到第一个 `<|endoftext|>` token），以及对输出流畅性的简短评论，以及至少两个影响输出好坏的因素。

---

## 7.3 消融与架构修改

理解 Transformer 的最佳方式是亲自修改它并观察其行为。

### 消融 1：层归一化

层归一化对于 Transformer 训练的稳定性常被提及。让我们移除每个 Transformer 块中的 RMSNorm，看看会发生什么。

### 问题 (layer_norm_ablation)：移除 RMSNorm 并训练（1 分）（1 个 H100 小时）

从 Transformer 中移除所有 RMSNorm 并训练。在之前最优学习率下会发生什么？能否通过使用更低的学习率获得稳定性？

**可交付物**：移除 RMSNorm 并训练的学习曲线，以及最佳学习率的学习曲线。

**可交付物**：关于 RMSNorm 影响的几句话评论。

### 问题 (pre_norm_ablation)：实现后归一化并训练（1 分）（1 个 H100 小时）

Pre-norm Transformer 块定义为：

$$z = x + \text{MultiHeadedSelfAttention}(\text{RMSNorm}(x))$$
$$y = z + \text{FFN}(\text{RMSNorm}(z))$$

这是对原始 Transformer 使用后归一化（post-norm）方法的少数几种"共识"修改之一，原始 post-norm 方法为：

$$z = \text{RMSNorm}(x + \text{MultiHeadedSelfAttention}(x))$$
$$y = \text{RMSNorm}(z + \text{FFN}(z))$$

将 pre-norm Transformer 实现修改为 post-norm，训练后看看会发生什么。

**可交付物**：post-norm Transformer 与 pre-norm 的对比学习曲线。

> **目的与价值**：这个消融实验让你亲身体会 pre-norm 与 post-norm 的差异。理论分析预测 pre-norm 更稳定，实验验证这一点。值得注意的是，post-norm 也不是完全无法训练——通常只是需要更低的学习率和更长的预热。

### 消融 2：位置嵌入

接下来研究位置嵌入对模型性能的影响。具体来说，我们将比较基础模型（带 RoPE）与完全不包含位置嵌入（NoPE）的情况。事实证明，带因果掩码的仅解码器 Transformer 理论上可以在不显式提供位置嵌入的情况下推断相对或绝对位置信息 [Tsai et al., 2019, Kazemnejad et al., 2023]。

### 问题 (no_pos_emb)：实现 NoPE（1 分）（1 个 H100 小时）

将带 RoPE 的 Transformer 实现修改为完全移除位置嵌入信息，观察结果。

**可交付物**：比较 RoPE 和 NoPE 性能的学习曲线。

> **目的与价值**：NoPE（无位置嵌入）实验揭示了 Transformer 的"位置感知"是否真的需要显式位置信息。现有研究表明，因果掩码本身就包含了隐式的位置信息（mask 的上三角结构让模型能区分"之前的"和"之后的"位置），所以 NoPE 并不完全失败。

### 消融 3：SwiGLU 对比 SiLU

### 问题 (swiglu_ablation)：SwiGLU 对比 SiLU（1 分）（1 个 H100 小时）

比较 SwiGLU 前馈网络与使用 SiLU 激活但没有门控线性单元（GLU）的前馈网络的性能：

$$\text{FFN}_{\text{SiLU}}(x) = W_2 \cdot \text{SiLU}(W_1 x) \tag{25}$$

在 `FFN_SiLU` 实现中，设 $d_{\text{ff}} = 4 \times d_{\text{model}}$，以近似匹配 SwiGLU 前馈网络的参数数量（SwiGLU 有三个而不是两个权重矩阵）。

**可交付物**：比较 SwiGLU 和 SiLU 前馈网络（参数数量近似匹配）性能的学习曲线。

**可交付物**：关于你的发现的几句话讨论。

---

## 7.4 在 OpenWebText 上运行

我们现在转向一个从网络爬取创建的更标准预训练数据集。以下是来自 OpenWebText 的一个示例（注意文本更真实、复杂和多样化）：

> **OpenWebText 示例**：Baseball Prospectus director of technology Harry Pavlidis...（略，见原文）

注意：对于此实验，可能需要重新调整学习率或批次大小等超参数。

### 问题 (main_experiment)：OWT 实验（2 分）（3 个 H100 小时）

使用与 TinyStories 相同的模型架构和总训练迭代次数，在 OpenWebText 上训练你的语言模型。这个模型表现如何？

**可交付物**：你的语言模型在 OpenWebText 上的学习曲线。描述与 TinyStories 的损失差异——我们应该如何解释这些损失？

**可交付物**：来自 OpenWebText LM 的生成文本，格式与 TinyStories 输出相同。这段文本的流畅性如何？为什么即使使用相同的模型和计算预算，输出质量也更差？

> **目的与价值**：TinyStories vs OpenWebText 的对比揭示了数据集难度的重要性。TinyStories 是专门为儿童设计的简单英语故事，词汇量小、句式简单；OpenWebText 是真实网络文本，复杂且多样。同样的模型架构和训练预算，在更难的数据集上会得到更高的验证损失——但这不一定意味着模型"更差"，只是任务更难。生成质量的对比也很有启发性。

---

## 7.5 你自己的修改 + 排行榜

### 排行榜规则

- **运行时间**：你的提交最多可以在 H100 上运行 1.5 小时
- **数据**：只能使用我们提供的 OpenWebText 训练数据集
- 其他方面不受限制

如果在寻找实现思路，可以参考：

- 最先进的开源 LLM 系列，如 Llama 3 [Grattafiori et al., 2024] 或 Qwen 2.5 [Yang et al., 2024]
- NanoGPT speedrun 仓库（社区成员发布了许多用于"快速运行"小规模语言模型预训练的有趣修改，例如共享输入和输出嵌入权重）

### 问题 (leaderboard)：排行榜（6 分）（10 个 H100 小时）

在上述排行榜规则下训练一个模型，目标是在 1.5 个 H100 小时内最小化语言模型的验证损失。

**可交付物**：记录的最终验证损失、对应的学习曲线（清楚显示 x 轴为墙上时钟时间且少于 1.5 小时），以及你所做工作的描述。期望排行榜提交至少超过 5.0 损失的朴素基准线。

---

## 章节总结

第 4-7 章构建了完整的训练与实验管线。核心组件链条是：

```
数据（分词后）→ 数据加载器 → [前向传播 → 损失 → 反向传播 → AdamW + 梯度裁剪 → 学习率调度] → 检查点 → 生成
```

每个组件的设计动机：

| 组件 | 作用 |
|------|------|
| **交叉熵损失** | 测量模型对真实下一词的惊讶程度 |
| **AdamW** | 自适应学习率 + 解耦权重衰减，收敛更快更稳定 |
| **余弦学习率调度** | 预热 + 平滑衰减，避免早期不稳定和后期震荡 |
| **梯度裁剪** | 防止梯度爆炸，提高训练鲁棒性 |
| **内存映射数据加载** | 支持大于 RAM 的数据集 |
| **检查点** | 支持长期训练的中断恢复 |
| **温度/Top-p 采样** | 控制生成多样性与质量的权衡 |
| **消融实验** | 验证每个架构选择的必要性 |
