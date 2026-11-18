# Apple MLX

> 2023 年 Apple 开源的数组计算框架，为 Apple Silicon 的统一内存架构量身设计，是 Mac 上跑大模型最顺手的方案。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Apple Machine Learning Research |
| 首次公开 | 2023-12-12 开源 |
| 类型 | 数组计算框架 + 自动微分（对标 NumPy / PyTorch 的子集） |
| 开闭源 · 许可 | 开源（MIT） |
| 核心语言 / 接口 | Python（NumPy 风格 API）、C++ 核心、MLX 数组内核 |
| 需要硬件 | Apple Silicon（M1 及以后），统一内存 |
| 当前状态 | 活跃，迭代速度快 |

## 设计思路

MLX 的关键假设是：**内存统一，就不该有「拷贝到设备」这一步**。它用惰性求值构建计算图，数组在 CPU 与 GPU 之间共享同一块物理内存，算子按需在合适的单元上执行。

第二个特色是**可组合的函数变换**：`mlx.grad`、`mlx.vmap`、`mlx.compile` 与 JAX 类似，但 API 更贴近 PyTorch/NumPy，迁移成本低。

## 最小示例

```python
import mlx.core as mx

a = mx.random.normal((1024, 1024))
b = mx.random.normal((1024, 1024))

c = a @ b                    # 惰性构图，此时不计算
mx.eval(c)                   # 显式触发执行（GPU 上）

# 自动微分
def loss_fn(w):
    return mx.mean((a @ w) ** 2)
grad_fn = mx.grad(loss_fn)
g = grad_fn(mx.ones((1024, 1024)))
```

大模型场景下 `mlx_lm` 提供 LoRA 微调与推理脚本，一条命令即可在 Mac 上跑 Llama / Qwen。

## 与其它方案的异同

- **vs PyTorch MPS**：MPS 后端是把 CUDA 语义硬搬到 Metal，部分算子缺失；MLX 是原生为统一内存设计，融合与调度更自然。
- **vs Metal / MPS**：MLX 是上层框架，底层仍走 Metal；普通用户不必直接写 MSL。
- **vs NumPy**：API 相近，但 MLX 支持 GPU 惰性求值与自动微分。
- **vs JAX**：变换语义相似，但 MLX 绑定 Apple 硬件、生态更聚焦本地推理。

## 优势

- 统一内存下零拷贝，Mac 上跑 70B 量化模型不需要显存焦虑（用系统内存）。
- 惰性求值 + 算子融合，减少 Metal kernel 启动次数。
- API 贴近 NumPy/PyTorch，迁移心智负担小。
- `mlx_lm` / `mlx_examples` 提供完整的大模型推理与微调链路。
- 开源且更新快，Apple 官方维护。

## 局限

- 只支持 Apple Silicon，无法用于 Linux 服务器或 NVIDIA 集群。
- 算子覆盖不及 PyTorch，部分自定义算子需要写 Metal kernel。
- 多机多卡训练能力缺失（无 NCCL 对等物）。
- 生态年轻，第三方库少，文档仍在完善。
- 训练大模型受限于单机内存带宽与容量。

## 现状与生态

Mac 本地推理与个人微调的主流选择，社区已产出大量量化模型（mlx-community）。与 llama.cpp 的 Metal 后端形成互补：前者偏 Python 生态与训练，后者偏 C++ 推理与跨平台。企业级训练场景不适用。

## 参考

- 官方文档：<https://ml-explore.github.io/mlx/>
- 官方仓库：<https://github.com/ml-explore/mlx>
- Apple ML Research: *MLX* 发布说明
