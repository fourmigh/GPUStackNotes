# XLA

> Google 的机器学习编译器，把 JAX / TensorFlow 的计算图编译成针对 TPU、GPU、CPU 优化的机器码。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Google |
| 首次公开 | TPU 2016 年公布；**XLA 2017 年开源**（TensorFlow 生态） |
| 类型 | 图编译器 + 运行时 |
| 开闭源 · 许可 | 开源（Apache-2.0），TPU 后端闭源 |
| 核心语言 / 接口 | HLO IR、StableHLO；前端为 JAX / TensorFlow |
| 需要硬件 | Google TPU、NVIDIA GPU、CPU |
| 当前状态 | 活跃，是 JAX 的默认编译路径 |

## 设计思路

XLA 把「立即执行」的算子逐个调用，改成「先整体编译成一张图」。它在 HLO（High Level Optimizer）IR 上做算子融合、常量折叠、布局优化、内存规划，再下沉到 LLVM 或 TPU 后端。

对 GPU 而言，XLA 的价值是**自动融合**：`x * 2 + 1` 这种链式操作不会产生多个 kernel 启动，而是融成一个。对 TPU 而言，XLA 是唯一入口，矩阵乘的调度由编译器与硬件协同决定。

## 最小示例

```python
import jax, jax.numpy as jnp

@jax.jit                      # 触发 XLA 编译
def selu(x, alpha=1.67, scale=1.05):
    return scale * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

print(selu(jnp.arange(5.0)))
# jax.make_jaxpr(selu)(x) 可查看 HLO 前后的 IR
```

同一份代码在 TPU 与 GPU 上都能跑，差异由 XLA 后端吸收。

## 与其它方案的异同

- **vs CUDA**：XLA 让你不写 kernel，靠编译器生成；CUDA 给你全部控制权。XLA 生成代码的性能在规整算子上接近手写，在奇异算子上会差。
- **vs TVM**：XLA 绑定 JAX/TF 前端，HLO 是图级 IR；TVM 前端开放、覆盖更多后端与端侧场景。
- **vs Triton**：Triton 面向手写 kernel，XLA 面向整图编译，两者常在同一栈里共存（PyTorch Inductor 同时用 Triton 和可选的 XLA 后端）。
- **vs Pallas**：Pallas 是 JAX 内的手写 kernel 层，用来补 XLA 表达不了的算子，二者是同一生态的上下层。

## 优势

- 算子融合效果好，减少 kernel 启动与显存往返。
- TPU 上性能接近手写，是使用 TPU 的唯一现实路径。
- 自动内存规划，大模型训练可显著降低显存占用。
- 与 JAX 的函数式变换（`jit` / `grad` / `vmap` / `pmap`）配合紧密。

## 局限

- 编译时间长，动态 shape 会触发重编译。
- 调试困难：出错信息常指向 HLO 或生成代码。
- TPU 后端闭源，无法自行扩展。
- 独立于 JAX/TF 使用几乎不可行。
- GPU 后端的性能通常不如 CUDA 手写或 Triton。

## 现状与生态

JAX 已成为 Google 内部与大模型研究的重要框架，XLA 是其性能基础。PyTorch 也有 XLA 后端（torch-xla），主要用于 TPU 集群。在纯 NVIDIA GPU 场景，XLA 的竞争力不如 CUDA 生态与 Triton。

## 参考

- 官方文档：<https://openxla.org/xla>
- *XLA: Optimizing Compiler for Machine Learning*（Google 博客）
