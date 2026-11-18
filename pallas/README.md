# JAX Pallas

> JAX 生态中的内核 DSL，让同一份手写 kernel 既能跑 TPU 的 MXU，也能跑 NVIDIA GPU。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Google（JAX 团队） |
| 首次公开 | 2023（TPU 后端）；GPU 后端 2024 逐步成熟 |
| 类型 | 内核编写 DSL（嵌入 Python） |
| 开闭源 · 许可 | 开源（Apache-2.0） |
| 核心语言 / 接口 | Python（`pl.pallas_call`、`pl.load` / `pl.store`）、Mosaic（TPU）/ Triton（GPU）后端 |
| 需要硬件 | Google TPU、NVIDIA GPU |
| 当前状态 | 活跃，仍在快速演进 |

## 设计思路

XLA 能自动融合大部分算子，但总有一些需要精细控制片上内存与流水线的场景（FlashAttention、MoE 路由、自定义量化）。Pallas 就是补这个缺口：用 Python 写 kernel，显式声明共享内存/scratch 布局，编译器再把它下沉到 TPU 的 Mosaic 或 GPU 的 Triton。

它比 Triton 的抽象更低一层（可以手动指定 pipeline 与 grid），又不像 CUDA 那样绑定单一硬件。

## 最小示例

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]     # 块内运算

def add(x, y, block=1024):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.size // block,),
        in_specs=[pl.BlockSpec((block,), lambda i: (i,))] * 2,
        out_specs=pl.BlockSpec((block,), lambda i: (i,)),
    )(x, y)
```

`BlockSpec` 描述「第 i 个 program 取哪一块数据」，这是 Pallas 控制数据流的核心机制。

## 与其它方案的异同

- **vs Triton**：抽象层级接近，Triton 只服务 GPU 且更成熟；Pallas 跨 TPU/GPU，但两端成熟度差异大。
- **vs XLA**：XLA 自动编译整图，Pallas 手写 kernel 嵌入 JAX 计算图，二者互补。
- **vs CUDA**：Pallas 不接触线程级细节，无法做 warp 级优化，但换来了硬件可移植。
- **vs TVM**：TVM 是独立编译器栈，Pallas 是 JAX 生态内的轻量 DSL。

## 优势

- 一套 kernel 代码覆盖 TPU 与 GPU，对同时使用两者的团队价值大。
- 显式控制片上内存与流水线，能实现高性能 attention 类算子。
- 与 JAX 的 `jit` / `grad` / `shard_map` 无缝组合。
- 比 CUDA 简单，比 XLA 的自动融合可控。

## 局限

- TPU 与 GPU 后端行为差异大，写出两端都最优的 kernel 很难。
- 调试困难，错误信息常指向 Mosaic/Triton 生成的代码。
- 生态极小，几乎只有 Google 内部与 JAX 社区使用。
- GPU 后端依赖 Triton，受 Triton 能力限制。
- 文档与示例偏少，学习曲线陡。

## 现状与生态

Google 内部用于 TPU 上的高性能算子（如 Pallas 版 FlashAttention、MoE），JAX 社区也在跟进。对绝大多数用户来说 XLA 已足够，Pallas 只在对性能有极致要求且需要跨 TPU/GPU 时才值得投入。

## 参考

- 官方文档：<https://jax.readthedocs.io/en/latest/pallas/index.html>
- Pallas 设计说明：<https://github.com/jax-ml/jax>
