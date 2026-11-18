# OpenAI Triton

> 2021 年 OpenAI 开源的 Python GPU 编程语言与编译器，用几十行 Python 写出接近手写 CUDA 性能的 kernel。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | OpenAI（现由社区与 PyTorch 团队共同维护） |
| 首次公开 | 2021-07 开源 |
| 类型 | 领域特定语言（DSL）+ 编译器 |
| 开闭源 · 许可 | 开源（MIT） |
| 核心语言 / 接口 | Python（`@triton.jit`）、Triton IR、LLVM 后端 |
| 需要硬件 | NVIDIA GPU（主要）、AMD GPU（ROCm 后端） |
| 当前状态 | 活跃，PyTorch 2.0 的 Inductor 后端默认使用它 |

## 设计思路

Triton 的核心抽象是「**块（block）级编程**」：程序员操作的是 `tl.load` 出来的张量块，而不是单个线程。编译器负责把块级操作映射到 warp、共享内存和向量化指令，并自动做内存合并与流水线调度。

这跳过了 CUDA 里最痛苦的部分——手算线程索引、手工布置 shared memory、处理 bank conflict。代价是失去对 warp 级细节的完全控制。

## 最小示例

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                      # 第几个块
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 块内偏移向量
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

x, y = torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")
out = torch.empty_like(x)
add_kernel[(triton.cdiv(1024, 1024),)](x, y, out, 1024, BLOCK=1024)
```

注意没有任何线程索引，只有块索引和块内向量。

## 与其它方案的异同

- **vs CUDA**：Triton 把线程调度交给编译器，开发效率高一个量级；但 warp 级 trick（如手工 `shfl`、非规则共享内存复用）表达不了。
- **vs TVM**：TVM 侧重编译器自动生成 + 自动调优，Triton 侧重人来写块级 kernel，编译器补细节。
- **vs Pallas**：Pallas 是 JAX 生态的同类思路，跨 TPU/GPU；Triton 更偏 NVIDIA GPU 且更成熟。
- **vs Mojo**：Mojo 是完整语言，Triton 是 Python 内嵌 DSL，上手成本更低。

## 优势

- 上手极快：会 PyTorch 就能写 kernel，几十行实现 FlashAttention 级别的融合。
- 性能通常在 CUDA 手写的 80%–100%，部分场景反超。
- 自动处理内存合并、共享内存分配、流水线并行。
- PyTorch 2.0 `torch.compile` 的默认后端，生态位极佳。
- 社区算子库丰富（triton kernels、flash-attention、vLLM 的自定义算子）。

## 局限

- 只支持 NVIDIA 与 AMD，无 Intel / Apple / 国产 GPU 后端。
- 抽象层级较高，极端优化（warp specialization、异步拷贝精细控制）受限。
- 调试困难，错误信息指向 IR 而非 Python 代码。
- 编译缓存与版本兼容问题较多，生产部署需注意。
- 非张量类算法（图计算、稀疏不规则访问）表达力弱。

## 现状与生态

已成为「写自定义深度学习算子」的事实标准：PyTorch Inductor、vLLM、SGLang、Unsloth、FlashAttention 均大量使用。大量 CUDA 手写 kernel 正在被 Triton 重写。2023 年后 AMD ROCm 后端逐步可用。

## 参考

- 官方文档：<https://triton-lang.org/>
- 官方仓库：<https://github.com/triton-lang/triton>
- Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MAPL 2019
