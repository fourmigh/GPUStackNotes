# Apache TVM

> 2017 年开源的深度学习编译器，用一套中间表示把模型编译到从服务器 GPU 到单片机的各种后端。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Apache 基金会（最初来自华盛顿大学 SAMPL 组） |
| 首次公开 | 2017-08 开源，2019 进入 Apache 孵化器 |
| 类型 | 深度学习编译器 + 运行时 |
| 开闭源 · 许可 | 开源（Apache-2.0） |
| 核心语言 / 接口 | Python（TE / TVMScript / Relay）、C++ 运行时 |
| 需要硬件 | NVIDIA、AMD、ARM、Intel、FPGA、自定义加速器（LLVM / CUDA / OpenCL / Metal 后端） |
| 当前状态 | 维护中，但社区活跃度较 2021 年高峰明显下降 |

## 设计思路

TVM 的核心是「**计算与调度分离**」：用 Tensor Expression（TE）只描述「算什么」，再用 schedule 描述「在哪算、怎么分块、怎么向量化」。这使得同一份算子定义可以针对不同硬件生成完全不同的代码。

两代 IR：Relay（图级，负责算子融合、常量折叠）和 TensorIR（TIR，循环级，负责 schedule 变换）。TVM 还提供 AutoTVM / Ansor 做自动调度搜索，用机器学习找最优分块参数。

## 最小示例

```python
import tvm
from tvm import te

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda i: A[i] * 2.0, name="B")

# 调度：把循环切成「块 × 块内线程」
s = te.create_schedule(B.op)
bx, tx = s[B].split(B.op.axis[0], factor=256)
s[B].bind(bx, te.thread_axis("blockIdx.x"))
s[B].bind(tx, te.thread_axis("threadIdx.x"))

f = tvm.build(s, [A, B], target="cuda")   # 或 "llvm"、"opencl"、"metal"
```

## 与其它方案的异同

- **vs CUDA**：TVM 是生成器，CUDA 是手写。TVM 适合算子数量多、形态规整的场景；极端定制算子仍要手写。
- **vs Triton**：Triton 面向手写 kernel 的 DSL，自动处理线程映射；TVM 更偏「编译器自动生成 + 自动调优」，覆盖后端更广（含 CPU/单片机）。
- **vs XLA**：XLA 服务 JAX/TF 生态，绑定 Google 技术栈；TVM 后端与前端都更开放。
- **vs oneAPI / SYCL**：TVM 是编译器层面的抽象，SYCL 是编程模型层面的抽象。

## 优势

- 后端覆盖面极广，一份模型可部署到 GPU、ARM CPU、RISC-V、FPGA。
- 计算/调度分离的设计对算子自动生成非常有效。
- Ansor 自动调优在部分算子上能超过手工实现。
- 与 MLC-LLM 等项目结合，可在手机、浏览器端跑大模型。

## 局限

- 学习曲线陡，写 schedule 比写 CUDA 还难。
- 自动调优耗时长（一个算子搜索数小时很常见），需要大量算力。
- 社区活跃度下降，部分模块维护不足。
- 对动态 shape、控制流复杂的模型支持仍弱于 PyTorch 原生。

## 现状与生态

在端侧推理（手机、IoT、浏览器）与自定义加速器编译（国产 NPU 的编译器常基于 TVM）中仍有重要地位。MLC-LLM 项目让它重新进入大模型部署视野。但通用 GPU 训练场景基本被 PyTorch + Triton 占据。

## 参考

- 官方文档：<https://tvm.apache.org/>
- Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI 2018
