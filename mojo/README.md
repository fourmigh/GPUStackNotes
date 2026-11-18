# Mojo

> 2023 年 Modular 发布的编程语言，语法接近 Python，性能目标对齐 C++/CUDA，定位是「AI 基础设施的统一语言」。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Modular（Chris Lattner 创立） |
| 首次公开 | 2023-05 发布；2024-03 标准库部分开源 |
| 类型 | 编程语言 + 编译器（基于 MLIR） |
| 开闭源 · 许可 | 部分开源（标准库与部分组件），编译器核心闭源 |
| 核心语言 / 接口 | Mojo（Python 超集语法）、`fn` / `struct` / `parallelize`、MLIR 后端 |
| 需要硬件 | NVIDIA GPU（MAX 平台）、AMD、Apple（逐步扩展） |
| 当前状态 | 活跃但早期，语言与工具链仍在快速变动 |

## 设计思路

Mojo 试图解决 Python 生态的核心矛盾：**好写但慢，快就要换语言**。它保留 Python 的语法与缩进，但引入静态类型（`fn`）、值语义（`struct`）、所有权与生命周期，让编译器能生成接近 C++ 的机器码。

它明确支持 GPU：`parallelize` 与 `vectorize` 等原语可把循环映射到 GPU 线程，MAX 平台（原 Modular 推理引擎）用 Mojo 重写了算子，据称在部分模型上比 CUDA 实现更快。

## 最小示例

```mojo
from algorithm import vectorize
from sys.info import simdwidthof

fn saxpy(a: Float32, x: DTypePointer[DType.float32],
         y: DTypePointer[DType.float32], n: Int):
    @parameter
    fn impl[i: Int]():
        y[i] = a * x[i] + y[i]

    vectorize[impl, simdwidthof[DType.float32]()](n)
```

GPU 路径需结合 MAX 的 `DeviceContext` 与 kernel 启动 API，语法仍在演进。

## 与其它方案的异同

- **vs Python + Triton**：Triton 是 Python 内嵌 DSL，只在 kernel 层提速；Mojo 是完整语言，整条链路都能编译。
- **vs CUDA**：Mojo 用高级语法表达并行，试图同时拿下易用性与性能；CUDA 生态与工具成熟度远胜。
- **vs Rust**：都强调性能与内存安全，Mojo 选择 Python 语法与 MLIR 后端，面向 AI 而非系统编程。
- **vs Julia**：Julia 也走「科学计算高性能」路线，但 GPU 支持靠包；Mojo 把 GPU 写进语言核心。

## 优势

- Python 语法零学习成本，可与 Python 代码互操作。
- 基于 MLIR，能同时生成 CPU、GPU 与专用加速器代码。
- 静态类型 + 所有权，性能可预测，无解释器开销。
- MAX 平台展示了在 LLM 推理上超越 CUDA 实现的潜力。
- Chris Lattner（LLVM、Swift、MLIR 作者）背书，技术路线清晰。

## 局限

- 生态几乎为零：没有成熟的第三方库、包管理器仍在早期。
- 编译器核心闭源，社区无法深度参与。
- GPU 编程接口尚未稳定，版本间 API 变更频繁。
- 与 Python 的互操作仍有边界（GIL、对象模型差异）。
- 「Python 超集」目前只覆盖子集，很多 Python 库无法直接跑。

## 现状与生态

MAX 平台（推理引擎）是主要落地产物，支持 Llama、Mistral 等模型。2024 年起逐步开源标准库并吸引社区贡献。作为「语言级重构 AI 基础设施」的尝试，长期潜力受关注，但短期内不具备替代 PyTorch/CUDA 的条件。

## 参考

- 官方文档：<https://docs.modular.com/mojo/>
- *Mojo: A New Programming Language for AI*（Modular 博客）
