# GPU 计算方案笔记

> 除了 NVIDIA CUDA，GPU / NPU 上还有哪些通用计算方案？各自属于哪一代技术路线、要什么硬件、代价是什么？

本仓库按**首次公开发布或规范 1.0 定稿时间**排序，把主流与历史方案各建一个文件夹，子目录 README 记录该方案的具体内容。CUDA 也收录一份，作为全表的基准对照。

## 演进主线

1. **2003–2012 借道图形管线**：BrookGPU、Renderscript、C++ AMP、WebCL 把着色器当 kernel 用，不需要独立驱动栈，但受限于图形 API 的表达能力。
2. **2009–2016 开放标准成型**：OpenCL、OpenACC、OpenMP offload、SYCL、Vulkan Compute 追求「一套源码跨厂商」，代价是实现质量参差、性能调优碎片化。
3. **2016–2020 厂商专有栈对标 CUDA**：ROCm/HIP、oneAPI、SynapseAI、CANN 各自复制 CUDA 的「语言 + 驱动 + 算子库」三件套，生态靠厂商投入而非标准背书。
4. **2020–2024 编译器与 DSL 上位**：TVM、Triton、Pallas、Mojo、XLA 让用户只写高层算子，由编译器下沉到各家硬件，硬件锁定问题被推到编译器层。
5. **横向的兼容层路线**：ZLUDA、chipStar 不做移植，而是把 CUDA / HIP 翻译到别的后端，用「吃生态」代替「建生态」。

## 通用 GPU 计算方案一览（按发布时间排序）

| 时间 | 名称（类型） | 主导方 | 开闭源 · 许可 | 核心语言 / 接口 | 需要硬件 | 主要优势 | 现状与局限 | 官方文档 |
|---|---|---|---|---|---|---|---|---|
| 2003 | [BrookGPU](./brookgpu/README.md)（流编程语言） | 斯坦福大学 | 开源（BSD 式） | Brook C（`brcc` 编译器） | 早期 NVIDIA / ATI GPU | GPU 通用计算的开山之作，首次提出 stream kernel 抽象 | ⚠️ 已废弃（2012 前后停止），仅学术价值 | [Stanford BrookGPU（归档）](https://web.archive.org/web/2015/http://graphics.stanford.edu/projects/brookgpu/) |
| 2007 | **[CUDA](./cuda/README.md)**（厂商专有 API，**基准对照**） | NVIDIA | 闭源（SDK 免费，部分组件开源） | C/C++ 扩展、PTX、cuBLAS/cuDNN | 仅 NVIDIA GPU | 生态最完整，文档、算子库、调试工具齐全 | 锁定 NVIDIA，无法跨厂商 | [developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit) |
| 2009 | [OpenCL](./opencl/README.md)（跨厂商开放标准） | Khronos Group | 开放规范（实现多闭源） | OpenCL C / SPIR-V | NVIDIA、AMD、Intel、部分 ARM/FPGA | 几乎所有 GPU 都能跑，工业界长期事实标准 | 样板代码冗长，编译器质量差异大，1.2 之后演进缓慢 | [khronos.org/opencl](https://www.khronos.org/opencl/) |
| 2009 | [DirectCompute](./directcompute/README.md)（图形 API 内计算） | Microsoft | 闭源（DX11 运行时） | HLSL Compute Shader | Windows + DX11 以上 GPU | 零额外驱动，与渲染管线共享资源 | 仅 Windows / Xbox，无 HPC 数学库 | [learn.microsoft.com/directcompute](https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader) |
| 2011 | [OpenACC](./openacc/README.md)（指令制导） | OpenACC 组织（PGI/NVIDIA 主导） | 开放规范（实现闭源为主） | `#pragma acc` | 多厂商加速器（实际以 NVIDIA 为主） | 增量改造成本最低，老 Fortran/C 代码几行即可上 GPU | 编译器支持面窄，跨厂商效果不稳 | [openacc.org](https://www.openacc.org/) |
| 2011 | [Renderscript Compute](./renderscript/README.md)（Android 运行时） | Google | 开源（AOSP） | RenderScript C99 | Android 设备 GPU / DSP | 一次编写自动分派到 CPU/GPU/DSP | ⚠️ 已废弃（Android 12 移除） | [Android 12 移除公告](https://developer.android.com/about/versions/12/behavior-changes-12#renderscript) |
| 2012 | [C++ AMP](./cpp-amp/README.md)（C++ 语言扩展） | Microsoft | 开源规范（MSVC 实现闭源） | C++ + `restrict(amp)` | Windows + DX11 GPU | 纯 C++ 语法，`array_view` 数据抽象优雅 | ⚠️ 已停止演进（2014 后无更新） | [learn.microsoft.com/cpp-amp](https://learn.microsoft.com/en-us/cpp/parallel/amp/cpp-amp-overview) |
| 2013 | [OpenMP target offload](./openmp-offload/README.md)（指令制导） | OpenMP ARB | 开放规范 | `#pragma omp target` | 多厂商 CPU / GPU / FPGA | 与现有 OpenMP 线程模型无缝衔接，GCC/Clang 均支持 | 卸载性能高度依赖具体编译器 | [openmp.org](https://www.openmp.org/) |
| 2014 | [Metal / MPS](./metal-mps/README.md)（Apple 专有） | Apple | 闭源 | Metal Shading Language（MSL）+ MPS | Apple A/M 系列芯片 | Apple 平台唯一低开销方案，统一内存零拷贝 | 仅 Apple 生态 | [developer.apple.com/metal](https://developer.apple.com/metal/) |
| 2014 | [SYCL](./sycl/README.md)（跨厂商单源 C++） | Khronos Group | 开放规范（AdaptiveCpp 开源，DPC++ 开源） | 标准 C++17/20 | Intel、AMD、NVIDIA、FPGA | 单源 C++，无 `#pragma` 无 `__global__`，可与模板/STL 混用 | 运行时（Level Zero/OpenCL/CUDA 后端）质量不均 | [khronos.org/sycl](https://www.khronos.org/sycl/) |
| 2014 | [WebCL](./webcl/README.md)（浏览器 API） | Khronos Group | 开放规范 | JavaScript 绑定 OpenCL C | 浏览器 + 支持 GPU | 曾被视为浏览器端通用计算的答案 | ⚠️ 已停滞（2014 后无浏览器实现跟进） | [khronos.org/webcl](https://www.khronos.org/webcl/) |
| 2016 | [Vulkan Compute](./vulkan-compute/README.md)（图形 API 内计算） | Khronos Group | 开放规范 | GLSL / HLSL / SPIR-V Compute Shader | 桌面与移动 GPU、部分 SoC | 跨 Windows/Linux/Android，驱动开销低，显存控制精细 | 需手工管理管线/内存屏障，学习曲线陡 | [vulkan.org](https://www.vulkan.org/) |
| 2016 | [ROCm + HIP](./rocm-hip/README.md)（厂商开放栈） | AMD | 开源（MIT/LLVM 许可） | HIP C++（CUDA 近亲）+ rocBLAS/MIOpen | AMD GPU（Radeon Instinct/MI 系列） | `hipify` 可把 CUDA 源码近乎直接迁移，源码级跨厂商 | Windows 支持晚且不全，算子库覆盖弱于 CUDA | [rocm.docs.amd.com](https://rocm.docs.amd.com/) |
| 2017 | [Apache TVM](./tvm/README.md)（深度学习编译器） | Apache 基金会 | 开源（Apache-2.0） | Python DSL / TE / TVMScript | NVIDIA、AMD、ARM、FPGA、自定义加速器 | 自动调度搜索，一套 IR 下沉多后端 | 手写调度调优门槛高，近年活跃度下降 | [tvm.apache.org](https://tvm.apache.org/) |
| 2017 | [XLA](./xla/README.md)（编译器 + 加速器） | Google | 开源（Apache-2.0） | HLO IR / JAX / TF 前端 | Google TPU、GPU、CPU | 图级融合优化，TPU 上性能接近手写内核 | 绑定 JAX/TF 生态，独立使用困难 | [openxla.org/xla](https://openxla.org/xla) |
| 2018 | [oneAPI / DPC++](./oneapi/README.md)（厂商开放栈） | Intel | 开源（Apache-2.0 with LLVM 例外） | SYCL + `sycl::` 扩展、Level Zero | Intel GPU / CPU / FPGA（可扩展到他家） | 真正跨架构（CPU+GPU+FPGA）统一编程模型 | 生态与算子在 Intel 之外支持薄弱 | [intel.com/oneapi](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html) |
| 2019 | [Habana SynapseAI](./synapseai/README.md)（AI 加速器栈） | Intel（收购 Habana） | 部分开源（驱动/图编译器） | PyTorch 前端 + 自定义算子 | Habana Gaudi / Goya | 训练吞吐与性价比对标 A100，`torch.compile` 直达 | 仅 Habana 硬件，玩家极少 | [docs.habana.ai](https://docs.habana.ai/) |
| 2020 | [ZLUDA](./zluda/README.md)（CUDA 兼容层） | 社区（原 AMD 资助） | 开源（MIT/Apache-2.0） | 直接运行 CUDA 二进制 / 源码 | AMD GPU、Intel GPU | 无源码也能跑 CUDA 程序，迁移成本几乎为零 | 实验性质，性能损耗明显，新 CUDA 特性跟进慢 | [github.com/vosen/ZLUDA](https://github.com/vosen/ZLUDA) |
| 2020 | [chipStar](./chipstar/README.md)（HIP 上的 OpenCL 兼容层） | 社区（LLVM 生态） | 开源（MIT） | HIP/OpenCL C++ 源码 | 任意支持 OpenCL 或 Level Zero 的设备 | 让 CHIP-SPV 类代码跑在非 AMD 硬件上 | 小众，调试与性能分析工具缺失 | [github.com/CHIP-SPV/chipStar](https://github.com/CHIP-SPV/chipStar) |
| 2021 | [OpenAI Triton](./triton/README.md)（Python DSL + 编译器） | OpenAI（现社区主导） | 开源（MIT） | Python（`@triton.jit`） | NVIDIA、AMD（ROCm 后端） | 用 Python 写接近手写性能的 kernel，比 CUDA C++ 快数倍上手 | 仅覆盖 NVIDIA/AMD，非张量场景表达力有限 | [triton-lang.org](https://triton-lang.org/) |
| 2023 | [WebGPU](./webgpu/README.md)（浏览器标准） | W3C / GPU for the Web WG | 开放规范（wgpu 等实现开源） | WGSL + JS/TS API | 浏览器 + 支持 GPU | 唯一被主流浏览器共同支持的 GPU 计算入口 | 能力保守（无 shared memory 编程模型、无 64 位原子），不适合重 HPC | [w3.org/TR/webgpu](https://www.w3.org/TR/webgpu/) |
| 2023 | [Apple MLX](./mlx/README.md)（数组框架） | Apple | 开源（MIT） | Python / C++（NumPy 风格） | Apple Silicon（统一内存） | 惰性求值 + 统一内存，Mac 上跑 LLM 推理开箱即用 | 仅 Apple Silicon，生态与算子覆盖有限 | [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx/) |
| 2023 | [Mojo](./mojo/README.md)（语言 + 编译器） | Modular | 部分开源（标准库开源，编译器闭源） | Mojo（Python 超集语法） | NVIDIA、AMD、Apple（逐步扩展） | 兼顾 Python 易用性与 C++ 级性能，MLIR 后端 | 生态尚早，语言与工具链仍在快速变动 | [docs.modular.com/mojo](https://docs.modular.com/mojo/) |
| 2024 | [JAX Pallas](./pallas/README.md)（内核 DSL） | Google（JAX 团队） | 开源（Apache-2.0） | Python（`pl.pallas_call`） | TPU、NVIDIA GPU | 一套 kernel 写法跨 TPU/GPU，可精细控制片上内存 | 后端成熟度不一，调试困难 | [jax.readthedocs.io/pallas](https://jax.readthedocs.io/en/latest/pallas/index.html) |

## 国产 / AI 专用芯片专有栈

这些栈的时间线跨度大、生态相对独立，单独列出，不参与主表排序。

| 时间 | 名称（类型） | 主导方 | 开闭源 · 许可 | 核心语言 / 接口 | 需要硬件 | 主要优势 | 现状与局限 | 官方文档 |
|---|---|---|---|---|---|---|---|---|
| 2019 | [CANN / AscendCL](./cann-ascendcl/README.md) | 华为 | 部分开源（驱动闭源） | AscendCL C/C++、MindSpore 前端 | 昇腾 310/910 系列 | 国产 AI 芯片中生态最完整，算子库与图引擎齐全 | 与 CUDA 差异大，迁移需重写算子 | [hiascend.com/cann](https://www.hiascend.com/cann) |
| 2019 | [Iluvatar CoreX](./iluvatar/README.md) | 天数智芯 | 闭源 SDK | CUDA 风格 API、兼容 PyTorch | 天垓 / 智铠系列 | 接口刻意贴近 CUDA，迁移成本低 | 算子覆盖率与版本跟进依赖厂商 | [iluvatar.com](https://www.iluvatar.com) |
| 2019 | [BANG / MagicMind](./cambricon-bang/README.md) | 寒武纪 | 部分开源（BANG C 语言族） | BANG C、MagicMind 图编译器 | MLU 系列 | 自研指令集，推理场景能效比高 | 编程模型独特，第三方生态薄 | [cambricon.com](https://www.cambricon.com/) |
| 2021 | [HG-DTK](./hg-dtk/README.md) | 海光 | 闭源（ROCm 衍生） | HIP / ROCm 兼容接口 | 海光 DCU | 直接复用 ROCm 生态，CUDA 代码经 HIP 可迁 | 绑定海光硬件，版本滞后于上游 ROCm | [hygon.cn](https://www.hygon.cn/) |
| 2021 | [TopsRider](./topsrider/README.md) | 燧原科技 | 闭源 SDK | 图编译 + 自定义算子 | 邃思系列 | 面向训练集群优化，支持分布式 | 社区资料少，主要靠厂商支持 | [enflame-tech.com](https://www.enflame-tech.com/) |
| 2022 | [MUSA](./musa/README.md) | 摩尔线程 | 部分开源（2024 起逐步开放） | MUSA C++（CUDA 风格）+ muBLAS | 摩尔线程 GPU | 接口高度对标 CUDA，`musify` 可转换源码 | 生态新，工具链成熟度待验证 | [mthreads.com](https://www.mthreads.com/) |
| 2022 | [BIRENSUPA](./birensupa/README.md) | 壁仞科技 | 闭源 SDK | BIRENSUPA C/C++、图编译器 | BR100/BR104 | 单卡算力高，面向大模型训练 | 软件栈较新，算子与框架适配在补齐中 | [birentech.com](https://www.birentech.com/) |
| 2024 | [MXMACA](./mxmaca/README.md) | 沐曦 | 闭源 SDK | MACA C++（CUDA 风格） | 曦云 C 系列 | 兼容 CUDA 编程习惯，迁移路径清晰 | 生态起步阶段，社区积累少 | [metax-tech.com](https://www.metax-tech.com/) |

## 按硬件平台速查

| 硬件平台 | 首选方案 | 可用方案 | 不要指望 |
|---|---|---|---|
| NVIDIA GPU | CUDA | OpenCL、OpenACC、OpenMP offload、SYCL、Triton、TVM、Vulkan Compute、DirectCompute | — |
| AMD GPU | ROCm + HIP | OpenCL、Vulkan Compute、SYCL（AdaptiveCpp）、Triton（ROCm 后端）、OpenMP、ZLUDA | CUDA 原生（需兼容层） |
| Intel GPU | oneAPI / DPC++ | OpenCL、SYCL、Vulkan Compute、OpenMP、Level Zero | CUDA 原生 |
| Apple Silicon | Metal / MPS、MLX | WebGPU（经 wgpu）、OpenCL（已弃用但仍存在） | CUDA、高效 OpenCL |
| 国产 GPGPU | 厂商专有栈 | 部分兼容 HIP / CUDA 风格 API | 通用 SYCL 支持 |
| 浏览器 | WebGPU | — | WebCL（已死） |

## 选型建议

1. **只有 NVIDIA 卡**：直接用 CUDA，不必为了「可移植」绕路。可移植性在单一硬件上没有收益。
2. **要从 CUDA 迁到 AMD**：优先 `hipify` 做源码级迁移到 HIP，而不是重写成 OpenCL。有大量 kernel 时评估工作量，别低估调试成本。
3. **Intel GPU + CPU 混合负载**：选 oneAPI / DPC++，这是唯一把 CPU、GPU、FPGA 放进同一编程模型的成熟栈。
4. **必须一套代码跨 NVIDIA / AMD / Intel GPU**：选 SYCL（oneAPI 或 AdaptiveCpp）。OpenCL 能跑但开发效率低，新项目不建议。
5. **写自定义算子（attention、量化 kernel 等）**：先试 Triton，上手成本远低于 CUDA C++；Triton 表达不了的再回到 CUDA / HIP。
6. **维护老 Fortran / C 科学计算代码**：OpenACC 或 OpenMP `target` 增量改造，改代码行数最少。
7. **Apple 平台本地推理**：用 MLX 或 MPS，不要用 OpenCL（Apple 已停止维护）。
8. **只有 CUDA 二进制、没有源码**：ZLUDA 可作为实验性方案，务必先做正确性与性能验证。
9. **已废弃方案（BrookGPU、Renderscript、C++ AMP、WebCL）**：不要用，只在读历史代码或做技术演进分析时参考。

## 说明

- 时间口径统一为「**首次公开发布**或**规范 1.0 定稿**」，两者都有的取较早者，并在对应子目录 README 的基本信息表里注明。
- 「官方文档」列为该方案的一手资料来源。已下线或已归档的方案，链接指向归档页或官方移除公告，并在单元格内注明。
- 每个方案的具体内容见其文件夹内的 `README.md`，公共章节模板为：基本信息 → 设计思路 → 最小示例 → 与其它方案的异同 → 优势 → 局限 → 现状与生态 → 参考。
- 已废弃方案的子 README 用「**为什么它消失了**」替代「现状与生态」。
