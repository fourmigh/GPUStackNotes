# chipStar

> 社区项目，把 HIP / OpenCL C++ 代码编译到任何支持 OpenCL 或 Level Zero 的设备上，让「HIP 源码」不再被 AMD 绑定。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 社区（原 CHIP-SPV 项目、LLVM 生态贡献者） |
| 首次公开 | 2020 年前后（CHIP-SPV 更名 chipStar） |
| 类型 | 源码级兼容层 + 编译器 |
| 开闭源 · 许可 | 开源（MIT） |
| 核心语言 / 接口 | HIP 源码、OpenCL C++；运行时映射到 OpenCL / Level Zero |
| 需要硬件 | 任意支持 OpenCL 2.0+ 或 Intel Level Zero 的设备 |
| 当前状态 | 小众活跃，功能覆盖持续完善 |

## 设计思路

chipStar 的逻辑是：既然 HIP 与 CUDA 语法同构，而 OpenCL 又几乎无处不在，那么可以**把 HIP 当作可移植的源语言**，编译时生成 OpenCL C 或 SPIR-V，运行时再通过 OpenCL / Level Zero 执行。

它提供自己的 `hip_runtime.h` 实现（映射到 OpenCL API），并用 Clang 插件把内核代码转成 SPIR-V。这样一份 HIP 源码可以跑在 Intel GPU、AMD GPU、甚至某些 ARM GPU 上。

## 最小示例

```bash
# 用 chipStar 的 clang 包装器编译 HIP 源码
chipstar-syclcc saxpy.hip -o saxpy_cl

# 运行时选择设备（OpenCL 或 Level Zero）
CHIP_BE=opencl ./saxpy_cl
CHIP_BE=level0 ./saxpy_cl
```

源码本身与 HIP / CUDA 完全一致：

```cpp
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

## 与其它方案的异同

- **vs HIP / ROCm**：HIP 官方运行时只服务 AMD；chipStar 提供「HIP 语法的第三方实现」，可跑在非 AMD 上。
- **vs ZLUDA**：ZLUDA 翻译已编译的 CUDA 二进制，chipStar 编译 HIP 源码，后者可优化、可调试。
- **vs SYCL**：SYCL 是标准化路线，chipStar 是「复用 HIP 语法 + OpenCL 后端」的工程捷径。
- **vs OpenCL 原生**：chipStar 让你用 CUDA 风格写代码，避免 OpenCL 的样板与带宽管理负担。

## 优势

- 一份 HIP 源码可跑在几乎所有 GPU 上，包括 Intel Arc 与国产 OpenCL 设备。
- 源码级编译，性能分析工具可介入（不像 ZLUDA 只能黑盒）。
- 与 LLVM 生态结合，编译链路透明。
- 对已有 HIP 代码库是「零重写」的跨厂商方案。

## 局限

- 功能覆盖不完整，动态并行、部分原子操作、复杂纹理语义支持有限。
- 依赖底层 OpenCL 实现质量，性能波动大。
- 社区规模小，文档与示例少。
- 调试体验差：错误常出现在生成的 SPIR-V 层。
- 高性能算子仍需针对具体设备手写。

## 现状与生态

主要被 Intel GPU 用户用来跑原本面向 AMD 的 HIP 代码，也被用于在国产 OpenCL 设备上运行开源科学计算程序。属于「技术上有意思、工程上小众」的项目，不建议作为生产主栈，但作为跨厂商验证手段很有价值。

## 参考

- 官方仓库：<https://github.com/CHIP-SPV/chipStar>
- CHIP-SPV 相关论文与技术报告
