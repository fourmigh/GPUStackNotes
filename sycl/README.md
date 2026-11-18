# SYCL

> 2014 年 Khronos 发布的单源 C++ 异构编程标准，被广泛视为「C++ 版 OpenCL」，也是 C++ AMP 思路的正统继承者。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Khronos Group（Codeplay 最初推动） |
| 首次公开 | 2014-03 公布 SYCL 1.2 草案；**SYCL 2020 于 2021-02 正式发布** |
| 类型 | 跨厂商单源 C++ 编程模型 |
| 开闭源 · 许可 | 开放规范；实现开源（AdaptiveCpp、DPC++）或闭源（ComputeCpp 已停） |
| 核心语言 / 接口 | C++17/20，`sycl::queue`、`handler::parallel_for`、USM、缓冲区/访问器 |
| 需要硬件 | Intel、AMD、NVIDIA GPU、CPU、FPGA（取决于后端） |
| 当前状态 | 活跃，SYCL 2020 是当前主线 |

## 设计思路

SYCL 解决 OpenCL 的两个痛点：**样板代码**和**类型安全**。它用 C++ 模板把 kernel 写成 lambda，主机代码与设备代码在同一文件、同一编译单元，无需单独的 `.cl` 文件与字符串拼接。

数据模型有两套：传统的 `buffer` + `accessor`（编译器分析依赖、自动插入同步）和 SYCL 2020 引入的 USM（Unified Shared Memory，手动指针 + `malloc_shared`，更接近 CUDA）。执行模型则用 `nd_range` 表达工作组层级。

## 最小示例

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
    queue q;                                   // 自动选设备，可指定 gpu_selector_v
    float *x = malloc_shared<float>(n, q);     // USM：主机设备共享
    float *y = malloc_shared<float>(n, q);

    q.parallel_for(nd_range<1>{n, 256}, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        y[i] = a * x[i] + y[i];                // 设备侧代码，普通 C++
    }).wait();

    free(x, q);
}
```

编译：`icpx -fsycl`（Intel DPC++）或 `acpp`（AdaptiveCpp），后者可直接生成 CUDA/HIP 后端。

## 与其它方案的异同

- **vs OpenCL**：SYCL 是 OpenCL 的 C++ 现代化重写，多数实现以 OpenCL 或 Level Zero 为后端。
- **vs CUDA / HIP**：SYCL 单源、跨厂商，但缺少 cuDNN 级别的算子库；CUDA 生态仍是断层领先。
- **vs C++ AMP**：概念几乎同源，SYCL 是真正多实现、跨平台的版本。
- **vs oneAPI**：oneAPI 是 Intel 的发行版生态，SYCL 是其核心语言；SYCL 本身不绑定厂商。

## 优势

- 单源 C++，可用模板、STL、`constexpr`，无 `#pragma`、无 `__global__`。
- 一份代码覆盖 NVIDIA（经 CUDA 后端）、AMD（经 ROCm 后端）、Intel（经 Level Zero），AdaptiveCpp 已做到生产可用。
- 规范由 Khronos 维护，多家实现互相校验。
- 与 OpenCL 生态兼容，存量 kernel 可逐步迁移。

## 局限

- 后端质量差异大：同一份代码在 Intel GPU 上顺滑，在 NVIDIA 上可能因 CUDA 后端转换而损失性能。
- 编译时间长，模板错误信息极长。
- 深度学习算子库缺失，AI 场景需自己接 oneDNN / cuDNN。
- SYCL 2020 之前的代码与 2020 语义差异较大，老教程易误导。

## 现状与生态

Intel 是最大推动者（DPC++ / oneAPI），AdaptiveCpp（原 hipSYCL）是跨厂商路线的代表，已支持 CUDA、HIP、Level Zero 多后端。科研与工业界用于「一份代码跑多种加速器」的中间层。GROMACS、部分 HPC 应用已有 SYCL 移植。

## 参考

- 官方文档：<https://www.khronos.org/sycl/>
- Khronos SYCL 2020 Specification
- AdaptiveCpp 文档、Intel oneAPI DPC++ 文档
