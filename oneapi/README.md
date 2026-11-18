# oneAPI / DPC++

> Intel 于 2018 年提出的跨架构编程计划，核心是 DPC++（基于 SYCL）加 Level Zero 运行时，目标是 CPU、GPU、FPGA 一套代码。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Intel（oneAPI 为开放行业倡议） |
| 首次公开 | 2018-12 宣布；2020-12 oneAPI 1.0 Gold |
| 类型 | 跨架构编程模型 + 工具链 + 库集合 |
| 开闭源 · 许可 | 开源（Apache-2.0 with LLVM exceptions） |
| 核心语言 / 接口 | DPC++（SYCL + 扩展）、Level Zero、oneMKL / oneDNN / oneCCL |
| 需要硬件 | Intel GPU（Arc / Max / Data Center GPU）、CPU、FPGA；DPC++ 可编译到他家 |
| 当前状态 | 活跃，持续随 Intel GPU 产品线更新 |

## 设计思路

oneAPI 是「**一个编程模型，多种架构**」的工程化落地。技术上分三层：

1. **DPC++**：SYCL 的超集，加了 USM、子组（sub-group）原语、设备端数学库等扩展。
2. **Level Zero**：贴近硬件的底层 API，类似 CUDA Driver API，供需要精细控制的场景。
3. **oneXXX 库**：oneMKL（数学）、oneDNN（深度学习）、oneCCL（通信）、oneTBB（线程），对标 MKL/cuDNN/NCCL/TBB。

## 最小示例

```cpp
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
    queue q(gpu_selector_v);                       // 选 Intel GPU
    buffer<float, 1> x{n}, y{n};
    q.submit([&](handler &h) {
        accessor a{x, h, read_only};
        accessor b{y, h, write_only};
        h.parallel_for(n, [=](id<1> i) {
            b[i] = a[i] * 2.0f;                    // 一套代码可跑 CPU/GPU/FPGA
        });
    }).wait();
}
```

编译：`icpx -fsycl prog.cpp`；运行时可加 `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` 指定后端。

## 与其它方案的异同

- **vs SYCL**：DPC++ 是 SYCL 的一个实现加扩展，oneAPI 则是围绕它的生态。纯 SYCL 代码通常可直接用 `icpx -fsycl` 编译。
- **vs CUDA**：oneAPI 追求跨架构，CUDA 追求单架构极致；Intel GPU 上没有 cuDNN 对等成熟度的库。
- **vs ROCm / HIP**：HIP 是「复制 CUDA 接口」，oneAPI 是「用标准 C++ 重写接口」。
- **vs OpenCL**：oneAPI 是 Intel 推动的 OpenCL 继任者，Level Zero 取代了 OpenCL 的驱动层。

## 优势

- 真正跨架构：同一份代码在 Intel CPU 与 GPU 上都能跑，CPU 侧性能尤其好（oneMKL/oneDNN 优化成熟）。
- 开源程度高，基于 LLVM，可自行编译与调试。
- Intel GPU（尤其 Max 系列）显存大、性价比高，适合推理。
- 工具链完整：VTune、Advisor、GDB 都有 GPU 支持。

## 局限

- 生态主要围绕 Intel 硬件，在 NVIDIA/AMD 上的支持属「能用」级别。
- DPC++ 扩展非标准 SYCL，代码可移植性打折。
- 编译器对模板错误信息极长，调试体验差。
- Intel GPU 的软件成熟度仍在爬坡，部分算子缺失。

## 现状与生态

Intel 数据中心 GPU（Max 1550/1750）与 Arc 消费卡的主力编程模型。PyTorch 有 Intel GPU 扩展（IPEX），llama.cpp 有 SYCL 后端。在 HPC 领域（尤其是 Aurora 超算）是核心栈。纯 NVIDIA 环境下通常不作为首选。

## 参考

- 官方文档：<https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html>
- oneAPI DPC++ Compiler 文档：<https://github.com/intel/llvm>
