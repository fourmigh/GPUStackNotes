# ROCm + HIP

> AMD 的开源 GPU 计算平台。ROCm 是对标 CUDA 的整体栈，HIP 是其中「让 CUDA 代码能编译到 AMD 上」的那一层。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | AMD |
| 首次公开 | ROCm 2016-04；HIP 2017 发布，2018 随 ROCm 开源 |
| 类型 | 厂商开放栈（驱动 + 编译器 + 库 + 迁移工具） |
| 开闭源 · 许可 | 开源（MIT / LLVM / Apache-2.0），部分固件与库闭源 |
| 核心语言 / 接口 | HIP C++（`__global__`、`hipLaunchKernelGGL`）、rocBLAS、MIOpen、rocFFT、RCCL |
| 需要硬件 | AMD GPU（MI 系列数据中心卡支持最好） |
| 当前状态 | 活跃，ROCm 6.x；2024 起有 HIP SDK for Windows |

## 设计思路

AMD 走的是「**兼容而非取代**」的路线。HIP 的 API 与 CUDA Runtime 几乎一一对应（`cudaMalloc` → `hipMalloc`），`hipify-perl` / `hipify-clang` 可以自动把 CUDA 源码转成 HIP 源码。

ROCm 则补齐 CUDA 有而 AMD 缺的部分：LLVM 化编译器（hipcc 基于 clang）、数学库、通信库、性能分析工具（rocprof、Omniperf）。目标很明确——让 CUDA 用户的迁移成本降到「改头文件 + 重编译」。

## 最小示例

```cpp
#include <hip/hip_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    hipMalloc(&dx, n * sizeof(float));
    hipMemcpy(dx, hx, n * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(saxpy, dim3((n + 255) / 256), dim3(256), 0, 0, n, 2.0f, dx, dy);
    hipDeviceSynchronize();
}
```

迁移命令：`hipify-perl saxpy.cu > saxpy.hip`，通常无需改动就能编译。

## 与其它方案的异同

- **vs CUDA**：源码级几乎同构，但库生态（尤其深度学习算子）和 Windows 支持明显弱；CUDA 15 年积累的调优经验无法直接复制。
- **vs OpenCL**：HIP 是显式线程模型、性能可控；OpenCL 更抽象、跨厂商。AMD 自身也在把重心从 OpenCL 转向 HIP。
- **vs SYCL（AdaptiveCpp）**：AdaptiveCpp 可在 HIP 后端上跑 SYCL，HIP 是更底层的选择。
- **vs ZLUDA**：HIP 需要源码，ZLUDA 直接跑 CUDA 二进制，后者是「无源码」场景的补丁。
- **vs 海光 HG-DTK**：HG-DTK 是 ROCm 的国产衍生版，接口兼容但版本滞后。

## 优势

- 开源，编译器与运行时行为可查可改。
- `hipify` 让存量 CUDA 代码迁移成本极低，是最现实的「去 NVIDIA 化」路径。
- 单一源码可同时编译到 AMD 和 NVIDIA（HIP 支持 CUDA 后端），便于维护双栈。
- MI300 系列在显存容量与性价比上对大模型推理有吸引力。
- 2024 年 HIP SDK for Windows 补上了长期缺失的平台。

## 局限

- 库覆盖弱于 CUDA：cuDNN 级别的优化算子、TensorRT 级别的推理引擎，AMD 对位产品成熟度不足。
- 版本碎片化严重，PyTorch 官方 wheel 需要匹配特定 ROCm 版本。
- 消费级 Radeon 卡的官方支持长期落后于 Instinct。
- 文档与社区规模明显小于 CUDA。

## 现状与生态

PyTorch 官方提供 ROCm 版本，vLLM / llama.cpp / SGLang 均有 ROCm 后端。数据中心市场以 MI300X 为核心争夺大模型推理。国内海光 DCU 直接复用 ROCm 生态，使它在国产替代方案中地位特殊。

## 参考

- 官方文档：<https://rocm.docs.amd.com/>
- HIP Programming Guide：<https://rocm.docs.amd.com/projects/HIP/en/latest/>
