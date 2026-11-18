# 海光 HG-DTK

> 海光 DCU 的软件栈，基于 ROCm 衍生，因此 HIP 代码与大部分 CUDA 迁移经验可以直接复用。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 海光信息（Hygon） |
| 首次公开 | DCU 产品 2019 年起；DTK 软件栈 2021 前后成体系 |
| 类型 | 通用 GPU（DCU）软件栈 |
| 开闭源 · 许可 | 闭源（基于开源 ROCm 的衍生发行版） |
| 核心语言 / 接口 | HIP / ROCm 兼容接口、rocBLAS 等库、PyTorch 适配 |
| 需要硬件 | 海光 DCU（深算系列） |
| 当前状态 | 商用中，国内 HPC 与 AI 场景有实际部署 |

## 设计思路

HG-DTK（DCU Toolkit）的定位是「**ROCm 兼容发行版**」。它复用了 ROCm 的编译器（基于 LLVM/clang 的 hipcc）、HIP 运行时、数学库与性能工具，针对海光 DCU 的硬件做后端适配。

这意味着：AMD 生态里能跑的东西，理论上改少量配置就能在海光上跑；CUDA 代码经 hipify 转 HIP 后也有较大概率可直接使用。

## 最小示例

```cpp
// 与 HIP 完全一致
#include <hip/hip_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    hipMalloc(&dx, n * sizeof(float));
    hipLaunchKernelGGL(saxpy, dim3((n + 255) / 256), dim3(256), 0, 0, n, 2.0f, dx, dy);
    hipDeviceSynchronize();
}
```

编译：`hipcc saxpy.hip -o saxpy`（使用 DTK 提供的工具链）。

## 与其它方案的异同

- **vs ROCm / HIP**：上游与衍生的关系；DTK 版本通常滞后于 ROCm 主线，但接口一致度很高。
- **vs CUDA**：不直接兼容，需经 hipify 走 HIP 路径。
- **vs CANN**：CANN 是专用架构自研栈，DTK 是通用 GPU + 开源栈衍生，迁移成本低得多。
- **vs 摩尔线程 MUSA / 沐曦 MXMACA**：它们自建 CUDA 风格 API；海光直接借用 HIP，生态起点更好。

## 优势

- 复用 ROCm 生态，HIP 代码、PyTorch ROCm 版经验可直接迁移。
- 通用 GPU 架构，HPC 与 AI 都能跑，适合科学计算 + AI 混合负载。
- 国产化方案中「迁移成本最低」的选项之一。
- x86 生态兼容性好，与海光 CPU 组合可做全国产整机。

## 局限

- 版本滞后于上游 ROCm，新特性跟进慢。
- 闭源发行版，遇到问题定位依赖厂商。
- 算子与框架适配需要海光版本，不能直接用 AMD 官方 wheel。
- 性能与 NVIDIA 数据中心卡仍有差距。
- 社区资料少，主要在 HPC 中心内部流转。

## 现状与生态

国内超算与智算中心的重要国产选项，尤其适合已有 HIP/CUDA 代码库的迁移项目。选型时建议用 ROCm 版本对齐的方式验证：先在 AMD 上跑通，再平移到 DCU。

## 参考

- 官方网站：<https://www.hygon.cn/>
- HG-DTK 用户手册（海光 DCU 开发者文档）
