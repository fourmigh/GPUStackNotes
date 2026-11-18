# Iluvatar CoreX

> 天数智芯的通用 GPU 软件栈，接口刻意贴近 CUDA，目标是让现有 CUDA 程序尽量少改就能跑。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 天数智芯（Iluvatar CoreX） |
| 首次公开 | 2019 发布天垓 100 训练卡，2021 量产 |
| 类型 | 通用 GPU 软件栈 |
| 开闭源 · 许可 | 闭源 SDK（免费提供） |
| 核心语言 / 接口 | CUDA 风格 C/C++ 接口、兼容 PyTorch / TensorFlow 的框架适配层 |
| 需要硬件 | 天垓（训练）/ 智铠（推理）系列 |
| 当前状态 | 商用中，生态在补齐 |

## 设计思路

天数智芯的路线非常明确：**做「接口兼容」的通用 GPU**。软件栈提供与 CUDA Runtime 高度相似的 API，框架侧提供 PyTorch 扩展，让用户把 `.cu` 代码少量修改即可编译运行。

这与 HIP 的思路一致，区别是 HIP 有开源生态与 AMD 背书，天数则完全自研硬件加自研软件栈。

## 最小示例

```cuda
// 接口与 CUDA 高度一致，通常只需替换头文件与少量 API 名
#include <cuda_runtime.h>          // 天数提供的兼容头文件

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

推理侧更常见的是直接装天数版的 PyTorch wheel：

```python
import torch
model = model.to("cuda")           # 对上层透明
y = model(x)
```

## 与其它方案的异同

- **vs CUDA**：API 兼容但非二进制兼容，仍需重新编译；库生态与调优工具差距明显。
- **vs ROCm / HIP**：目标相同（承接 CUDA 迁移），HIP 开源且生态大，CoreX 闭源但针对国产硬件。
- **vs MUSA / MXMACA**：国产厂商普遍采用「CUDA 风格 API」策略，差异主要在硬件性能与算子覆盖。
- **vs CANN**：CANN 面向专用 AI 加速器（达芬奇架构），CoreX 面向通用 GPU（可编程性更强）。

## 优势

- 迁移成本相对低，CUDA 程序员几乎零学习成本。
- 通用 GPU 架构，兼顾训练与推理，也能跑非 AI 的并行计算。
- 国产供应链，适合有自主可控要求的场景。
- 提供 PyTorch / TensorFlow 适配，主流模型可直接跑。

## 局限

- 算子覆盖率不如 CUDA，冷门算子可能缺失或性能差。
- 框架适配版本滞后于上游 PyTorch。
- 闭源，问题定位依赖厂商支持。
- 性能与 H100 级别仍有差距，大规模训练场景需谨慎评估。
- 社区资料少，公开文档有限。

## 现状与生态

主要在国产替代与信创项目中落地，推理场景多于训练。对已有 CUDA 代码库的团队，是迁移成本较低的选项之一。选型时应要求厂商提供目标模型的实测数据，不要只看峰值算力。

## 参考

- 官方网站：<https://www.iluvatar.com>
- Iluvatar CoreX SDK 用户手册
