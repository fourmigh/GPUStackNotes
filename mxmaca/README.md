# 沐曦 MXMACA

> 沐曦 GPU 的软件栈，提供 CUDA 风格的 MACA 编程接口，2024 年起随曦云 C 系列推向市场。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 沐曦（MetaX） |
| 首次公开 | 2024 发布 MXMACA 软件栈（配合曦云 C500） |
| 类型 | 通用 GPU 软件栈 |
| 开闭源 · 许可 | 闭源 SDK |
| 核心语言 / 接口 | MACA C/C++（CUDA 风格）、mcBLAS / mcDNN 等库、PyTorch 适配 |
| 需要硬件 | 曦云 C 系列（C500 等） |
| 当前状态 | 起步阶段，商用推广中 |

## 设计思路

MXMACA 延续了国产 GPU 的主流策略：**API 命名与 CUDA 一一对应**（`macaMalloc`、`macaLaunchKernel`），并提供源码转换工具，把 `.cu` 文件的 `cuda*` 前缀替换为 `maca*`。

沐曦强调「CUDA 生态兼容」，目标客户是已有 CUDA 代码库、需要国产化替代的训练与推理用户。软件栈还包含多卡互联（MetaXLink）与集群管理组件。

## 最小示例

```cuda
#include <maca_runtime.h>          // 对应 cuda_runtime.h

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    macaMalloc(&dx, n * sizeof(float));
    macaMemcpy(dx, hx, n * sizeof(float), macaMemcpyHostToDevice);
    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, dx, dy);
    macaDeviceSynchronize();
}
```

## 与其它方案的异同

- **vs CUDA**：源码风格一致但需重编译与转换，非二进制兼容。
- **vs MUSA / BIRENSUPA**：同类竞品（国产通用 GPU + CUDA 风格 API），差别在硬件规模与生态投入。
- **vs HIP**：HIP 开源、跨 AMD/海光；MXMACA 闭源、只服务沐曦硬件。
- **vs CANN**：CANN 是专用架构、推理为主；MXMACA 面向通用 GPU，训练推理兼顾。

## 优势

- 迁移路径清晰，CUDA 代码转换成本低。
- 曦云 C500 面向大模型训练，提供多卡互联方案。
- 国产化选项，供应链相对自主。
- 团队多来自 AMD/NVIDIA，对 GPU 软件栈理解较深。

## 局限

- 生态刚起步，算子库与框架适配覆盖不全。
- 闭源，问题排查依赖厂商。
- 社区积累几乎为零，开发者自助能力弱。
- 性能与稳定性缺少公开的第三方评测。
- 软件版本迭代快，兼容性可能反复。

## 现状与生态

2024 年起在国内智算中心与运营商项目中推广，属于「新生代国产 GPU」阵营。由于生态尚早，实际项目通常由沐曦工程师驻场支持。选型建议先做小规模 PoC，重点验证目标模型与训练的稳定性。

## 参考

- 官方网站：<https://www.metax-tech.com/>
- MXMACA SDK 用户手册
