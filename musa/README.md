# 摩尔线程 MUSA

> 摩尔线程的通用 GPU 软件栈，MUSA 编程接口高度对标 CUDA，并提供 `musify` 做源码转换。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 摩尔线程（Moore Threads） |
| 首次公开 | 2022 发布 MUSA 架构与 MTT S80 显卡 |
| 类型 | 通用 GPU 软件栈 |
| 开闭源 · 许可 | 部分开源（2024 起逐步开放 MUSA SDK 与驱动组件） |
| 核心语言 / 接口 | MUSA C/C++（CUDA 风格）、muBLAS / muDNN / muFFT、musify 转换工具 |
| 需要硬件 | 摩尔线程 GPU（MTT S 系列、数据中心卡） |
| 当前状态 | 活跃，MUSA 4.0（2024），持续开源 |

## 设计思路

MUSA 是国产厂商里「**最直接对标 CUDA**」的一套：API 命名从 `cuda*` 换成 `musa*`，kernel 语法保留 `__global__` / `<<<>>>`，数学库对应 cuBLAS → muBLAS、cuDNN → muDNN。

`musify` 工具可以自动把 `.cu` 源码转成 `.mu`，把 `cudaMalloc` 替换为 `musaMalloc`。对已有 CUDA 项目的迁移成本是国产方案里最低的一档。

## 最小示例

```cuda
#include <musa_runtime.h>          // 对应 cuda_runtime.h

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    musaMalloc(&dx, n * sizeof(float));
    musaMemcpy(dx, hx, n * sizeof(float), musaMemcpyHostToDevice);
    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, dx, dy);
    musaDeviceSynchronize();
}
```

迁移命令（示意）：`musify saxpy.cu -o saxpy.mu`

## 与其它方案的异同

- **vs CUDA**：语法与 API 几乎同构，但非二进制兼容，必须重新编译；性能和库覆盖仍有差距。
- **vs HIP**：HIP 开源且有 AMD 与海光生态；MUSA 闭源但更贴近中国供应链。
- **vs ZLUDA**：ZLUDA 跑未修改的 CUDA 二进制，MUSA 要求源码级转换，后者可控性更好。
- **vs CANN**：CANN 是专用架构，MUSA 是通用 GPU，可编程性与迁移路径更接近 CUDA。

## 优势

- 迁移成本极低，CUDA 程序员学习曲线近乎为零。
- 通用 GPU 架构，图形与计算兼顾（摩尔线程本身做显卡）。
- 2024 年起逐步开源 SDK，透明度提升。
- 消费级显卡价格有竞争力，适合本地推理与教学。

## 局限

- 性能与 NVIDIA 同代产品仍有明显差距，大模型训练场景受限。
- 库覆盖不全，部分 cuDNN 算子需要替代实现。
- 框架适配（PyTorch）版本跟进需要时间。
- 开源刚起步，社区积累薄。
- 驱动稳定性与软件成熟度仍需时间验证。

## 现状与生态

国内少数同时做消费级显卡与数据中心卡的厂商，MUSA 是其主要技术路线。适合「CUDA 代码迁移 + 国产化」的中小规模推理场景。训练与大规模集群能力仍需实测验证。

## 参考

- 官方网站：<https://www.mthreads.com/>
- MUSA SDK 文档：<https://github.com/MooreThreads>
