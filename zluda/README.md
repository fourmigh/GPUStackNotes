# ZLUDA

> 2020 年出现的 CUDA 兼容层，让未经修改的 CUDA 程序直接跑在 AMD（以及后来的 Intel）GPU 上。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 社区项目（作者 Andrzej Janik），曾获 AMD 资助 |
| 首次公开 | 2020-02 首次发布 |
| 类型 | CUDA 二进制 / API 兼容层（翻译层） |
| 开闭源 · 许可 | 开源（MIT / Apache-2.0） |
| 核心语言 / 接口 | 直接拦截 CUDA Runtime / Driver API，转发到 HIP 或 Level Zero |
| 需要硬件 | AMD GPU（ROCm）、Intel GPU（Level Zero 后端） |
| 当前状态 | 实验性，可运行部分真实应用，不建议生产使用 |

## 设计思路

ZLUDA 不去改源码，而是在**二进制层拦截**：用自己实现的 `libcuda.so` / `nvcuda.dll` 顶替原版，把 `cuLaunchKernel`、`cuMemAlloc` 等调用翻译成 HIP 或 Level Zero 调用。PTX 也需要被翻译成目标 GPU 的 ISA，这一步通过修改 LLVM 的 NVPTX 后端实现。

它解决的是一个很现实的问题：**很多 CUDA 程序没有源码**（商业软件、旧二进制），或者重编译成本极高。

## 最小示例

```bash
# 把 ZLUDA 生成的 nvcuda.dll 放到程序目录，或设置库路径
export LD_LIBRARY_PATH=/path/to/zluda:$LD_LIBRARY_PATH

# 然后直接运行原本的 CUDA 程序，无需重编译
./my_cuda_app
```

也可以用它运行 PyTorch 的 CUDA wheel（不换 ROCm 版本），但需要匹配的 PyTorch 版本，且性能与稳定性均无保证。

## 与其它方案的异同

- **vs HIP / hipify**：HIP 是「改源码、重编译」，可控、性能好；ZLUDA 是「不改任何东西、直接跑」，但性能损耗大。
- **vs chipStar**：chipStar 把 HIP/OpenCL 源码编译到别的后端；ZLUDA 针对的是已经编译好的 CUDA 二进制。
- **vs 国产 GPU 的 CUDA 兼容层**（MUSA musify 等）：思路相同，都是「吃 CUDA 生态」，但各家实现深度不同。
- **vs 原生 CUDA**：ZLUDA 始终是「追赶者」，CUDA 每出一个新特性都要重新适配。

## 优势

- 对无源码的 CUDA 程序是唯一可选项。
- 项目本身证明了 CUDA API 层的可替代性，有研究价值。
- 同一套机制可扩展到 Intel GPU，具备跨厂商潜力。
- 开源，可针对特定应用打补丁。

## 局限

- 性能损耗明显（常见 20%–60%），且不稳定。
- 只覆盖 CUDA API 的一个子集，新版本 CUDA 特性跟进滞后。
- 依赖特定驱动/ROCm 版本，环境极其脆弱。
- 部分库（cuDNN、TensorRT、NCCL）需要单独适配，实际可用性打折。
- 法律与授权状态模糊，AMD 曾在 2021 年短暂停止资助。

## 现状与生态

被用于在 AMD 显卡上跑一些老 CUDA 工具和游戏 PhysX 场景，也有实验性跑 PyTorch 的尝试。2024 年前后 Intel 版本重新活跃。整体定位是「技术验证 + 应急方案」，不能作为长期架构选择。

## 参考

- 官方仓库：<https://github.com/vosen/ZLUDA>
- 作者关于 PTX → AMDGCN 翻译的技术说明
