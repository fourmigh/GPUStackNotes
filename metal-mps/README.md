# Metal / Metal Performance Shaders

> 2014 年 Apple 自研的底层图形与计算 API，是 Apple Silicon 上唯一能同时兼顾低开销与高性能的 GPU 通路。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Apple |
| 首次公开 | 2014-06（Metal，iOS 8）；MPS 框架 2015（iOS 9） |
| 类型 | 厂商专有图形 + 计算 API |
| 开闭源 · 许可 | 闭源 |
| 核心语言 / 接口 | Metal Shading Language（基于 C++14）、Metal API、MPS / MPSGraph |
| 需要硬件 | Apple A7 及以后的 SoC、M 系列芯片 |
| 当前状态 | 活跃，Metal 3 / 4 持续更新 |

## 设计思路

Metal 的设计目标是「去掉 OpenGL 的历史包袱」：显式管理命令缓冲区、资源状态、同步，避免驱动在运行时猜测。计算侧通过 `MTLComputeCommandEncoder` 提交 compute kernel，线程组织用「线程组 × 组内线程」，与 CUDA 的块/线程同构。

真正的杀手锏是 **统一内存架构（UMA）**：CPU 和 GPU 共享物理内存，`MTLBuffer` 用 `storageModeShared` 就能零拷贝访问。MPS 则在此之上提供卷积、矩阵乘、归一化等预置 kernel。

## 最小示例

```metal
// kernel.metal
#include <metal_stdlib>
using namespace metal;

kernel void saxpy(device float *y [[buffer(0)]],
                  const device float *x [[buffer(1)]],
                  constant float &a [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    y[id] = a * x[id] + y[id];
}
```

```swift
let enc = cmdBuf.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setBuffer(bufY, offset: 0, index: 0)
enc.setBuffer(bufX, offset: 0, index: 1)
enc.setBytes(&a, length: 4, index: 2)
enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
enc.endEncoding()
```

Swift 里可直接拿到 `MTLBuffer.contents()` 的指针，UMA 下不需要任何拷贝。

## 与其它方案的异同

- **vs CUDA**：线程组织与内存模型高度类似，但没有 warp shuffle 之外的细粒度原语，且只支持 Apple 硬件。
- **vs OpenCL**：Metal 是 Apple 为取代 OpenCL 而做的；OpenCL 在 macOS 上已停止更新，功能停在 1.2。
- **vs Vulkan**：定位对等（显式低开销），但 Vulkan 跨平台、Metal 只服务 Apple。
- **vs MLX**：MLX 是 Metal 之上的数组/自动微分框架，面向 ML；Metal/MPS 是更底层的通用计算接口。

## 优势

- Apple 平台唯一低开销方案，驱动开销远低于 OpenGL/OpenCL。
- 统一内存零拷贝，对 LLM 推理（权重常驻内存）特别友好。
- MPS / MPSGraph 提供高质量预置算子，PyTorch 的 MPS 后端即基于此。
- 与 Xcode GPU 调试器、Metal System Trace 深度集成。

## 局限

- 完全锁定 Apple 生态，无法用于服务器或跨平台项目。
- 文档与社区规模远小于 CUDA，遇到问题资料少。
- LLM 训练场景缺少成熟的多卡通信库（无 NCCL 对等物）。
- MSL 与 C++ 有差异，移植 CUDA 代码需手工改写。

## 现状与生态

Mac 上 PyTorch（MPS 后端）、llama.cpp（Metal 后端）、MLX、Core ML 都以 Metal 为底座。Apple Silicon 的统一内存让它在「本地跑大模型」场景具有独特优势，但训练与多卡场景基本不参与竞争。

## 参考

- 官方文档（Metal）：<https://developer.apple.com/metal/>
- 官方文档（MPS）：<https://developer.apple.com/documentation/metalperformanceshaders>
- *Metal by Tutorials*
