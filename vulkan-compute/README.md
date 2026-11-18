# Vulkan Compute

> 2016 年 Khronos 发布的跨平台低开销图形 API，其 compute shader 通路是除 CUDA/OpenCL 之外最通用的 GPU 计算入口。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Khronos Group（AMD Mantle 捐赠为起点） |
| 首次公开 | 2016-02（Vulkan 1.0） |
| 类型 | 低开销图形与计算 API |
| 开闭源 · 许可 | 开放规范；驱动多为闭源（Mesa 的 RADV/ANV 开源） |
| 核心语言 / 接口 | GLSL / HLSL（经 glslang、DXC 编译为 SPIR-V）、Vulkan API |
| 需要硬件 | 桌面与移动 GPU、部分 SoC、部分国产 GPU |
| 当前状态 | 活跃，Vulkan 1.3 / 1.4 |

## 设计思路

Vulkan 把「驱动替你做决定」改成「你自己做决定」：显存分配、内存屏障、管线状态、命令缓冲区提交全部显式。计算侧以 compute pipeline + descriptor set 组织资源，用 `vkCmdDispatch` 派发工作组。

代价是代码量巨大（一个最小计算程序数百行），收益是可预测的性能和跨 Windows/Linux/Android 的一致性。

## 最小示例

```glsl
// shader.comp
#version 450
layout(local_size_x = 64) in;
layout(binding = 0) buffer Buf { float data[]; } buf;

void main() {
    uint i = gl_GlobalInvocationID.x;
    buf.data[i] = buf.data[i] * 2.0;
}
```

```cpp
// 关键调用（省略管线/描述符创建）
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
vkCmdDispatch(cmd, (n + 63) / 64, 1, 1);   // 按工作组数派发
```

## 与其它方案的异同

- **vs OpenCL**：Vulkan 更底层、显存控制更精细，且天然与图形管线共享资源；OpenCL 抽象更高、更偏纯计算。
- **vs CUDA**：Vulkan 无 warp 级原语、无 HPC 数学库、无多卡通信，做深度学习需自己补一大层。
- **vs DirectCompute**：定位对等，Vulkan 跨平台、D3D 限 Windows。
- **vs WebGPU**：WebGPU 是 Vulkan/Metal/D3D12 之上的浏览器抽象层，能力是其子集。

## 优势

- 真正跨平台：Windows、Linux、Android、部分嵌入式。
- 驱动开销低，CPU 侧提交成本远小于 OpenGL。
- 显存类型、内存别名、时间线信号量等控制力接近裸金属。
- 与渲染共用资源，图形 + 计算混合场景效率高。
- 国产 GPU（摩尔线程、部分 ARM SoC）普遍优先支持 Vulkan。

## 局限

- 样板代码极多，一个 saxpy 需要数百行初始化代码。
- 内存屏障与管线布局错误会导致静默的错误结果，调试成本高。
- 没有线性代数库，深度学习生态基本为零。
- 各驱动实现质量不一，移动端尤为明显。

## 现状与生态

游戏引擎（Unreal、Unity）、浏览器（Chrome 的 WebGPU 后端）、机器学习推理框架（llama.cpp Vulkan 后端、ncnn、MNN）都用它做跨平台 GPU 加速。NVIDIA 显卡上性能通常低于 CUDA，但可接受；在国产 GPU 上往往是唯一的通用计算通路。

## 参考

- 官方文档：<https://www.vulkan.org/>
- Vulkan Specification 1.3（Chapter: Compute Pipelines）
- *Vulkan Programming Guide*（Sellers & Kessenich）
