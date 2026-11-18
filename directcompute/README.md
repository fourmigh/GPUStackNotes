# DirectCompute

> 2009 年随 DirectX 11 发布的 GPU 计算接口，把 compute shader 正式纳入图形 API。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Microsoft |
| 首次公开 | 2009-10（随 Windows 7 / DirectX 11 发布） |
| 类型 | 图形 API 内的计算通路 |
| 开闭源 · 许可 | 闭源，随 Windows SDK / DirectX 运行时提供 |
| 核心语言 / 接口 | HLSL Compute Shader（CS 4.0 / 5.0）+ D3D11 API |
| 需要硬件 | Windows Vista+ 与 DX11 及以上 GPU（NVIDIA、AMD、Intel 均可） |
| 当前状态 | 已被 D3D12 的 Compute / DirectML 取代，D3D11 仍广泛在用 |

## 设计思路

DirectCompute 不新建编程模型，而是给已有的 HLSL 加一种 shader 阶段 `cs_5_0`。它复用 D3D 的资源绑定（SRV / UAV）、纹理和缓冲区，让游戏在渲染同一帧里顺便做后处理、粒子模拟、光照探针计算。

核心价值是**零额外依赖**：不装 CUDA 运行时、不装 OpenCL 驱动，游戏或 Windows 应用直接就能用。

## 最小示例

```hlsl
// 每个线程处理一个元素
[numthreads(64, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    float4 v = inputBuffer[id.x];
    outputBuffer[id.x] = v * 2.0;
}
```

```cpp
// Host 侧：绑定 UAV 后按线程组派发
ctx->CSSetShader(csShader, nullptr, 0);
ctx->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
ctx->Dispatch(256, 1, 1);        // 256 组 × 64 线程 = 16384 个元素
```

## 与其它方案的异同

- **vs OpenCL**：DirectCompute 绑定 D3D 资源和图形管线，适合「图形 + 计算」混合负载；OpenCL 是独立计算栈，适合纯计算。
- **vs Vulkan Compute**：定位几乎对等，差别在平台（D3D 仅 Windows/Xbox，Vulkan 跨平台）和显存控制粒度（Vulkan 更底层）。
- **vs CUDA**：无 warp 级原语、无 HPC 数学库，也不支持跨 GPU 大规模通信，不适合 HPC。
- **vs DirectML**：DirectML 是 D3D12 之上的机器学习算子库，属于 DirectCompute 思路在 AI 场景的延续。

## 优势

- 无需额外驱动，Windows 上部署成本最低。
- 与渲染管线共享纹理/缓冲区，图形计算混合场景效率高。
- HLSL 工具链成熟，PIX 可直接调试 compute shader。

## 局限

- 仅 Windows 与 Xbox，完全不可移植。
- CS 5.0 限制多（无 64 位原子、线程组大小受限、无跨组同步）。
- 没有线性代数、FFT 等库，纯计算场景要自己造轮子。
- D3D11 的驱动开销高于 D3D12 与 CUDA。

## 现状与生态

游戏引擎（Unity、Unreal）和 Windows 图像/视频处理软件里仍大量使用。通用计算与 AI 场景已被 CUDA、DirectML、Vulkan Compute 分流。微软没有为它建立独立生态，它始终是「图形 API 的附加能力」。

## 参考

- 官方文档：<https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader>
- *Practical Rendering and Computation with Direct3D 11*
