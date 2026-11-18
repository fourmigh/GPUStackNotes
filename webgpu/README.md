# WebGPU

> 2023 年落地的 W3C 浏览器 GPU 标准，是 WebCL 失败之后「网页跑 GPU 计算」唯一被主流浏览器共同支持的方案。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | W3C GPU for the Web Working Group（Apple、Google、Mozilla、Microsoft） |
| 首次公开 | 2023-04 Chrome 113 首发；**2024-10 成为 W3C Recommendation** |
| 类型 | 浏览器 / 跨平台 GPU API 标准 |
| 开闭源 · 许可 | 开放规范；实现开源（wgpu、Dawn） |
| 核心语言 / 接口 | WGSL 着色语言 + JavaScript/TypeScript API；Rust 侧为 wgpu |
| 需要硬件 | 浏览器 + 支持 Vulkan / Metal / D3D12 的 GPU |
| 当前状态 | 活跃，浏览器与原生（wgpu）双线推进 |

## 设计思路

WebGPU 吸取了 WebGL 与 WebCL 的教训：**抽象层要足够低以发挥性能，又要足够高以保证跨平台与安全**。它把 Vulkan / Metal / D3D12 的共同子集提取出来，用显式管线、绑定组、命令编码器组织，但去掉了内存屏障、同步原语等易错部分。

计算侧用 `dispatchWorkgroups` 提交 compute pipeline，WGSL 里用 `@workgroup_size` 声明线程组，模型与 CUDA 类似但更保守。

## 最小示例

```wgsl
// shader.wgsl
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = data[id.x] * 2.0;
}
```

```javascript
const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(n / 64));    // 按工作组数派发
pass.end();
device.queue.submit([encoder.finish()]);
```

同一份 WGSL 在原生侧（Rust `wgpu`）也能直接使用。

## 与其它方案的异同

- **vs WebCL**：WebGPU 是重新设计而非绑定桌面 API，浏览器厂商全程参与，因此真正落地。
- **vs Vulkan Compute**：WebGPU 是 Vulkan 子集，牺牲部分底层控制换取跨平台一致性与安全沙箱。
- **vs CUDA**：能力保守得多，没有 warp 级原语、无 64 位原子（部分实现）、无共享内存的完整控制。
- **vs WebAssembly SIMD**：Wasm 跑 CPU 并行，WebGPU 跑 GPU，两者是互补关系。

## 优势

- 唯一跨浏览器、跨操作系统、跨 GPU 厂商的网页 GPU 计算入口。
- 安全模型明确，资源生命周期由浏览器管理。
- Rust 的 `wgpu` 让同一套代码可用在原生桌面与服务端。
- 学习曲线比 Vulkan 平缓，比 WebGL 的 GPGPU hack 规范。
- 已被 Chrome、Edge、Safari、Firefox 支持。

## 局限

- 能力是 Vulkan 的保守子集，重 HPC、大模型训练无法承载。
- 无共享内存编程模型（只有 workgroup 内有限变量），无法实现复杂 kernel。
- 浏览器对可用显存、执行时间有限制，长任务需拆分。
- 部分特性（如 subgroup 操作、64 位原子）通过扩展提供，兼容性不一。
- WASM + WebGPU 的胶水层仍有性能开销。

## 现状与生态

浏览器端机器学习（Transformers.js、ONNX Runtime Web、WebLLM）是主要用户，也让「本地浏览器跑小模型」成为现实。原生侧 wgpu 被用于 Rust 图形与计算应用，以及 llama.cpp 的 WebGPU 后端。定位是「广覆盖」而非「高性能」。

## 参考

- 规范正文：<https://www.w3.org/TR/webgpu/>
- wgpu 文档：<https://wgpu.rs/>
