# WebCL

> 2014 年 Khronos 发布的浏览器端 OpenCL 绑定，曾被视为「网页也能跑 GPU 通用计算」的答案。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Khronos Group（Nokia、Samsung 等推动） |
| 首次公开 | 2014-03 发布 WebCL 1.0 规范 |
| 类型 | 浏览器 JavaScript API（OpenCL 绑定） |
| 开闭源 · 许可 | 开放规范 |
| 核心语言 / 接口 | JavaScript API + OpenCL C kernel 字符串 |
| 需要硬件 | 浏览器 + 支持 OpenCL 的 GPU |
| 当前状态 | ⚠️ 已停滞（2014 年后无主流浏览器实现） |

## 设计思路

WebCL 直接照搬 OpenCL 的对象模型到 JavaScript：`WebCLPlatform`、`WebCLDevice`、`WebCLContext`、`WebCLKernel`，kernel 仍用 OpenCL C 写，以字符串形式传入编译。

设计上没有任何创新，纯粹是「让 OpenCL 在浏览器里可用」。这与 WebGL（把 OpenGL ES 暴露给 JS）是同一套思路。

## 最小示例

```javascript
const platforms = webcl.getPlatforms();
const device = platforms[0].getDevices()[0];
const ctx = webcl.createContext(device);

// kernel 以字符串形式编译
const prog = ctx.createProgram(`
  __kernel void add(__global const float* a,
                    __global const float* b,
                    __global float* c) {
      int i = get_global_id(0);
      c[i] = a[i] + b[i];
  }
`);
prog.build([device]);

const kernel = prog.createKernel('add');
kernel.setArg(0, bufA);
kernel.setArg(1, bufB);
kernel.setArg(2, bufC);
queue.enqueueNDRangeKernel(kernel, 2, null, [1024], [64]);
```

## 与其它方案的异同

- **vs WebGPU**：WebGPU 是 W3C 主导、所有主流浏览器已实现的继任者，用 WGSL 而非 OpenCL C，且包含图形能力。
- **vs OpenCL**：WebCL 只是薄绑定，能力受浏览器沙箱与 OpenCL 1.2 限制。
- **vs WebAssembly + SIMD**：Wasm 提供可移植的 CPU 并行，不依赖 GPU 驱动。
- **vs WebGL**：WebGL 只能通过「渲染到纹理」间接做计算，WebCL 本应填补这个空白。

## 优势

- 复用 OpenCL 生态，kernel 代码可沿用。
- 理论上跨浏览器、跨 GPU 厂商。
- 比 WebGL hack 式 GPGPU 更规范。

## 局限

- 浏览器厂商普遍认为「暴露 OpenCL 等于暴露驱动攻击面」，安全顾虑大。
- OpenCL 本身在移动端支持就不完整，浏览器端更难保证。
- 规范与实现脱节，没有一家主流浏览器真正发布生产版本。
- 同期 WebGL 2.0 compute 缺失，WebCL 又迟迟不落地，开发者转投 WebAssembly。

## 为什么它消失了

- **只有 Nokia 和 Samsung 做过实验性实现**，Chrome、Firefox、Safari 均未跟进。
- **安全与沙箱**：GPU 驱动历史上漏洞多，浏览器不愿直接暴露。
- **W3C 另起炉灶**：2017 年起 W3C 的 GPU for the Web 工作组推进 WebGPU，2023 年落地，WebCL 被正式放弃。
- **OpenCL 自身衰退**：Apple 弃用 OpenCL 后，WebCL 在 Safari 上更无可能。

结论：WebCL 是「把桌面 API 直接搬进浏览器」的失败尝试，WebGPU 是重新设计后的正确答案。

## 参考

- 规范归档：<https://www.khronos.org/webcl/>
- Khronos WebCL 1.0 Specification（已停止维护）
- W3C GPU for the Web 工作组章程
