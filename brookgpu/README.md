# BrookGPU

> 2003 年斯坦福大学的流式 GPU 编程语言，GPU 通用计算的起点。CUDA 的主要设计者 Ian Buck 出自这个项目。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 斯坦福大学图形实验室（Ian Buck 等） |
| 首次公开 | 2003 年（SIGGRAPH / GPGPU 论文） |
| 类型 | 流编程语言 + 源码到源码编译器 |
| 开闭源 · 许可 | 开源，BSD 式许可 |
| 核心语言 / 接口 | Brook C（`kernel` / `reduce` 流扩展），`brcc` 编译器 |
| 需要硬件 | 支持 shader 的早期 NVIDIA / ATI GPU（SM 2.0+） |
| 当前状态 | ⚠️ 已废弃（约 2012 年后无维护） |

## 设计思路

Brook 的核心洞察是：**把 GPU 当成流处理器，而不是显卡**。程序员只描述「对流里每个元素做什么」，不用管顶点/片元管线细节，编译器负责把 kernel 翻译成图形 API 的着色器，把输出写到纹理再读回内存。

它第一次把 stream kernel、gather/scatter、reduction 这套抽象带进 GPU 编程，直接影响了后来 CUDA 的语法与心智模型。

## 最小示例

```c
// 每个元素的流内核：b[i] = a[i] * 2
kernel void double_it(float a<>, out float b<>) {
    b = a * 2.0f;
}

int main() {
    float in[1024], out[1024];
    streamRead(in, inputStream);
    double_it(1024, in, out);   // 隐式并行：1024 个元素同时算
    streamWrite(outputStream, out);
}
```

`<>` 是 Brook 的流类型标记，和 CUDA 的 `__global__` 加线程索引是同一层的抽象。

## 与其它方案的异同

- **vs CUDA**：抽象层级几乎一致，但 Brook 缺少线程块、共享内存这些显式控制手段，性能上限低；CUDA 拿走了「流内核」的思想，补上了硬件映射。
- **vs OpenCL**：OpenCL 的 `__kernel` + NDRange 是同一思路的标准化版本，且直接对接驱动而非图形 API。
- **vs 图形 API 手写 shader**：Brook 屏蔽了「渲染成纹理再读回」的样板代码，这是它当时真正的价值。

## 优势

- 首次证明 GPU 可以当通用计算设备用，学术影响力极大。
- 语法极简，流式思维对非图形程序员友好。
- 跨 NVIDIA / ATI，当时少见的可移植。

## 局限

- 依赖图形 API 中转，性能和可编程性都被着色器模型限制。
- 无共享内存、无原子操作、无异步计算队列。
- 浮点精度受纹理格式约束（早期多为 16/32 位定点或半精度）。

## 为什么它消失了

- 2007 年 CUDA 发布，提供了 Brook 没有的硬件直通、共享内存和完整工具链，Ian Buck 本人也加入 NVIDIA 主导 CUDA。
- 后续 OpenCL 用开放标准覆盖了「跨厂商」这个卖点。
- 学术原型定位，缺少工业级维护和驱动适配。

## 参考

- 官方项目页（已下线，Web Archive 归档）：<https://web.archive.org/web/2015/http://graphics.stanford.edu/projects/brookgpu/>
- Buck et al., *Brook for GPUs: Stream Computing on Graphics Hardware*, SIGGRAPH 2004
