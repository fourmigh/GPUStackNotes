# C++ AMP

> 2012 年微软提出的 C++ 语言扩展，试图用「纯 C++ 语法 + 数组视图」把 GPU 计算写进标准 C++ 习惯里。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Microsoft |
| 首次公开 | 2012-02（随 Visual Studio 2012 发布） |
| 类型 | C++ 语言扩展 + 运行时库 |
| 开闭源 · 许可 | 规范开放（微软公开提案），实际实现仅 MSVC 闭源 |
| 核心语言 / 接口 | C++11 + `restrict(amp)`、`concurrency::array_view`、`parallel_for_each` |
| 需要硬件 | Windows + DirectX 11 及以上 GPU |
| 当前状态 | ⚠️ 已停止演进（2014 年后无更新，MSVC 仍保留以兼容旧代码） |

## 设计思路

C++ AMP 的目标是「不发明新语言」。它用 `restrict(amp)` 标注一个 lambda 只能在加速器上跑，用 `array_view<T>` 表达「主机与设备共享的逻辑数组」，运行时负责在 GPU 上分配 `array` 并做同步拷贝。

它是**规范层面**的努力：微软把规范公开，希望其它编译器厂商跟进，让 C++ 标准吸收 GPU 计算。这个设想最终失败了，但影响了后来的 SYCL 和 C++ 标准中的并行算法。

## 最小示例

```cpp
#include <amp.h>
using namespace concurrency;

void saxpy(int n, float a, const std::vector<float>& x, std::vector<float>& y) {
    array_view<const float, 1> av_x(n, x);
    array_view<float, 1> av_y(n, y);

    parallel_for_each(av_y.extent, [=](index<1> idx) restrict(amp) {
        av_y[idx] = a * av_x[idx] + av_y[idx];   // 在 GPU 上执行
    });
    av_y.synchronize();   // 拷回主机
}
```

`array_view` 的惰性同步是亮点：多次访问会合并拷贝，不需要手写 `cudaMemcpy`。

## 与其它方案的异同

- **vs CUDA**：C++ AMP 无法指定线程块大小、无共享内存原语、无 `warp` 概念，性能天花板明显更低。
- **vs OpenCL**：都是「跨厂商」思路，但 C++ AMP 用现代 C++ 表达，比 OpenCL 的 C99 + 样板代码更安全。
- **vs SYCL**：SYCL 是 C++ AMP 思路的「正统继承人」——单源 C++、模板化、由 Khronos 标准化并真正跨厂商。
- **vs OpenMP offload**：C++ AMP 用 API，OpenMP 用 pragma，迁移策略不同。

## 优势

- 语法最接近普通 C++，无 `__global__`、无 `#pragma`。
- `array_view` 的自动同步模型对初学者友好。
- 与 PPL（Parallel Patterns Library）衔接，CPU 并行和 GPU 并行可混写。

## 局限

- 仅 MSVC 实现，且仅 Windows + DX11，实际上「跨厂商」只是口号。
- 无法精细控制 GPU，性能不如手写 CUDA/HLSL。
- 编译器对模板与 `restrict(amp)` 的组合限制多，报错晦涩。
- 微软自身战略转向 C++17 并行算法与 DirectML 后停止投入。

## 为什么它消失了

- **没有第二个实现**：规范开放但无人跟进，等于专有技术。
- **平台绑定**：Windows + DirectX 11，与 CUDA 的「全平台」定位竞争失败。
- **SYCL 的替代**：Khronos 用 SYCL 提供了真正标准化、多实现的单源 C++ 方案。
- **微软重心转移**：2014 年后资源转向 DirectX 12、DirectML 和 C++ 标准并行算法（P0024）。

结论：C++ AMP 是正确方向的过早尝试，概念被 SYCL 继承。

## 参考

- 官方文档：<https://learn.microsoft.com/en-us/cpp/parallel/amp/cpp-amp-overview>
- Microsoft: *C++ AMP: Language and Programming Model*（规范）
- Kate Gregory / Ade Miller, *C++ AMP*
