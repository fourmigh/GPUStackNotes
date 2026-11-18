# RenderScript Compute

> 2011 年 Android 平台上的异构计算框架，让 Java 应用把计算任务自动分派到 CPU、GPU 或 DSP。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Google |
| 首次公开 | 2011-02（Android 3.0 Honeycomb 引入）；2012 年 Android 4.1 加入 Compute |
| 类型 | Android 运行时框架 + 类 C99 语言 |
| 开闭源 · 许可 | 开源（AOSP） |
| 核心语言 / 接口 | RenderScript C99（`.rs` 文件）+ Java/Kotlin 反射 API |
| 需要硬件 | Android 设备（GPU / DSP 后端由驱动提供） |
| 当前状态 | ⚠️ 已废弃（Android 12 起移除，官方推荐 Vulkan 或 OpenCL） |

## 设计思路

RenderScript 的定位非常「Android」：**开发者不关心底层是什么芯片**。你写 `.rs` 内核，运行时根据设备能力在 CPU、多核 CPU、GPU、DSP 之间选后端，并自动处理数据在内存间的搬运。

它同时提供 `rsForEach` 这类高层原语，以及 `Allocation` 这种跨设备共享缓冲区，思路接近后来的「统一内存 + 自动调度」。

## 最小示例

```c
// blur.rs
#pragma version(1)
#pragma rs java_package_name(com.example.app)

uchar4 __attribute__((kernel)) invert(uchar4 in) {
    uchar4 out = in;
    out.r = 255 - in.r;
    out.g = 255 - in.g;
    out.b = 255 - in.b;
    return out;
}
```

```java
// Java 侧：一次 forEach 自动并行
ScriptC_blur script = new ScriptC_blur(rs);
script.forEach_invert(inputAllocation, outputAllocation);
```

## 与其它方案的异同

- **vs OpenCL**：RenderScript 是更高层的「设备无关」抽象，开发者不写 NDRange；OpenCL 需要显式管理设备、上下文和内存。
- **vs Vulkan Compute**：Vulkan 给开发者完全控制权，RenderScript 把选择权交给运行时——代价是性能不可预测。
- **vs WebCL**：同属「把 GPU 计算藏进运行时」的思路，一个面向浏览器一个面向 Android。

## 优势

- 真正设备无关，同一份代码在低端机和高通/联发科旗舰上都能跑。
- 自动后端选择，无需为不同 SoC 分支。
- 与 Android 图形栈（Surface、Bitmap）集成自然。

## 局限

- 后端选择由驱动决定，性能调优无从下手。
- 编译期生成大量胶水代码，APK 体积膨胀。
- 支持的内核语言受限，复杂算法表达困难。
- 各厂商 GPU 后端实际优化水平差异巨大。

## 为什么它消失了

- **性能不可控**：开发者抱怨同一份 RenderScript 在不同 SoC 上性能差数倍，最后还是要写多份特化代码。
- **Google 战略转向**：Android 7 起推荐 Vulkan，Android 11 起标记弃用，Android 12（2021）彻底移除 Java API。
- **生态碎片化**：厂商对 RenderScript 驱动的投入远低于对 Vulkan / OpenCL 的投入。
- **被更明确的方案替代**：图像处理走 Vulkan Compute 或 RenderEffect，AI 推理走 NNAPI / TFLite GPU delegate。

现在读老 Android 代码可能还会遇到 `.rs` 文件，新项目不要使用。

## 参考

- 官方移除公告：<https://developer.android.com/about/versions/12/behavior-changes-12#renderscript>
- Android Developers（归档）: *RenderScript*（页面已下线）
