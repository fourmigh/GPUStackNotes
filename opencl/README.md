# OpenCL

> 2009 年由 Apple 发起、Khronos 标准化的跨厂商并行计算框架。工业界第一个真正的「GPU 通用计算开放标准」。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Khronos Group（Apple 最初提案） |
| 首次公开 | 2008-12 发布 1.0 规范，**2009-08 由 Khronos 正式批准** |
| 类型 | 跨厂商开放标准（API + 类 C 语言） |
| 开闭源 · 许可 | 规范开放；实现多为厂商闭源驱动（POCL 为开源实现） |
| 核心语言 / 接口 | OpenCL C（基于 C99）、SPIR-V（2.1 起）、Host API |
| 需要硬件 | NVIDIA / AMD / Intel GPU、CPU、部分 ARM GPU、DSP、FPGA |
| 当前状态 | 维护中，但演进缓慢；3.0 后厂商跟进不足 |

## 设计思路

OpenCL 把「平台 → 设备 → 上下文 → 命令队列 → 内核」这套对象模型标准化。硬件被抽象成 NDRange（全局工作项 + 工作组），程序员用 `get_global_id()` 取索引，写法和 CUDA 的线程索引高度类似。

它的根本取舍是：**先保证任何厂商都能实现，再考虑性能上限**。因此规范要覆盖 GPU、CPU、DSP、FPGA，抽象必须足够保守。

## 最小示例

```c
// kernel.cl
__kernel void saxpy(__global const float *x, __global float *y, float a) {
    int i = get_global_id(0);
    y[i] = a * x[i] + y[i];
}
```

```c
// Host 侧
cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
cl_kernel k = clCreateKernel(prog, "saxpy", &err);
clEnqueueNDRangeKernel(queue, k, 1, NULL, global, local, 0, NULL, NULL);
```

注意 Host 侧需要几十行样板代码，这是 OpenCL 最被诟病的地方。

## 与其它方案的异同

- **vs CUDA**：语法相近，但 OpenCL 不提供 warp 级原语、动态并行、统一虚拟寻址等能力；CUDA 的库生态也远强。
- **vs Vulkan Compute**：Vulkan 面向图形与低层显存控制，需要手工管理管线；OpenCL 抽象更高、更偏计算。
- **vs SYCL**：SYCL 是「C++ 单源版 OpenCL」，可跑在 OpenCL 后端之上，解决样板代码和类型安全问题。
- **vs WebCL**：WebCL 只是 OpenCL 的浏览器 JS 绑定，随 WebCL 停滞而失去意义。

## 优势

- 覆盖面最广：同一份 kernel 能从手机 GPU 跑到服务器 GPU 再到 FPGA。
- 长期稳定的工业标准，老代码库存量巨大（图像处理、科学计算、挖矿、视频编码）。
- 部分厂商（Intel、AMD）的 OpenCL 实现性能相当接近原生。

## 局限

- Host 侧样板代码冗长，无 RAII、无模板，易出错。
- 各厂商编译器优化水平差异大，同一 kernel 跨平台性能不可预测。
- 1.2 → 2.x 的特性（SVM、管道、泛型地址空间）落地不全，导致实际项目停留在 1.2。
- 深度学习生态几乎为零，没有对标 cuDNN 的库。

## 现状与生态

仍在被大量存量项目使用（FFmpeg 滤镜、OpenCV 的 T-API、Blender Cycles、部分密码学库），但新项目普遍转向 SYCL、Vulkan Compute 或直接 CUDA/HIP。Apple 已停止在 macOS 上更新 OpenCL 支持。

## 参考

- 官方文档：<https://www.khronos.org/opencl/>
- Khronos OpenCL Specification 1.2 / 3.0
- *OpenCL Programming Guide*（Munshi et al.）
