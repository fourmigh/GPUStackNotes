# CUDA

> 2007 年 NVIDIA 发布的 GPU 通用计算平台。本仓库所有其它方案，基本都是在回答「怎么不依赖 CUDA」这个问题。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | NVIDIA |
| 首次公开 | 2006-11 公测，**2007-06 发布 CUDA 1.0** |
| 类型 | 厂商专有计算平台（语言 + 运行时 + 驱动 + 算子库） |
| 开闭源 · 许可 | 闭源，SDK 免费下载；部分组件（如 cuDNN 前身、Nsight 插件、Thrust）开源 |
| 核心语言 / 接口 | C/C++ 扩展（`__global__` / `<<<>>>`）、PTX 中间表示、cuBLAS / cuDNN / NCCL / CUTLASS |
| 需要硬件 | 仅 NVIDIA GPU，Compute Capability 决定可用特性 |
| 当前状态 | 活跃，主线版本 CUDA 12.x / 13.x |

## 设计思路

CUDA 把 GPU 抽象成「网格 → 线程块 → 线程」三层，并暴露共享内存、warp 同步、原子操作等硬件细节。关键是它**直接对接驱动**，不经过图形 API，所以能榨干硬件。

生态是它真正的护城河：编译器 `nvcc`、调试器 `cuda-gdb`、性能分析 `Nsight`、数学库（cuBLAS/cuFFT/cuSPARSE）、深度学习库（cuDNN/TensorRT）、通信库（NCCL）全部齐备，且深度学习框架默认后端就是它。

## 最小示例

```cuda
__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    float *dx, *dy;
    cudaMalloc(&dx, N * sizeof(float));      // 显式管理设备内存
    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, dx, dy);   // 网格×块 的启动配置
    cudaDeviceSynchronize();
}
```

## 与其它方案的异同

- **vs OpenCL**：CUDA 是「一套实现、一套工具链」；OpenCL 是「一套规范、多套实现」，可移植但质量参差。
- **vs HIP**：HIP 是 CUDA API 的近亲克隆，源码级可迁移，但库生态与 Windows 支持弱于 CUDA。
- **vs Triton / Pallas**：CUDA 是底层手写；这些 DSL 让编译器替你生成 kernel，牺牲少量控制力换取开发效率。
- **vs ZLUDA**：ZLUDA 是让 CUDA 程序跑在非 NVIDIA 硬件上的翻译层，属于「逆向兼容」。

## 优势

- 生态完整度断层第一，几乎不存在「想做的事没有库」。
- 文档、教程、社区问答体量巨大。
- 性能可控性最强，从 warp 级到集群级都能调。
- 与 PyTorch / TensorFlow / vLLM / llama.cpp 等默认深度绑定。

## 局限

- 只能跑在 NVIDIA 硬件上，供应链风险集中。
- 闭源，编译器和驱动内部行为不可改。
- 学习曲线陡，写出正确代码容易、写出高性能代码难。
- 版本与驱动 / Compute Capability 之间存在兼容矩阵，运维有成本。

## 现状与生态

深度学习的事实标准后端。即便上层框架支持多后端，绝大多数生产环境仍以 CUDA 为第一优先级，其它后端常处于「能跑但不保证性能」的状态。这也是国产 GPU 厂商普遍提供 CUDA 风格 API 或兼容层的原因。

## 参考

- 官方下载与文档：<https://developer.nvidia.com/cuda-toolkit>
- *Programming Massively Parallel Processors*（Kirk & Hwu）

## 本机示例（examples/）

本机环境：GeForce MX230（sm_61，2 GB）、CUDA 12.5、MSVC 14.44。

- `examples/demo.cu`：入门示例（逐行中文注释），含设备查询、向量加法、矩阵乘法三段，适合学习。
- `examples/demo_bench.cu`：让同一批计算跑三遍做性能对比——`[1]` GPU + `cudaMemcpy`、`[2]` GPU + `cudaMallocManaged`（统一内存）、`[3]` CPU，逐步骤打印耗时。

编译运行（在仓库根目录执行）：

```powershell
# 编译运行 demo.cu（默认，不传参数即可）
powershell -ExecutionPolicy Bypass -File cuda/scripts/build.ps1

# 编译运行 demo_bench.cu（第一个参数是源文件）
powershell -ExecutionPolicy Bypass -File cuda/scripts/build.ps1 cuda/examples/demo_bench.cu
```

通用脚本还支持 `-Out`（输出名）、`-Arch`（架构，默认 sm_61）、`-NoRun`（只编译）、`-NoClean`（保留 .lib/.exp）等参数。

> MSVC 14.44 比 CUDA 12.5 官方支持的新，编译需加 `-allow-unsupported-compiler`（脚本已内置）。
> CUDA 安装在 `F:\CUDA\v12.5`（`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5` 是指向它的 Junction），MSVC 安装在 `F:\VSBuildTools`。

`scripts/relocate_cuda.ps1` 是把 CUDA 从 C 盘搬到 F 盘并建立 Junction 的一次性脚本，留作参考。
