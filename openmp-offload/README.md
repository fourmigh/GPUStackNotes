# OpenMP target offload

> 2013 年 OpenMP 4.0 引入的 `target` 指令，让已有 OpenMP 并行代码用几行 pragma 就能把循环卸载到 GPU。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | OpenMP Architecture Review Board |
| 首次公开 | 2013-07（OpenMP 4.0）；后续 5.0（2018）大幅增强 |
| 类型 | 编译指令（pragma）标准 |
| 开闭源 · 许可 | 开放规范；GCC、Clang/LLVM 开源实现，厂商编译器闭源 |
| 核心语言 / 接口 | `#pragma omp target` / `teams` / `distribute` / `map` |
| 需要硬件 | 多厂商 GPU、CPU、FPGA（取决于编译器后端） |
| 当前状态 | 活跃，OpenMP 5.2 / 6.0 持续演进 |

## 设计思路

OpenMP 的哲学是「同一份代码，编译器决定在哪跑」。4.0 加入 `target` 后，原本的 `#pragma omp parallel for` 只需再加一层 `target`，就能把执行搬到一个「设备」上。

关键在于 `map` 子句显式声明数据方向（`to` / `from` / `tofrom`），比 OpenACC 的自动分析更可控。5.0 之后引入了 `declare variant`、`requires`、异步任务依赖等，能力已相当完整。

## 最小示例

```c
#pragma omp target data map(to: x[0:n], y[0:n]) map(from: z[0:n])
{
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++) {
        z[i] = a * x[i] + y[i];
    }
}
```

编译：`gcc -fopenmp -foffload=nvptx-none ...` 或 `clang -fopenmp --offload-arch=sm_80 ...`

三层循环指令（`teams` → `distribute` → `parallel for`）分别对应「线程块 → 块内分配 → 线程」，是 OpenMP 表达 GPU 硬件层级的方式。

## 与其它方案的异同

- **vs OpenACC**：几乎是同类竞品，OpenMP 标准化组织更大、GCC/Clang 支持更好；OpenACC 在 GPU 数据布局上细节更丰富。
- **vs CUDA / HIP**：OpenMP 缺少显式共享内存、warp 原语，极致性能调优受限。
- **vs SYCL**：SYCL 是显式 API，OpenMP 是 pragma；前者适合新代码，后者适合存量代码。
- **vs C++ 标准并行算法**：`std::execution::par` 目前仍主要面向 CPU，GPU 卸载靠 vendor 扩展。

## 优势

- 迁移成本最低，一个已有 OpenMP 程序加两三行即可试用 GPU。
- GCC 与 Clang 都有开源实现，不需要商业编译器。
- 与 CPU 多线程、任务并行共用一套语法，代码只有一份。
- OpenMP 5.x 已支持 `target` 上的异步任务与设备端函数调用。

## 局限

- 卸载性能极度依赖编译器版本与后端成熟度，GCC 的 offload 优化弱于 Clang 和厂商编译器。
- 设备端调试困难，错误信息常指向生成的中间代码。
- 缺少 warp 级编程，无法实现 CUDA 那种极致优化。
- 多厂商 GPU 后端的实际可用性不如宣传——NVIDIA 支持最好，AMD 次之。

## 现状与生态

HPC 领域的主流选择之一，与 OpenACC 形成竞争并逐渐占优。大量科学计算软件（LAMMPS、GROMACS 部分路径、混合精度 CFD 代码）采用 OpenMP target 做渐进式 GPU 化。Clang 的 offload 支持是当前最活跃的开源实现。

## 参考

- 官方文档：<https://www.openmp.org/>
- OpenMP 5.0 / 5.2 Specification（Chapter: Device Constructs）
- LLVM OpenMP GPU offload 设计文档
