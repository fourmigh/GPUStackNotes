# OpenACC

> 2011 年发布的指令制导式并行编程标准，目标是用几行 `#pragma` 把已有的 Fortran / C 代码卸载到加速器。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | OpenACC 组织（PGI 发起，NVIDIA / Cray / CAPS 参与） |
| 首次公开 | 2011-11 发布 1.0 规范；2012-11 发布 2.0 |
| 类型 | 编译指令（pragma）标准 |
| 开闭源 · 许可 | 规范开放；编译器实现以商业闭源为主（NVHPC、Cray），GCC 有部分支持 |
| 核心语言 / 接口 | `#pragma acc`（`parallel` / `kernels` / `data` / `loop`） |
| 需要硬件 | 多厂商加速器，实际以 NVIDIA GPU 支持最好 |
| 当前状态 | 维护中（3.x），但生态被 OpenMP target 挤压 |

## 设计思路

OpenACC 的假设是：**科学计算代码已经写了几十年，不能重写成 CUDA**。所以它把并行化交给编译器，程序员只标注「这段循环可以并行」「这块数据需要拷到设备上」。

一个 `#pragma acc kernels` 就能让编译器自动分析循环依赖、生成 kernel、插入数据传输。相比 OpenMP target，OpenACC 更早专注加速器卸载，数据子句（`copyin` / `copyout` / `create`）设计更贴合 GPU。

## 最小示例

```c
#pragma acc data copyin(x[0:n], y[0:n]) copyout(z[0:n])
{
    // 编译器自动把循环映射成 GPU kernel
    #pragma acc parallel loop
    for (int i = 0; i < n; i++) {
        z[i] = a * x[i] + y[i];
    }
}
```

Fortran 版本几乎等价：

```fortran
!$acc parallel loop
do i = 1, n
    z(i) = a * x(i) + y(i)
end do
```

## 与其它方案的异同

- **vs OpenMP `target`**：目的高度重叠，OpenMP 4.0 起步晚但背靠更大的标准组织，现已成为主流选择；OpenACC 在 GPU 数据管理和 `kernels` 自动并行上更成熟。
- **vs CUDA / HIP**：OpenACC 无法精确控制线程块大小、共享内存布局，性能上限低于手写 kernel。
- **vs SYCL**：SYCL 是 C++ 单源、显式控制；OpenACC 是最小改动、隐式控制，两者面向完全不同的迁移预算。

## 优势

- 改造成本极低，几十万行 Fortran 天气/流体代码可以逐步迁移。
- 同一份代码用 `-acc` 开关即可在 CPU 与 GPU 间切换。
- NVIDIA HPC SDK 的 OpenACC 实现成熟，性能常接近手写 CUDA。

## 局限

- 编译器自动分析能力决定性能，遇到复杂依赖时可能完全不并行。
- 非 NVIDIA 平台的实现质量差，跨厂商可移植性名不副实。
- 调试困难：出问题时需要看编译器生成的中间代码。
- GCC 支持长期不完整，开源工具链体验差。

## 现状与生态

主要用户是 HPC 中心的老 Fortran/C 代码库（气象、CFD、分子动力学）。NVIDIA 是最大推动者（NVHPC 编译器）。随着 OpenMP 5.x 的 offload 能力完善，新项目更倾向 OpenMP，OpenACC 逐渐退守「历史代码增量加速」定位。

## 参考

- 官方文档：<https://www.openacc.org/>
- OpenACC 2.7 / 3.x Specification
- NVIDIA HPC SDK 文档
