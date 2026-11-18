// ============================================================================
// demo_bench.cu —— CUDA C++ 入门示例（三遍耗时对比）
//
// 演示三个最核心的用法：
//   1. 查询 GPU 设备信息（型号、计算能力、SM 数量、显存大小）
//   2. 向量加法（一维线程网格）
//   3. 矩阵乘法（二维线程网格）
//
// 每个计算各跑「三遍」，对比三种方式的耗时：
//   [1] GPU + cudaMemcpy        —— 显式把数据拷进/拷出显存
//   [2] GPU + cudaMallocManaged —— 统一内存，不写 memcpy，底层隐式迁移
//   [3] CPU                     —— 单线程 for 循环
//
// 计时工具：
//   cudaEvent —— 精确测量 GPU 上的异步操作（memcpy、kernel 执行）
//   <chrono> —— 测量 host 端同步操作（cudaMalloc / cudaFree）和 CPU 循环
//
// 名词约定：
//   host   = CPU + 主机内存（普通的 C/C++ 代码跑在 host）
//   device = GPU + 显存（__global__ 函数跑在 device）
//   kernel = 在 GPU 上执行的函数，由 host 用 <<<...>>> 语法启动
// ============================================================================

#include <cstdio>          // C 标准输入输出库：printf（打印到屏幕）、fprintf（打印到 stderr）
#include <cstdlib>         // C 标准库：malloc / free（分配释放主机内存）、exit（退出）、system（调用外部命令）
#include <cmath>           // C 数学库：fabs（浮点取绝对值，用于结果校验）
#include <chrono>          // C++ 高精度计时库：high_resolution_clock，测量 CPU 耗时
#include <cuda_runtime.h>  // CUDA 运行时 API：cudaMalloc / cudaMemcpy / cudaMallocManaged / kernel 启动等

#ifdef _WIN32
#include <windows.h>   // Windows API：SetConsoleOutputCP 用来设置控制台输出代码页
#include <conio.h>     // _getch：无缓冲读取一个按键（Windows 专属）
#endif

// ---------------------------------------------------------------------------
// CUDA_CHECK 宏：给每个 CUDA 调用包一层错误检查。
//
// CUDA 运行时 API 绝大多数返回 cudaError_t 状态码，只有 cudaSuccess 表示成功。
// 用这个宏包住调用，出错时打印「错误原因 + 文件名 + 行号」并立即退出，
// 省得每个调用后面都手写一段 if 判断。
//
// 宏体逐行含义：
//   cudaError_t err = (call)      —— 执行调用，保存返回的状态码
//   if (err != cudaSuccess)       —— 状态码不是「成功」就报错
//   cudaGetErrorString(err)       —— 把状态码翻译成可读的错误描述
//   __FILE__ / __LINE__           —— 出错处的文件名和行号，方便定位
//   exit(EXIT_FAILURE)            —— 以失败码结束整个程序
//
// 外层 do { ... } while(0) 是宏的经典写法：让宏展开后始终是「一条语句」，
// 这样无论调用处后面有没有分号、放在 if/else 里都不会产生语法错误。
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 向量加法 kernel：计算 out[i] = a[i] + b[i]，由 GPU 上成千上万个线程并行完成。
//
// __global__ 修饰符表示这是「kernel」——
// 由 host 用 <<<块数, 线程数>>> 语法启动，实际在 device（GPU）上执行。
// kernel 的返回类型必须是 void，结果通过指针参数写回。
// 参数里的 const 表示这个指针在 kernel 里只读、不修改。
//
// 关于 blockIdx / blockDim / threadIdx：
//   它们是「CUDA 编译器内建变量」，不是 cuda_runtime.h 里的 API——
//   由 nvcc 编译 device 代码时自动提供，值由 GPU 硬件在启动 kernel 时填充，
//   所以即使不 include 任何头文件，kernel 里也能直接用；host 代码里则不能用。
//   同类内建还有 gridDim（网格总块数）和 warpSize（一个 warp 恒为 32 个线程）。
// ---------------------------------------------------------------------------
__global__ void vecAdd(const float* a, const float* b, float* out, int n) {
    // 计算当前线程的全局下标 i：
    //   blockIdx.x   当前线程块在整个网格中的编号（第几块）
    //   blockDim.x   每个线程块里有多少线程
    //   threadIdx.x  当前线程在自己块内的编号（第几个）
    // 三者组合 = 块号 * 每块线程数 + 块内编号 = 全局唯一的线程号
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界保护：线程总数（块数 * 每块线程数）通常向上取整、略大于 n，
    // 多出来的线程不干活，否则会访问越界内存。
    if (i < n) out[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// 矩阵乘法 kernel：计算 out = a * b（都是 n x n 的方阵）。
// 与向量加法不同，这里用「二维」线程块，让每个线程对应输出矩阵的一个元素。
// ---------------------------------------------------------------------------
__global__ void matMul(const float* a, const float* b, float* out, int n) {
    // 当前线程负责输出矩阵的第 row 行：
    // blockIdx.y / blockDim.y / threadIdx.y 是 y 方向（行）的内建变量
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 第 col 列：x 方向的内建变量
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界保护（线程网格按 16 对齐，可能比 n 大）
    if (row < n && col < n) {
        // acc 累加「a 的第 row 行」与「b 的第 col 列」的点积
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) {
            // a[row][k] * b[k][col]，逐项相乘再累加；
            // 矩阵在显存里按「行优先」一维存放，所以用 row*n+k 定位第 row 行第 k 列
            acc += a[row * n + k] * b[k * n + col];
        }
        // 把累加结果写回输出矩阵的对应位置
        out[row * n + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// main 跑在 host（CPU）上，负责「调度」GPU，并对三种方式分别计时。
// ---------------------------------------------------------------------------
int main() {
#ifdef _WIN32
    // 把控制台输出代码页设为 UTF-8：源码里的中文字符串是 UTF-8 编码，
    // 而 Windows 控制台默认按 GBK 解释，不设置的话中文会乱码。
    SetConsoleOutputCP(CP_UTF8);
#endif

    // ===================== 1. 查询设备信息 =====================
    int dev = 0;                 // 设备编号：本机只有一张 NVIDIA 卡，就是 0 号
    cudaDeviceProp prop;         // cudaDeviceProp 结构体，用来装设备的各种属性
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));  // 把 0 号设备的属性读进 prop

    std::printf("=== Device ===\n");
    std::printf("Name:      %s\n", prop.name);                 // 显卡型号
    std::printf("CC:        %d.%d\n", prop.major, prop.minor); // 计算能力（Compute Capability），决定可用特性
    std::printf("SMs:       %d\n", prop.multiProcessorCount);  // SM（流多处理器）数量，越多并行能力越强
    std::printf("Memory:    %.0f MiB\n", prop.totalGlobalMem / 1024.0 / 1024.0); // 显存总量（字节换算成 MiB）

    // 创建两个 CUDA 事件，后面反复用来给 GPU 上的异步操作打时间戳
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // 计时用的变量，两个例子里复用（只声明一次，后面重新赋值）
    std::chrono::high_resolution_clock::time_point t0, t1;  // 一组时间点
    float mallocMs, h2dMs, kernelMs, d2hMs, freeMs;   // [1] GPU+memcpy 各步骤
    float mgAllocMs, mgKernelMs, mgReadMs, mgFreeMs;  // [2] GPU+managed 各步骤
    float mgTotalMs;                                  // [2] 端到端总时间
    float cpuMs;                                      // [3] CPU

    // ===================== 2. 向量加法：三遍对比 =====================
    std::printf("\n=== Vector add (N=10000000) ===\n");
    const int N = 10000000;              // 向量长度：一千万个元素
    size_t bytes = N * sizeof(float);    // 需要的字节数 = 元素个数 * 每个 float 占 4 字节
                                         // size_t 是无符号整数类型，专门用来表示「字节大小」这类非负数

    // 输入数据（host 侧），第 [1]、[3] 遍共用
    float *h_a = (float*)std::malloc(bytes);   // 加数 a；malloc 返回 void*，需强转成 float*
    float *h_b = (float*)std::malloc(bytes);   // 加数 b
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // ===================== [1] GPU + cudaMemcpy =====================
    float *h_out = (float*)std::malloc(bytes);   // 接收 GPU 结果
    float *d_a, *d_b, *d_out;                    // d_ 前缀表示 device 侧，指向显存地址

    // cudaMalloc：host 端同步调用，用 chrono 计时。
    // 注意：第一次调用会触发 CUDA 上下文初始化，耗时异常大，属正常现象。
    t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_a, bytes));   // 传 &d_a（指针的地址），让 cudaMalloc 把分到的显存地址写进 d_a
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    t1 = std::chrono::high_resolution_clock::now();
    mallocMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // memcpy Host->Device：异步操作，用 cudaEvent 计时
    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&h2dMs, ev0, ev1);

    // kernel：异步执行，用 cudaEvent 计时
    int threads = 256;                        // 每个线程块放 256 个线程
                                              // 256 是 32 的整数倍：GPU 以 32 线程为一个 warp（线程束）调度，凑整便于充分利用硬件
    int blocks = (N + threads - 1) / threads; // 需要的块数 = 向上取整(N / 256)
    cudaEventRecord(ev0);
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_out, N);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&kernelMs, ev0, ev1);
    CUDA_CHECK(cudaGetLastError());           // 检查 kernel 启动有没有报错

    // memcpy Device->Host：把结果拷回，异步操作，用 cudaEvent 计时
    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&d2hMs, ev0, ev1);

    // cudaFree：host 端同步调用，用 chrono 计时
    t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    t1 = std::chrono::high_resolution_clock::now();
    freeMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ===================== [2] GPU + cudaMallocManaged =====================
    // 统一内存：CPU 和 GPU 共享同一地址空间，代码里不写 cudaMemcpy。
    // 但底层搬运并未消失——kernel 首次访问数据时按需「隐式迁移」，
    // 所以下面的 kernel 计时里其实包含了隐式的 H->D 迁移。
    float *h_mout = (float*)std::malloc(bytes);   // 用于把 managed 结果读到 host 侧保存
    {
        float *m_a, *m_b, *m_out;                 // m_ 前缀表示 managed（统一内存）
        auto gStart = std::chrono::high_resolution_clock::now();   // 端到端计时起点

        // 分配 + host 初始化（统一内存不用 memcpy，host 直接写）
        t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMallocManaged(&m_a, bytes));
        CUDA_CHECK(cudaMallocManaged(&m_b, bytes));
        CUDA_CHECK(cudaMallocManaged(&m_out, bytes));
        for (int i = 0; i < N; ++i) { m_a[i] = 1.0f; m_b[i] = 2.0f; }  // host 直接写，无需 memcpy
        t1 = std::chrono::high_resolution_clock::now();
        mgAllocMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        cudaEventRecord(ev0);                     // kernel 首次访问 m_a/m_b 触发隐式 H->D
        vecAdd<<<blocks, threads>>>(m_a, m_b, m_out, N);
        cudaEventRecord(ev1);
        CUDA_CHECK(cudaEventSynchronize(ev1));
        cudaEventElapsedTime(&mgKernelMs, ev0, ev1);

        // host 读回：第一次读 m_out 触发隐式 D->H 迁移，用 chrono 计时
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; ++i) h_mout[i] = m_out[i];
        t1 = std::chrono::high_resolution_clock::now();
        mgReadMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 释放
        t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaFree(m_a)); CUDA_CHECK(cudaFree(m_b)); CUDA_CHECK(cudaFree(m_out));
        t1 = std::chrono::high_resolution_clock::now();
        mgFreeMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto gEnd = std::chrono::high_resolution_clock::now();     // 端到端计时终点
        mgTotalMs = (float)std::chrono::duration<double, std::milli>(gEnd - gStart).count();
    }

    // ===================== [3] CPU =====================
    float *h_cpu = (float*)std::malloc(bytes);    // CPU 结果（同时作校验基准）
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) h_cpu[i] = h_a[i] + h_b[i];
    t1 = std::chrono::high_resolution_clock::now();
    cpuMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 校验：两个 GPU 结果分别和 CPU 基准逐项比较。
    // 不用 == 精确比较：GPU 浮点运算有舍入误差，应留一个容差（tolerance）
    bool ok_memcpy = true, ok_managed = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_out[i] - h_cpu[i]) > 1e-5f) { ok_memcpy = false; break; }
    }
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_mout[i] - h_cpu[i]) > 1e-5f) { ok_managed = false; break; }
    }

    std::printf("  三遍计算量相同：1000 万次加法\n");
    std::printf("  [1] GPU + memcpy   : %8.3f ms   (分配 %.3f / H2D %.3f / 计算 %.3f / D2H %.3f / 释放 %.3f)\n",
                mallocMs + h2dMs + kernelMs + d2hMs + freeMs, mallocMs, h2dMs, kernelMs, d2hMs, freeMs);
    std::printf("  [2] GPU + managed  : %8.3f ms   (分配+初始化 %.3f / 计算 %.3f含H2D迁移 / 读回 %.3f含D2H迁移 / 释放 %.3f)\n",
                mgTotalMs, mgAllocMs, mgKernelMs, mgReadMs, mgFreeMs);
    std::printf("  [3] CPU            : %8.3f ms   (纯计算)\n", cpuMs);
    std::printf("  correct: memcpy=%s, managed=%s\n", ok_memcpy ? "YES" : "NO", ok_managed ? "YES" : "NO");

    std::free(h_out); std::free(h_mout); std::free(h_cpu);

    // ===================== 3. 矩阵乘法：三遍对比 =====================
    std::printf("\n=== Matrix mul (512x512) ===\n");
    const int M = 512;                              // 512 x 512 的方阵
    size_t mbytes = (size_t)M * M * sizeof(float);  // 矩阵的总字节数

    // 输入数据（host 侧）
    float *h_x = (float*)std::malloc(mbytes); // 左矩阵
    float *h_y = (float*)std::malloc(mbytes); // 右矩阵
    for (int i = 0; i < M * M; ++i) {
        h_x[i] = (i % 7) * 0.1f;
        h_y[i] = ((i * 3) % 5) * 0.2f;
    }

    // 二维启动配置（三遍共用）：
    // dim3 是三维向量类型；每个线程块 16x16 = 256 个线程（凑满 warp 的整数倍）
    dim3 block(16, 16);
    dim3 grid(M / 16, M / 16);   // 网格也是二维：每维 512/16 = 32 个块，共 32x32 个块

    // ===================== [1] GPU + cudaMemcpy =====================
    float *h_z = (float*)std::malloc(mbytes);   // 接收 GPU 结果
    float *d_x, *d_y, *d_z;
    t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_x, mbytes));
    CUDA_CHECK(cudaMalloc(&d_y, mbytes));
    CUDA_CHECK(cudaMalloc(&d_z, mbytes));
    t1 = std::chrono::high_resolution_clock::now();
    mallocMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, mbytes, cudaMemcpyHostToDevice));
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&h2dMs, ev0, ev1);

    cudaEventRecord(ev0);
    matMul<<<grid, block>>>(d_x, d_y, d_z, M);
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&kernelMs, ev0, ev1);
    CUDA_CHECK(cudaGetLastError());

    cudaEventRecord(ev0);
    CUDA_CHECK(cudaMemcpy(h_z, d_z, mbytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(ev1);
    CUDA_CHECK(cudaEventSynchronize(ev1));
    cudaEventElapsedTime(&d2hMs, ev0, ev1);

    t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    t1 = std::chrono::high_resolution_clock::now();
    freeMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ===================== [2] GPU + cudaMallocManaged =====================
    float *h_mz = (float*)std::malloc(mbytes);   // 用于把 managed 结果读到 host 侧
    {
        float *m_x, *m_y, *m_z;
        auto gStart = std::chrono::high_resolution_clock::now();   // 端到端计时起点

        t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMallocManaged(&m_x, mbytes));
        CUDA_CHECK(cudaMallocManaged(&m_y, mbytes));
        CUDA_CHECK(cudaMallocManaged(&m_z, mbytes));
        for (int i = 0; i < M * M; ++i) {
            m_x[i] = (i % 7) * 0.1f;
            m_y[i] = ((i * 3) % 5) * 0.2f;
        }
        t1 = std::chrono::high_resolution_clock::now();
        mgAllocMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        cudaEventRecord(ev0);
        matMul<<<grid, block>>>(m_x, m_y, m_z, M);
        cudaEventRecord(ev1);
        CUDA_CHECK(cudaEventSynchronize(ev1));
        cudaEventElapsedTime(&mgKernelMs, ev0, ev1);

        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < M * M; ++i) h_mz[i] = m_z[i];
        t1 = std::chrono::high_resolution_clock::now();
        mgReadMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaFree(m_x)); CUDA_CHECK(cudaFree(m_y)); CUDA_CHECK(cudaFree(m_z));
        t1 = std::chrono::high_resolution_clock::now();
        mgFreeMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

        auto gEnd = std::chrono::high_resolution_clock::now();
        mgTotalMs = (float)std::chrono::duration<double, std::milli>(gEnd - gStart).count();
    }

    // ===================== [3] CPU =====================
    float *h_cpu_z = (float*)std::malloc(mbytes);  // CPU 结果（同时作校验基准）
    t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            float s = 0.0f;
            for (int k = 0; k < M; ++k) s += h_x[r * M + k] * h_y[k * M + c];
            h_cpu_z[r * M + c] = s;
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    cpuMs = (float)std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 校验：两个 GPU 结果和 CPU 基准逐项比较，记录最大误差
    float maxErrMemcpy = 0.0f, maxErrManaged = 0.0f;
    for (int i = 0; i < M * M; ++i) {
        float d1 = std::fabs(h_z[i] - h_cpu_z[i]);
        float d2 = std::fabs(h_mz[i] - h_cpu_z[i]);
        if (d1 > maxErrMemcpy) maxErrMemcpy = d1;
        if (d2 > maxErrManaged) maxErrManaged = d2;
    }

    std::printf("  三遍计算量相同：512³ = 1.34 亿次乘加\n");
    std::printf("  [1] GPU + memcpy   : %8.3f ms   (分配 %.3f / H2D %.3f / 计算 %.3f / D2H %.3f / 释放 %.3f)\n",
                mallocMs + h2dMs + kernelMs + d2hMs + freeMs, mallocMs, h2dMs, kernelMs, d2hMs, freeMs);
    std::printf("  [2] GPU + managed  : %8.3f ms   (分配+初始化 %.3f / 计算 %.3f含H2D迁移 / 读回 %.3f含D2H迁移 / 释放 %.3f)\n",
                mgTotalMs, mgAllocMs, mgKernelMs, mgReadMs, mgFreeMs);
    std::printf("  [3] CPU            : %8.3f ms   (纯计算)\n", cpuMs);
    std::printf("  correct: memcpy=%s (max err %g), managed=%s (max err %g)\n",
                maxErrMemcpy < 1e-2f ? "YES" : "NO", maxErrMemcpy,
                maxErrManaged < 1e-2f ? "YES" : "NO", maxErrManaged);

    std::free(h_z); std::free(h_mz); std::free(h_cpu_z);

    // ===================== 4. 释放剩余资源 =====================
    std::free(h_a); std::free(h_b);
    std::free(h_x); std::free(h_y);

    CUDA_CHECK(cudaEventDestroy(ev0));  // 销毁计时事件
    CUDA_CHECK(cudaEventDestroy(ev1));

    // ===================== 5. 结束前暂停 =====================
    std::printf("\nCUDA C++ demo succeeded.\n");
#ifdef _WIN32
    std::printf("按任意键退出...\n");
    _getch();                              // 无缓冲读一个键，不用按回车（替代 system("pause")，避免 GBK 乱码）
#else
    std::printf("Press Enter to exit...\n");
    std::getchar();                        // Linux/macOS：按回车退出
#endif
    return 0;
}
