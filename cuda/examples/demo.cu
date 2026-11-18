// ============================================================================
// demo.cu —— CUDA C++ 入门示例
//
// 演示三个最核心的用法：
//   1. 查询 GPU 设备信息（型号、计算能力、SM 数量、显存大小）
//   2. 向量加法（一维线程网格）
//   3. 矩阵乘法（二维线程网格）
//
// 同时串起 CUDA 编程的完整流程：
//   host 分配内存 -> 设备分配显存 -> 数据拷进设备 -> 启动 kernel 计算
//   -> 同步等待 -> 结果拷回 host -> 校验
//
// 名词约定：
//   host   = CPU + 主机内存（普通的 C/C++ 代码跑在 host）
//   device = GPU + 显存（__global__ 函数跑在 device）
//   kernel = 在 GPU 上执行的函数，由 host 用 <<<...>>> 语法启动
// ============================================================================

#include <cstdio>          // C 标准输入输出库：printf（打印到屏幕）、fprintf（打印到 stderr）
#include <cstdlib>         // C 标准库：malloc / free（分配释放主机内存）、exit（退出）、system（调用外部命令）
#include <cmath>           // C 数学库：fabs（浮点取绝对值，用于结果校验）
#include <cuda_runtime.h>  // CUDA 运行时 API：cudaMalloc / cudaMemcpy / cudaFree / kernel 启动等

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
// main 跑在 host（CPU）上，负责「调度」GPU：
// 分配显存、拷贝数据、启动 kernel、同步、取回结果、校验。
// ---------------------------------------------------------------------------
int main() {
    // ===================== 1. 查询设备信息 =====================
    int dev = 0;                 // 设备编号：本机只有一张 NVIDIA 卡，就是 0 号
    cudaDeviceProp prop;         // cudaDeviceProp 结构体，用来装设备的各种属性
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));  // 把 0 号设备的属性读进 prop

    std::printf("=== Device ===\n");
    std::printf("Name:      %s\n", prop.name);                 // 显卡型号
    std::printf("CC:        %d.%d\n", prop.major, prop.minor); // 计算能力（Compute Capability），决定可用特性
    std::printf("SMs:       %d\n", prop.multiProcessorCount);  // SM（流多处理器）数量，越多并行能力越强
    std::printf("Memory:    %.0f MiB\n", prop.totalGlobalMem / 1024.0 / 1024.0); // 显存总量（字节换算成 MiB）

    // ===================== 2. 向量加法 =====================
    std::printf("\n=== Vector add (N=10000000) ===\n");
    const int N = 10000000;              // 向量长度：一千万个元素
    size_t bytes = N * sizeof(float);    // 需要的字节数 = 元素个数 * 每个 float 占 4 字节
                                         // size_t 是无符号整数类型，专门用来表示「字节大小」这类非负数

    // 在 host（CPU 内存）分配三段数组，h_ 前缀表示 host 侧
    float *h_a = (float*)std::malloc(bytes);   // 加数 a；malloc 返回 void*，需强转成 float*
    float *h_b = (float*)std::malloc(bytes);   // 加数 b
    float *h_out = (float*)std::malloc(bytes); // 结果，用来接收 GPU 算完拷回的数据
    // 初始化：a 全填 1.0、b 全填 2.0，因此正确结果恒为 3.0，便于校验
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    // 在 device（显存）上分配对应的三段，d_ 前缀表示 device 侧
    // 这三个指针指向的是「显存地址」，host 端不能直接解引用读写
    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes)); // 在显存里给 a 分配空间
                                         // 传 &d_a（指针的地址），让 cudaMalloc 把分到的显存地址写进 d_a
    CUDA_CHECK(cudaMalloc(&d_b, bytes)); // 给 b 分配
    CUDA_CHECK(cudaMalloc(&d_out, bytes)); // 给结果分配

    // 把 host 的数据拷贝到 device：
    // cudaMemcpyHostToDevice 是枚举常量，表示拷贝方向 host -> device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 启动 kernel：<<<块数, 每块线程数>>> 是 CUDA 的启动配置语法
    int threads = 256;                        // 每个线程块放 256 个线程
                                              // 256 是 32 的整数倍：GPU 以 32 线程为一个 warp（线程束）调度，凑整便于充分利用硬件
    int blocks = (N + threads - 1) / threads; // 需要的块数 = 向上取整(N / 256)
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_out, N);  // GPU 从这里开始并行计算

    CUDA_CHECK(cudaGetLastError());      // 检查 kernel 启动这一步有没有报错
    CUDA_CHECK(cudaDeviceSynchronize()); // 等 GPU 把活干完——kernel 是异步的，不调这句 host 会直接往下跑
                                         //（默认流上的操作会按顺序排队，但相对 host 而言是异步的）
    // 把结果从 device 拷回 host：cudaMemcpyDeviceToHost 表示方向 device -> host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // 在 host 上校验：每个元素都应等于 3.0
    bool ok1 = true;
    for (int i = 0; i < N; ++i) {
        // 与 3.0 的误差超过 1e-5 就判定失败。
        // 不用 == 精确比较：GPU 浮点运算有舍入误差，应留一个容差（tolerance）
        if (std::fabs(h_out[i] - 3.0f) > 1e-5f) { ok1 = false; break; }
    }
    std::printf("Result correct: %s  (out[0]=%.1f)\n", ok1 ? "YES" : "NO", h_out[0]);

    // ===================== 3. 矩阵乘法 =====================
    std::printf("\n=== Matrix mul (512x512) ===\n");
    const int M = 512;                              // 512 x 512 的方阵
    size_t mbytes = (size_t)M * M * sizeof(float);  // 矩阵的总字节数

    float *h_x = (float*)std::malloc(mbytes); // 左矩阵
    float *h_y = (float*)std::malloc(mbytes); // 右矩阵
    float *h_z = (float*)std::malloc(mbytes); // 结果矩阵
    // 用有规律的伪随机数填矩阵，方便后面和 CPU 重算结果对比
    for (int i = 0; i < M * M; ++i) {
        h_x[i] = (i % 7) * 0.1f;
        h_y[i] = ((i * 3) % 5) * 0.2f;
    }

    float *d_x, *d_y, *d_z;                  // device 侧的三段显存
    CUDA_CHECK(cudaMalloc(&d_x, mbytes));
    CUDA_CHECK(cudaMalloc(&d_y, mbytes));
    CUDA_CHECK(cudaMalloc(&d_z, mbytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, mbytes, cudaMemcpyHostToDevice));

    // 这次用二维启动配置：
    dim3 block(16, 16);        // dim3 是三维向量类型；这里每个线程块 16x16 = 256 个线程（同样凑满 warp 的整数倍）
    dim3 grid(M / 16, M / 16); // 网格也是二维：每维 512/16 = 32 个块，共 32x32 个块
    matMul<<<grid, block>>>(d_x, d_y, d_z, M); // 启动二维 kernel

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_z, d_z, mbytes, cudaMemcpyDeviceToHost));

    // 在 host 上用最朴素的三重循环重算一遍，和 GPU 结果逐项比较
    float maxErr = 0.0f;                     // 记录最大误差
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            float ref = 0.0f;                // 用 CPU 算出的「参考答案」
            for (int k = 0; k < M; ++k) ref += h_x[r * M + k] * h_y[k * M + c];
            float diff = std::fabs(h_z[r * M + c] - ref); // GPU 结果与参考答案之差
            if (diff > maxErr) maxErr = diff;
        }
    }
    std::printf("Max error: %g\n", maxErr);
    std::printf("Matrix mul correct: %s\n", maxErr < 1e-2f ? "YES" : "NO");

    // ===================== 4. 释放资源 =====================
    // 显存和主机内存都要手动释放，养成好习惯
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    std::free(h_a); std::free(h_b); std::free(h_out);
    std::free(h_x); std::free(h_y); std::free(h_z);

    // ===================== 5. 结束前暂停 =====================
    std::printf("\nCUDA C++ demo succeeded.\n");
#ifdef _WIN32
    std::system("pause");                 // Windows：按任意键退出，避免双击运行时窗口一闪而过
#else
    std::printf("Press Enter to exit...\n");
    std::getchar();                       // Linux/macOS：按回车退出
#endif
    return 0;
}
