#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err), __FILE__, __LINE__);          \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

__global__ void vecAdd(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void matMul(const float* a, const float* b, float* out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; ++k) acc += a[row * n + k] * b[k * n + col];
        out[row * n + col] = acc;
    }
}

int main() {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("=== Device ===\n");
    std::printf("Name:      %s\n", prop.name);
    std::printf("CC:        %d.%d\n", prop.major, prop.minor);
    std::printf("SMs:       %d\n", prop.multiProcessorCount);
    std::printf("Memory:    %.0f MiB\n", prop.totalGlobalMem / 1024.0 / 1024.0);

    std::printf("\n=== Vector add (N=10000000) ===\n");
    const int N = 10000000;
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)std::malloc(bytes);
    float *h_b = (float*)std::malloc(bytes);
    float *h_out = (float*)std::malloc(bytes);
    for (int i = 0; i < N; ++i) { h_a[i] = 1.0f; h_b[i] = 2.0f; }

    float *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok1 = true;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_out[i] - 3.0f) > 1e-5f) { ok1 = false; break; }
    }
    std::printf("Result correct: %s  (out[0]=%.1f)\n", ok1 ? "YES" : "NO", h_out[0]);

    std::printf("\n=== Matrix mul (512x512) ===\n");
    const int M = 512;
    size_t mbytes = (size_t)M * M * sizeof(float);
    float *h_x = (float*)std::malloc(mbytes);
    float *h_y = (float*)std::malloc(mbytes);
    float *h_z = (float*)std::malloc(mbytes);
    for (int i = 0; i < M * M; ++i) {
        h_x[i] = (i % 7) * 0.1f;
        h_y[i] = ((i * 3) % 5) * 0.2f;
    }
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, mbytes));
    CUDA_CHECK(cudaMalloc(&d_y, mbytes));
    CUDA_CHECK(cudaMalloc(&d_z, mbytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, mbytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(M / 16, M / 16);
    matMul<<<grid, block>>>(d_x, d_y, d_z, M);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_z, d_z, mbytes, cudaMemcpyDeviceToHost));

    float maxErr = 0.0f;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            float ref = 0.0f;
            for (int k = 0; k < M; ++k) ref += h_x[r * M + k] * h_y[k * M + c];
            float diff = std::fabs(h_z[r * M + c] - ref);
            if (diff > maxErr) maxErr = diff;
        }
    }
    std::printf("Max error: %g\n", maxErr);
    std::printf("Matrix mul correct: %s\n", maxErr < 1e-2f ? "YES" : "NO");

    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y)); CUDA_CHECK(cudaFree(d_z));
    std::free(h_a); std::free(h_b); std::free(h_out);
    std::free(h_x); std::free(h_y); std::free(h_z);

    std::printf("\nCUDA C++ demo succeeded.\n");
#ifdef _WIN32
    std::system("pause");                 // Windows: 按任意键退出，避免双击运行闪退
#else
    std::printf("Press Enter to exit...\n");
    std::getchar();                       // POSIX: 回车退出
#endif
    return 0;
}
