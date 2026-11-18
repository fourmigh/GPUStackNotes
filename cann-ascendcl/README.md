# CANN / AscendCL

> 华为昇腾 AI 处理器的软件栈，CANN 是整体架构，AscendCL 是其中的 C/C++ 编程接口。国产 AI 芯片里生态最完整的一套。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 华为（昇腾计算） |
| 首次公开 | 2018-10 昇腾 310/910 与 CANN 架构发布；CANN 2019 起部分开源 |
| 类型 | AI 加速器软件栈（驱动 + 编译器 + 算子库 + 图引擎） |
| 开闭源 · 许可 | 部分开源（部分算子与工具链），驱动与核心闭源 |
| 核心语言 / 接口 | AscendCL C/C++、TBE 算子开发（DSL/TIK）、MindSpore / PyTorch 适配 |
| 需要硬件 | 昇腾 310（推理）/ 310P / 910A / 910B（训练） |
| 当前状态 | 活跃，CANN 8.x，国内大模型部署重要选项 |

## 设计思路

CANN 是分层设计：底层驱动 + 运行时管理设备与流，中间是图引擎（GE）负责子图切分与算子融合，上层 AscendCL 提供类似 CUDA Runtime 的设备/内存/流/内核管理接口。

算子开发走 TBE（Tensor Boost Engine），可以用 DSL 快速写，也可以用 TIK 直接控制指令与片上缓冲。对用户而言，最常用的路径是 MindSpore 或 PyTorch（torch_npu）直接调用现成算子。

## 最小示例

```cpp
// AscendCL：申请设备内存并执行
aclInit(nullptr);
aclrtSetDevice(0);
void *devPtr = nullptr;
aclrtMalloc(&devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST);

acldvppMalloc(...);                    // 图像类数据走 DVPP
aclrtMemcpy(devPtr, size, hostPtr, size, ACL_MEMCPY_HOST_TO_DEVICE);

aclmdlExecute(modelId, inputDesc, outputDesc);   // 执行离线模型（om）
aclrtSynchronizeDevice();
aclrtFree(devPtr);
aclFinalize();
```

实际部署中更常见的是「ONNX → ATC 转 om → AscendCL 推理」的链路。

## 与其它方案的异同

- **vs CUDA**：都提供 Runtime 级 API，但 AscendCL 更偏「模型加载执行」，手写 kernel 的自由度低于 CUDA。
- **vs ROCm / HIP**：HIP 追求与 CUDA 源码同构；CANN 是独立设计，迁移需要重写算子。
- **vs TVM**：CANN 是封闭的完整栈，TVM 是开放编译器；部分国产芯片的编译器基于 TVM，CANN 则自研。
- **vs MindSpore**：MindSpore 是上层框架，CANN 是它（以及 torch_npu）的运行时底座。

## 优势

- 国产 AI 芯片中生态最完整：框架适配、算子库、迁移工具、文档都相对齐全。
- 910B 在国产训练卡中算力与集群方案（Atlas 800/900）成熟度较高。
- 与 MindSpore 深度协同，图优化与分布式训练有一体化设计。
- 供货相对稳定，国内大模型部署采用率高。

## 局限

- 与 CUDA 差异大，已有 CUDA 算子迁移需重写，不是「改头文件」级别的工作量。
- 开源程度有限，遇到问题深度排查困难。
- PyTorch 适配（torch_npu）算子覆盖不全，新模型常需等待适配。
- 工具链体验与 Nsight 级别仍有差距。
- 受出口管制影响，先进制程供应存在不确定性。

## 现状与生态

国内大模型训练与推理的主力国产方案之一，昇腾社区（Gitee / HiAI）有大量适配代码。vLLM-Ascend、MindIE 等推理框架已可用。典型部署链路：模型 → ATC 编译为 om → AscendCL/MindIE 推理。

## 参考

- 官方文档：<https://www.hiascend.com/cann>
- AscendCL 应用开发指南（昇腾社区）
