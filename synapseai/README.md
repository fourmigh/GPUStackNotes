# Habana SynapseAI

> Intel 收购 Habana 后形成的 AI 加速器软件栈，Gaudi 系列芯片靠它把 PyTorch 图直接编译成硬件指令。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | Habana Labs（2019 被 Intel 收购） |
| 首次公开 | 2019（Goya 推理卡、Gaudi 训练卡发布） |
| 类型 | AI 加速器软件栈（驱动 + 图编译器 + 库） |
| 开闭源 · 许可 | 部分开源（PyTorch 桥接、图编译器组件），驱动闭源 |
| 核心语言 / 接口 | PyTorch（`torch.compile` / habana 后端）、自定义 TPC 算子（C++） |
| 需要硬件 | Habana Gaudi / Gaudi2 / Gaudi3、Goya |
| 当前状态 | 活跃（Gaudi3 已发布），但用户基数小 |

## 设计思路

SynapseAI 的思路是「**不改模型代码**」：用户沿用 PyTorch，通过 `torch.compile` 的 habana 后端把图下沉到 Habana 图编译器，再生成 Gaudi 的指令。算子以 TPC（Tensor Processing Core）为单位，可用 C++ 自定义。

硬件侧集成了 24 个 100GbE RoCE 端口，直接做节点间通信而不需要额外网卡——这是它在大模型训练集群里对标 NVIDIA + InfiniBand 的主要卖点。

## 最小示例

```python
import torch
import habana_frameworks.torch.core as htcore

model = MyModel().to("hpu")            # HPU = Habana Processing Unit
x = torch.randn(32, 3, 224, 224).to("hpu")

with torch.no_grad():
    y = model(x)
htcore.mark_step()                     # 触发图编译与执行
```

启动脚本时设置 `PT_HPU_LAZY_MODE=1` 等环境变量控制图模式。

## 与其它方案的异同

- **vs CUDA**：SynapseAI 只需 PyTorch 前端，用户无需写 kernel；但算子覆盖与社区支持远不及 CUDA。
- **vs CANN / AscendCL**：两者都是「专用 AI 芯片 + 自家编译器」路线，CANN 生态更大、文档更多。
- **vs ROCm / HIP**：HIP 面向通用 GPU 编程，SynapseAI 是纯 AI 加速器栈，不追求通用计算。
- **vs Trainium / Inferentia（AWS）**：思路接近，都是云厂商自研芯片 + 框架桥接。

## 优势

- 训练性价比高，Gaudi2/3 在部分 LLM 训练基准上与 A100/H100 有竞争力。
- 板载 RoCE 网络，集群组网成本低。
- PyTorch 前端零改动，迁移门槛不高。
- 支持 FP8 等低精度格式，适合大模型。

## 局限

- 只支持 PyTorch，TensorFlow / JAX 生态缺失。
- 算子覆盖不全，遇到不支持的算子要手写 TPC。
- 只服务 Habana 硬件，无法跨平台复用。
- 社区极小，遇到问题基本只能靠 Intel 官方支持。
- 供应与云上可获得性有限（主要 AWS、Intel Developer Cloud）。

## 现状与生态

主要部署在 AWS（dl1 / dl2 实例）与 Intel 自家云。在大模型训练成本敏感的场景有实际用户，但相对于 NVIDIA 生态仍是小众。Intel 已将 Habana 与 GPU 业务整合，未来路线存在不确定性。

## 参考

- 官方文档：<https://docs.habana.ai/>
- Gaudi2 / Gaudi3 白皮书
