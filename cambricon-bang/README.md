# 寒武纪 BANG / MagicMind

> 寒武纪 MLU 加速器的软件栈，BANG 是自研编程语言族，MagicMind 是推理图编译器。

## 基本信息

| 项目 | 内容 |
|---|---|
| 主导方 | 寒武纪（Cambricon） |
| 首次公开 | 2019 发布 BANG 语言与 MLU270；MagicMind 2020 推出 |
| 类型 | AI 加速器软件栈（语言 + 编译器 + 运行时） |
| 开闭源 · 许可 | 部分开源（BANG C 语言族、部分工具），核心闭源 |
| 核心语言 / 接口 | BANG C / BANG Py、MagicMind C++ API、PyTorch 适配 |
| 需要硬件 | MLU 系列（MLU270 / 290 / 370 等） |
| 当前状态 | 商用中，推理为主 |

## 设计思路

寒武纪自研 MLU 指令集，因此不能直接复用 CUDA 或 OpenCL。BANG 是面向 MLU 的编程语言族：BANG C 提供类 C 的 kernel 写法，BANG Py 面向 Python 用户。

MagicMind 是推理专用图编译器：输入 ONNX / Caffe / TensorFlow 模型，输出针对 MLU 优化的执行图，负责算子融合、量化、内存复用。训练侧则依托 PyTorch 适配层。

## 最小示例

```c
// BANG C kernel 风格（示意）
__mlu_global__ void saxpy(int n, float a, float *x, float *y) {
    int i = taskId.x * 128 + coreId * 64 + laneId;
    if (i < n) y[i] = a * x[i] + y[i];
}
```

实际推理部署更多走 MagicMind：

```cpp
// 加载编译好的模型并推理
auto engine = magicmind::CreateIEngine();
auto model = engine->CreateIModel();
model->DeserializeFromFile("model.cambricon");
auto ctx = model->CreateIContext();
ctx->Enqueue(input_tensors, output_tensors, queue);
```

## 与其它方案的异同

- **vs CUDA**：编程模型完全不同（任务/核/ lane 三层），无法源码级迁移。
- **vs CANN**：定位相似（专用 AI 加速器 + 自研栈），CANN 生态更大、框架适配更全。
- **vs TVM**：TVM 是通用编译器，寒武纪曾参与 TVM 生态但主栈仍是自研。
- **vs BIRENSUPA / MXMACA**：壁仞与沐曦走通用 GPU 路线、CUDA 风格；寒武纪是专用架构，可编程性更低但能效比路线不同。

## 优势

- 自研指令集与架构，推理场景能效比有优势。
- MagicMind 的量化与图优化针对 MLU 深度定制，常见 CV/NLP 模型效果好。
- 国产化程度高，供应链自主。
- 在视频分析与边缘推理场景有成熟落地。

## 局限

- 编程模型独特，生态薄，开发者学习成本高。
- 第三方库与框架支持远不如 CUDA，新模型适配慢。
- BANG 工具链与调试体验与主流差距大。
- 训练场景竞争力弱，主要定位推理。
- 公开资料与社区问答稀少。

## 现状与生态

主要面向安防、视频分析、边缘推理等场景，在国产化项目中作为推理卡使用。对开发者而言，除非项目明确要求 MLU，否则优先选择生态更成熟的方案。

## 参考

- 官方网站：<https://www.cambricon.com/>
- MagicMind 用户手册（寒武纪开发者社区）
