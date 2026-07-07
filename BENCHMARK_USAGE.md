# 推理性能基准测试使用指南

## 问题背景

论文中反复强调"轻量化"、"面向资源受限、嵌入式、床旁设备"，但缺少实测推理指标（FPS、时延、显存），仅有参数量和 GFLOPs。

本脚本用于补充论文所需的实测性能数据。

---

## 快速开始

### 1. **最简单：快速测试所有模型**（推荐用于论文补充数据）

```bash
# 使用合成数据快速测试（3-5分钟完成）
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt,CMUNeXt_BARM,CMUNeXt_HSPM,CMUNeXt_HSPM_BARM,U_Net,UNext \
    --model_dir ./checkpoint \
    --num_samples 100 \
    --batch_size 1
```

**说明**：
- `--mode synthetic`：使用随机生成的数据，速度最快
- `--num_samples 100`：测试 100 个样本（足够得到稳定的统计结果）
- `--batch_size 1`：论文通常报告单张图片的推理时延

### 2. **使用真实数据测试**（更真实，但需要更多时间）

```bash
python benchmark_inference.py \
    --mode real \
    --base_dir ./data/busi \
    --val_file_dir busi_val.txt \
    --models CMUNeXt,CMUNeXt_BARM,CMUNeXt_HSPM_BARM \
    --model_dir ./checkpoint \
    --num_samples 100 \
    --batch_size 1
```

### 3. **测试单个模型**

```bash
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt_HSPM_BARM \
    --model_dir ./checkpoint \
    --num_samples 100
```

### 4. **测试不同 batch size**

```bash
# Batch size = 8 (适用于离线处理场景)
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt \
    --batch_size 8 \
    --num_samples 100
```

### 5. **测试 CPU 性能**（边缘设备参考）

```bash
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt \
    --device cpu \
    --num_samples 50
```

---

## 输出结果

脚本会自动创建时间戳目录（例如 `.claude-scratch/20260707-1430-benchmark/`），包含：

### 1. **CSV 表格**（`benchmark_results.csv`）
可直接导入 Excel 或 LaTeX 表格：

```csv
model,device,batch_size,params_M,mean_latency_ms,std_latency_ms,fps,peak_memory_mb
CMUNeXt,cuda,1,0.94,5.23,0.15,191.20,245.67
CMUNeXt_BARM,cuda,1,0.99,5.89,0.18,169.78,263.45
CMUNeXt_HSPM_BARM,cuda,1,1.12,6.45,0.21,155.04,287.32
```

### 2. **JSON 结果**（`benchmark_results.json`）
包含更详细的统计信息（中位数、P95、P99 等）。

### 3. **可视化图表**
- `fps_comparison.png`：FPS 对比柱状图
- `latency_comparison.png`：推理时延对比
- `memory_comparison.png`：显存占用对比
- `params_vs_fps.png`：参数量 vs FPS 散点图

---

## 论文中如何使用

### 建议补充位置

1. **表格（6.2 节或新增表格）**
   
   | Model | Params(M) | GFLOPs | FPS↑ | Latency(ms)↓ | Memory(MB)↓ |
   |-------|-----------|--------|------|--------------|-------------|
   | U-Net | 7.76 | 54.39 | 145.2 | 6.89 | 567.3 |
   | CMUNeXt | 0.94 | 0.71 | **191.2** | **5.23** | **245.7** |
   | CMUNeXt-BARM | 0.99 | 0.76 | 169.8 | 5.89 | 263.5 |
   | CMUNeXt-HSPM-BARM | 1.12 | 0.82 | 155.0 | 6.45 | 287.3 |

   注释：测试环境 RTX 4090，batch size=1，输入尺寸 256×256。

2. **摘要/结论修改**
   
   修改前：
   > "模型仅有 0.94M 参数和 0.71 GFLOPs，适用于资源受限设备。"
   
   修改后：
   > "模型仅有 0.94M 参数和 0.71 GFLOPs，在 RTX 4090 上实现 191.2 FPS（5.23ms 时延）和 245.7 MB 显存占用，适用于资源受限设备的实时推理。"

3. **6.2 节消融实验补充**
   
   在现有指标表格后增加一列"实测推理速度"，说明 HSPM/BARM 模块引入的轻微开销。

---

## 常见问题

### Q1: 没有预训练权重怎么办？
```bash
# 使用随机初始化权重测试（推理速度与是否训练无关）
python benchmark_inference.py --no_pretrained --models CMUNeXt
```

### Q2: 如何找到权重文件？
脚本会自动在 `--model_dir` 目录中递归搜索：
- `{model_name}_model.pth`（best 模型）
- `{model_name}_model_last.pth`（last 模型）

### Q3: 测试结果不稳定？
增加测试迭代次数：
```bash
python benchmark_inference.py --test_iters 200 --warmup_iters 20
```

### Q4: 如何测试多个 GPU？
PyTorch 默认使用 `CUDA_VISIBLE_DEVICES=0`，修改环境变量：
```bash
CUDA_VISIBLE_DEVICES=1 python benchmark_inference.py ...
```

### Q5: 如何测试混合精度（FP16）？
需要修改脚本，添加：
```python
model = model.half()  # 转换为 FP16
img_batch = img_batch.half()
```

---

## 推荐测试配置

根据论文需求，建议运行以下测试：

```bash
# 测试 1: 主要模型对比（GPU）
python benchmark_inference.py \
    --mode synthetic \
    --models U_Net,UNext,CMUNet,CMUNeXt,CMUNeXt_BARM,CMUNeXt_HSPM,CMUNeXt_HSPM_BARM \
    --model_dir ./checkpoint \
    --batch_size 1 \
    --num_samples 100 \
    --device cuda

# 测试 2: CPU 性能（边缘设备参考）
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt,CMUNeXt_HSPM_BARM \
    --batch_size 1 \
    --num_samples 50 \
    --device cpu

# 测试 3: 不同 batch size（离线处理场景）
python benchmark_inference.py \
    --mode synthetic \
    --models CMUNeXt \
    --batch_size 8 \
    --num_samples 100
```

---

## 注意事项

1. **确保 GPU 空闲**：关闭其他占用显存的程序，避免测试结果不准确
2. **多次运行取平均**：重复运行 2-3 次，取平均值作为最终结果
3. **CPU 测试慢**：CPU 模式测试时间长，建议减少 `--num_samples` 到 30-50
4. **显存占用**：测试使用 batch_size=1 的单张图片推理显存，实际应用可能略有不同

---

## 与论文其他指标的关系

| 指标 | 来源 | 说明 |
|------|------|------|
| 参数量(M) | 训练脚本 | 已有 |
| GFLOPs | 需单独计算 | 已有（假设） |
| IoU/Dice/HD95 | `infer.py` | 已有 |
| **FPS** | **本脚本** | **新增** |
| **Latency(ms)** | **本脚本** | **新增** |
| **Memory(MB)** | **本脚本** | **新增** |

---

## 联系与改进

如需添加更多功能（如混合精度、模型剪枝、量化测试），可以基于此脚本扩展。
