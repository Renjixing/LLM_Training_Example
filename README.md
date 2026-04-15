# LLM_Training_Example

一个基于 **C4 流式数据集** 训练 **LLaMA ~19M 参数模型** 的最小示例，包含：

- 使用 `datasets` 的 streaming 模式读取 C4。
- 构建约 19M 参数量的 LLaMA 模型。
- 训练 token 总量按“模型参数量的 4 倍”自动计算。
- 统计训练过程中累计梯度数据量（元素总数与字节大小）。
- 使用 perplexity（困惑度）在验证集上评估模型性能。

## 环境准备

```bash
pip install torch transformers datasets sentencepiece
```

## 运行训练

```bash
python train_llama19m_c4_streaming.py \
  --seq_len 512 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --epochs 1 \
  --target_multiplier 4
```

## 关键输出

训练结束后会打印：

- 模型参数量
- 目标训练 token 总量（默认 4x 参数量）
- 实际训练消耗 token 总量
- 累计梯度元素总量
- 累计梯度数据总量（GB）
- 最终验证集 perplexity
