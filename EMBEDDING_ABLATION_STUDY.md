# Embedding 消融实验使用说明

本文档说明如何使用不同的embedding策略进行消融实验，比较TDD-TCNN在以下四种embedding策略下的性能：
1. **CBOW** (Continuous Bag of Words)
2. **Skip-gram**
3. **FastText**
4. **CodeBERT**

## 目录

1. [训练Embedding](#1-训练embedding)
2. [运行消融实验](#2-运行消融实验)
3. [参数说明](#3-参数说明)
4. [输出文件](#4-输出文件)
5. [完整示例](#5-完整示例)

---

## 1. 训练Embedding

### 1.1 训练CBOW、Skip-gram和FastText

使用 `train_embeddings.py` 脚本训练embedding：

```bash
python train_embeddings.py \
    --train_file dataset/Ant-DFS/data/train.txt \
    --output_dir embeddings/ \
    --embedding_type all \
    --size 300 \
    --window 5 \
    --negative 5 \
    --min_count 1
```

**参数说明：**
- `--train_file`: 训练文件路径（用于构建词表）
- `--output_dir`: 输出目录
- `--embedding_type`: 要训练的embedding类型
  - `cbow`: 只训练CBOW
  - `skipgram`: 只训练Skip-gram
  - `fasttext`: 只训练FastText
  - `all`: 训练所有三种（推荐）
- `--size`: 词向量维度（默认300）
- `--window`: 窗口大小（默认5）
- `--negative`: 负采样数量（默认5）
- `--min_count`: 最小词频（默认1）

**输出文件：**
训练完成后，会在输出目录生成以下文件：
- `cbow_embedding.npy`: CBOW embedding矩阵
- `cbow_embedding_vocab.pkl`: CBOW词表
- `cbow_embedding_word2idx.pkl`: CBOW词到索引映射
- `skipgram_embedding.npy`: Skip-gram embedding矩阵
- `skipgram_embedding_vocab.pkl`: Skip-gram词表
- `fasttext_embedding.npy`: FastText embedding矩阵
- `fasttext_embedding_vocab.pkl`: FastText词表

### 1.2 为每个项目训练Embedding（可选）

如果需要为每个项目单独训练embedding：

```bash
# 为Ant项目训练
python train_embeddings.py \
    --train_file dataset/Ant-DFS/data/train.txt \
    --output_dir embeddings/Ant/ \
    --embedding_type all

# 为其他项目训练（类似）
python train_embeddings.py \
    --train_file dataset/ArgoUML-DFS/data/train.txt \
    --output_dir embeddings/ArgoUML/ \
    --embedding_type all
```

---

## 2. 运行消融实验

### 2.1 使用Random Embedding（基线）

```bash
python run_cmd.py \
    --model CNNTransformer-Seq \
    --dataset DFS \
    --embedding_strategy random \
    --device 0
```

### 2.2 使用CBOW Embedding

```bash
python run_cmd.py \
    --model CNNTransformer-Seq \
    --dataset DFS \
    --embedding_strategy cbow \
    --embedding_path embeddings/cbow_embedding.npy \
    --device 0
```

### 2.3 使用Skip-gram Embedding

```bash
python run_cmd.py \
    --model CNNTransformer-Seq \
    --dataset DFS \
    --embedding_strategy skipgram \
    --embedding_path embeddings/skipgram_embedding.npy \
    --device 0
```

### 2.4 使用FastText Embedding

```bash
python run_cmd.py \
    --model CNNTransformer-Seq \
    --dataset DFS \
    --embedding_strategy fasttext \
    --embedding_path embeddings/fasttext_embedding.npy \
    --device 0
```

### 2.5 使用CodeBERT Embedding

```bash
python run_cmd.py \
    --model CNNTransformer-Seq \
    --dataset DFS \
    --embedding_strategy codebert \
    --device 0
```

**注意：** CodeBERT需要特殊的模型架构，当前实现中会使用random embedding作为占位符。完整的CodeBERT集成需要修改模型架构。

---

## 3. 参数说明

### run_cmd.py 新增参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--embedding_strategy` | str | Embedding策略：random, cbow, skipgram, fasttext, codebert | random |
| `--embedding_path` | str | 预训练embedding文件路径（.npy格式） | None |

### train_embeddings.py 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--train_file` | str | 训练文件路径（必需） | - |
| `--output_dir` | str | 输出目录（必需） | - |
| `--embedding_type` | str | 训练类型：cbow, skipgram, fasttext, all | all |
| `--size` | int | 词向量维度 | 300 |
| `--window` | int | 窗口大小 | 5 |
| `--negative` | int | 负采样数量 | 5 |
| `--min_count` | int | 最小词频 | 1 |

---

## 4. 输出文件

### 4.1 训练Embedding输出

训练embedding后，每个embedding类型会生成：
- `{type}_embedding.npy`: Embedding矩阵（numpy格式）
- `{type}_embedding_vocab.pkl`: 词表（pickle格式）
- `{type}_embedding_word2idx.pkl`: 词到索引映射
- `{type}_embedding_idx2word.pkl`: 索引到词映射
- `{type}_embedding.model`: Gensim模型文件（可选）

### 4.2 实验输出

运行实验后，会生成以下CSV文件：

1. **分项目评估结果：**
   - `{model_name}-within-project-{embedding_strategy}-{timestamp}.csv`
   - 包含每个项目的precision, recall, f1指标

2. **跨项目评估结果：**
   - `{model_name}-cross-project-all-merged-{embedding_strategy}-{timestamp}.csv`
   - 包含合并所有项目后的整体指标

3. **LOO-CV评估结果：**
   - `{model_name}-loo-cv-{embedding_strategy}-{timestamp}.csv`
   - 包含每个项目作为unseen测试集时的指标

---

## 5. 完整示例

### 5.1 完整工作流程

#### 步骤1：合并所有项目的训练数据（用于训练embedding）

```bash
# 合并所有项目的训练文件
cat dataset/*/data/train.txt > merged_train.txt
```

#### 步骤2：训练所有embedding类型

```bash
python train_embeddings.py \
    --train_file merged_train.txt \
    --output_dir embeddings/ \
    --embedding_type all \
    --size 300 \
    --window 5 \
    --negative 5
```

#### 步骤3：运行消融实验

```bash
# 1. Random (基线)
python run_cmd.py --model CNNTransformer-Seq --dataset DFS \
    --embedding_strategy random --device 0

# 2. CBOW
python run_cmd.py --model CNNTransformer-Seq --dataset DFS \
    --embedding_strategy cbow \
    --embedding_path embeddings/cbow_embedding.npy \
    --device 0

# 3. Skip-gram
python run_cmd.py --model CNNTransformer-Seq --dataset DFS \
    --embedding_strategy skipgram \
    --embedding_path embeddings/skipgram_embedding.npy \
    --device 0

# 4. FastText
python run_cmd.py --model CNNTransformer-Seq --dataset DFS \
    --embedding_strategy fasttext \
    --embedding_path embeddings/fasttext_embedding.npy \
    --device 0
```

### 5.2 批量运行所有实验

创建一个脚本 `run_ablation_study.sh`：

```bash
#!/bin/bash

MODEL="CNNTransformer-Seq"
DATASET="DFS"
DEVICE="0"
EMBEDDING_DIR="embeddings/"

# Random baseline
echo "Running Random embedding..."
python run_cmd.py --model $MODEL --dataset $DATASET \
    --embedding_strategy random --device $DEVICE

# CBOW
echo "Running CBOW embedding..."
python run_cmd.py --model $MODEL --dataset $DATASET \
    --embedding_strategy cbow \
    --embedding_path ${EMBEDDING_DIR}cbow_embedding.npy \
    --device $DEVICE

# Skip-gram
echo "Running Skip-gram embedding..."
python run_cmd.py --model $MODEL --dataset $DATASET \
    --embedding_strategy skipgram \
    --embedding_path ${EMBEDDING_DIR}skipgram_embedding.npy \
    --device $DEVICE

# FastText
echo "Running FastText embedding..."
python run_cmd.py --model $MODEL --dataset $DATASET \
    --embedding_strategy fasttext \
    --embedding_path ${EMBEDDING_DIR}fasttext_embedding.npy \
    --device $DEVICE

echo "All experiments completed!"
```

运行：
```bash
chmod +x run_ablation_study.sh
./run_ablation_study.sh
```

### 5.3 只运行LOO-CV评估

如果只想运行LOO-CV评估（跳过其他评估）：

```bash
python run_cmd.py --model CNNTransformer-Seq --dataset DFS \
    --embedding_strategy cbow \
    --embedding_path embeddings/cbow_embedding.npy \
    --loo_cv_only \
    --device 0
```

---

## 6. CBOW参数设置

### 6.1 默认参数

当前实现的默认参数：
- **Dimension size**: 300
- **Window size**: 5
- **Negative sampling**: 5

### 6.2 自定义参数

如果需要修改这些参数，在训练embedding时指定：

```bash
python train_embeddings.py \
    --train_file merged_train.txt \
    --output_dir embeddings/ \
    --embedding_type cbow \
    --size 200 \        # 修改维度为200
    --window 10 \       # 修改窗口大小为10
    --negative 10       # 修改负采样为10
```

### 6.3 参数建议

根据经验，以下参数组合通常效果较好：

| 场景 | Size | Window | Negative |
|------|------|--------|----------|
| 小数据集 | 100-200 | 3-5 | 5 |
| 中等数据集 | 200-300 | 5-10 | 5-10 |
| 大数据集 | 300 | 10 | 10-20 |

---

## 7. 注意事项

1. **词表一致性**：确保训练embedding时使用的词表与实验时使用的词表一致
2. **文件路径**：确保embedding文件路径正确，否则会回退到random embedding
3. **内存需求**：训练embedding可能需要较大内存，特别是大数据集
4. **CodeBERT**：当前CodeBERT实现不完整，需要进一步开发
5. **跨项目评估**：使用合并的embedding时，确保embedding是在所有项目数据上训练的

---

## 8. 结果分析

实验完成后，可以比较不同embedding策略的结果：

1. **分项目评估**：查看每个项目在不同embedding下的表现
2. **跨项目评估**：查看整体性能
3. **LOO-CV评估**：查看模型在unseen项目上的泛化能力

建议使用表格或图表对比不同embedding策略的precision、recall和f1分数。

---

## 9. 故障排除

### 问题1：找不到embedding文件

**错误信息：** `警告: xxx.npy 不存在，使用random embedding`

**解决方案：**
- 检查embedding文件路径是否正确
- 确认文件已成功生成
- 检查文件权限

### 问题2：词表不匹配

**错误信息：** 运行时出现词汇表错误

**解决方案：**
- 确保使用相同的训练数据训练embedding
- 检查词表文件是否正确加载

### 问题3：内存不足

**解决方案：**
- 减小embedding维度（--size）
- 减小窗口大小（--window）
- 使用更少的负采样（--negative）

---

## 10. 引用

如果在论文中使用这些结果，请引用相关的embedding方法：
- CBOW/Skip-gram: Mikolov et al., "Efficient Estimation of Word Representations in Vector Space", 2013
- FastText: Bojanowski et al., "Enriching Word Vectors with Subword Information", 2017
- CodeBERT: Feng et al., "CodeBERT: A Pre-Trained Model for Programming and Natural Languages", 2020
