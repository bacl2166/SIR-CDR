# Amazon Sports-to-Clothing Preprocessing Guide

本文档定义 SIR-CDR 首轮正式实验 `Amazon Sports and Outdoors -> Clothing, Shoes and Jewelry` 的数据下载、清洗、切分与文本向量化协议。

## 1. 实验范围

- 数据版本：Amazon Reviews 2014 5-core。
- 源域：`Sports_and_Outdoors`。
- 目标域：`Clothing_Shoes_and_Jewelry`。
- 用户协议：仅保留同时出现在两个领域的共享 `reviewerID`。
- 交互阈值：每个保留用户在两个领域中均至少有 5 次交互。
- 推荐方向：Sports -> Clothing。
- 物品特征：仅使用商品文本信息。
- 原始商品 ID：只用于索引、监督标签和结果反查，不参与特征编码。
- 文本向量：阿里云百炼 `text-embedding-v4`，输出 768 维。
- 评测：全目标域评测，不使用随机负采样。

## 2. 与 GenCDR 的关系

数据来源、原始字段、领域内时序排序、元数据文本化、Embedding 缓存和 Semantic ID 两阶段训练借鉴 GenCDR。

SIR-CDR 的模型假设优先于 GenCDR。GenCDR 在联合数据中为用户添加领域前缀，从而允许两域用户不重叠；SIR-CDR 通过同一用户的源域和目标域历史执行双重结构注入与隐式推理，因此本实验按原始 `reviewerID` 求交集，不采用领域前缀用户命名空间。

参考实现：

- GenCDR: https://github.com/hupeiyu21/GenCDR
- Amazon processor: https://github.com/hupeiyu21/GenCDR/blob/main/dataloader/amazon_data_processor.py
- Joint dataset builder: https://github.com/hupeiyu21/GenCDR/blob/main/dataloader/create_joint_dataset.py
- Dataset splitting: https://github.com/hupeiyu21/GenCDR/blob/main/dataloader/dataset.py

## 3. 目录布局

```text
/root/autodl-tmp/SIR-CDR/
|-- configs/
|-- data/
|   |-- raw/amazon2014/
|   `-- processed/sports_to_clothing/
|-- artifacts/
|   |-- embeddings/
|   `-- tokenizer/
|-- checkpoints/sports_to_clothing/
|-- outputs/sports_to_clothing/
`-- experiments/
```

`data/`、`artifacts/`、`checkpoints/` 和 `outputs/` 已由 `.gitignore` 排除，不应提交到 GitHub。

## 4. 下载原始数据

```bash
cd /root/autodl-tmp/SIR-CDR
mkdir -p data/raw/amazon2014
cd data/raw/amazon2014

wget -c --show-progress \
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz

wget -c --show-progress \
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Sports_and_Outdoors.json.gz

wget -c --show-progress \
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz

wget -c --show-progress \
  https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Clothing_Shoes_and_Jewelry.json.gz
```

`wget -c` 支持中断后继续下载。

## 5. 校验下载文件

```bash
for file in \
  reviews_Sports_and_Outdoors_5.json.gz \
  meta_Sports_and_Outdoors.json.gz \
  reviews_Clothing_Shoes_and_Jewelry_5.json.gz \
  meta_Clothing_Shoes_and_Jewelry.json.gz
do
  test -s "$file" || {
    echo "Missing or empty file: $file"
    exit 1
  }
  gzip -t "$file" || exit 1
  echo "Valid: $file"
done
```

记录原始记录数和本地哈希：

```bash
for file in *.json.gz
do
  printf "%s: " "$file"
  gzip -cd "$file" | wc -l
done | tee raw_record_counts.txt

sha256sum *.json.gz > SHA256SUMS.local
```

## 6. 原始字段

评论文件至少使用：

| 字段 | 用途 |
| --- | --- |
| `reviewerID` | 用户标识及两域交集键 |
| `asin` | 原始商品标识 |
| `unixReviewTime` | 时序排序和防止未来信息泄漏 |
| `overall` | 可选统计字段；首轮实验将交互作为隐式反馈 |

首轮实验不使用 `reviewText` 构造商品静态表示，避免把用户评论信息泄漏到商品特征。

元数据文件使用 `title`、`brand`、`categories`、`feature` 和 `description`。缺失字段直接跳过。解析器应使用 JSON 解析或 `ast.literal_eval` 兼容旧格式，禁止使用不受信任的 `eval`。

## 7. 用户过滤与 ID 映射

预处理顺序：

1. 分别读取两个领域的 5-core 评论。
2. 按 `reviewerID` 聚合交互，并按 `(unixReviewTime, asin)` 稳定排序。
3. 求两个领域用户集合的交集。
4. 仅保留两个领域中均至少有 5 次交互的用户。
5. 为共享用户建立统一连续索引。
6. 为两个领域物品建立统一且不重叠的连续索引。
7. 保存正向和反向映射，以便推荐结果恢复为原始 ASIN。

不能将 ASIN 数字化后直接作为模型特征，也不能因为局部编号相同而把两个不同商品视为同一物品。

## 8. 无泄漏时序切分

对每个共享用户，以目标域 Clothing 序列定义验证和测试目标：

```text
target[-1]  -> test positive item
target[-2]  -> validation positive item
target[:-2] -> training target history
```

构造任一预测样本时，只允许使用正样本时间戳之前的历史：

```text
source_history = Sports events with timestamp < label timestamp
target_history = Clothing events before the label event
```

训练图只能由训练阶段可见交互构造。验证和测试正样本及其未来邻接关系不能进入图、码本监督或统计特征。

变长序列需要 padding mask。均值池化、注意力和结构注入必须忽略 padding，不能直接对补零位置或空序列求均值。

## 9. 商品文本模板

```text
Title: <title>.
Brand: <brand>.
Categories: <category path>.
Features: <feature list>.
Description: <description>.
```

文本处理要求：

- 解码 HTML 实体并移除 HTML 标签。
- 合并重复空白。
- 保留英文商品名称和必要标点。
- 限制极长字段，避免超过 API 输入上限。
- 不使用测试阶段用户评论生成商品静态文本。
- 记录文本模板版本，模板变化后重新生成 Embedding。

## 10. 千问 Embedding

首轮实验固定：

```text
provider: qwen
model: text-embedding-v4
dimension: 768
encoding_format: float
```

加载服务器环境变量：

```bash
source /root/autodl-tmp/sir-cdr-api.env
test -n "$DASHSCOPE_API_KEY" && echo "API key loaded"
test -n "$DASHSCOPE_BASE_URL" && echo "API base URL loaded"
```

不要将 API Key 写入配置、日志、命令历史或仓库。已经公开的 Key 必须在百炼控制台重置。

Embedding 阶段必须支持：

- 按物品批量请求并校验每个向量恰好为 768 维。
- 输出统一转换为 `float32`。
- 保存物品索引到向量行号的映射。
- 每批完成后持久化进度。
- 对超时、限流和临时服务错误执行有限次数重试。
- 已成功缓存的物品不能重复调用 API。

## 11. 两阶段训练

```text
Stage 1:
Qwen text embeddings
  -> domain-adaptive semantic tokenizer/codebook
  -> fixed Semantic IDs

Stage 2:
paired cross-domain histories + graphs + fixed Semantic IDs
  -> structural encoding
  -> gated implicit reasoning
  -> CPF
  -> autoregressive recommendation
```

原始商品 ID 不进入特征编码，只参与索引、监督标签、候选过滤和结果反查。

## 12. 全目标域评测

- 不采用随机负采样。
- 排除用户训练历史中已经出现的目标域物品。
- 验证或测试正样本必须保留在候选集合中。
- 使用领域约束前缀树与 Beam Search 生成候选。
- 报告 `HR@5/10/20`、`NDCG@5/10/20`、`MRR@5/10/20`。
- 同时报告 Coverage、Semantic ID collision rate 和 token lookup hit rate。

## 13. 流水线命令契约

正式流水线计划提供：

```bash
python experiments/prepare_amazon.py --config configs/amazon_sports_clothing.yaml
python experiments/embed_items.py --config configs/amazon_sports_clothing.yaml
python experiments/train_tokenizer.py --config configs/amazon_sports_clothing.yaml
python experiments/train.py --config configs/amazon_sports_clothing.yaml
python experiments/evaluate.py --config configs/amazon_sports_clothing.yaml
```

这些脚本仍处于待实现状态。在对应脚本提交前，不应在服务器执行这些命令。当前可执行入口为：

```bash
python -m unittest discover -s tests -v
python experiments/run_synthetic.py
```

## 14. 可复现性记录

每次正式运行必须保存：

- Git commit hash 和完整 YAML 配置快照。
- Python、PyTorch、CUDA 和 GPU 型号。
- 数据文件大小与 SHA-256。
- 原始及过滤后的用户、物品、交互数量。
- Embedding 模型、维度、API 地域和文本模板版本。
- 随机种子、训练日志、最佳 checkpoint 和最终指标。

```bash
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
nvidia-smi
```
