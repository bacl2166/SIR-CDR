# Amazon Sports-to-Clothing Preprocessing Guide

本文档定义 SIR-CDR 首轮正式实验 `Amazon Sports and Outdoors -> Clothing, Shoes and Jewelry` 的数据下载、清洗、切分与文本向量化协议。

> 当前状态：原始数据可按本文档手动下载和校验；正式预处理入口
> `experiments/prepare_amazon.py` 及其配套模块仍需按实现计划完成。文中标记为
> “实现后执行”的命令，在对应代码合入 GitHub 前不可运行。

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

### 3.1 标准配置文件

预处理实现完成后，`configs/amazon_sports_clothing.yaml` 应作为本实验唯一的默认配置入口：

```yaml
dataset:
  source_domain: Sports_and_Outdoors
  target_domain: Clothing_Shoes_and_Jewelry
  raw_dir: data/raw/amazon2014
  processed_dir: data/processed/sports_to_clothing
  min_interactions_per_domain: 5
  max_text_chars: 12000
  seed: 42
```

字段含义：

| 字段 | 含义 | 约束 |
| --- | --- | --- |
| `source_domain` | 源域类别 | 必须为 `Sports_and_Outdoors` |
| `target_domain` | 目标域类别 | 必须为 `Clothing_Shoes_and_Jewelry` |
| `raw_dir` | 四个 gzip 原始文件目录 | 相对路径以仓库根目录为基准 |
| `processed_dir` | 预处理产物目录 | 相对路径以仓库根目录为基准 |
| `min_interactions_per_domain` | 双域最低交互数 | 不小于 3，正式实验固定为 5 |
| `max_text_chars` | 单个商品文本最大字符数 | 必须为正整数 |
| `seed` | 确定性处理和后续实验种子 | 正式实验固定记录为 42 |

命令行参数只覆盖本次运行行为，领域、阈值和路径等实验定义由 YAML 管理，避免服务器命令与论文设置不一致。

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
3. 只为评论中实际出现的 ASIN 构造商品文本。
4. 删除缺少元数据或清洗后商品文本为空的交互。
5. 在删除无效交互后重新统计每个用户在两个领域中的交互数。
6. 求两个领域用户集合的交集，仅保留两个领域均至少有 5 次有效交互的用户。
7. 为共享用户建立确定性的统一连续索引。
8. 为两个领域物品建立从 `1` 开始、统一且不重叠的连续索引，键采用 `<domain>:<asin>`；物品索引 `0` 保留给 padding。
9. 保存正向和反向映射，以便推荐结果恢复为原始 ASIN。

映射必须在全部过滤完成后生成，保证不会出现没有样本的用户或物品。相同输入、配置和代码版本必须生成完全相同的映射顺序。

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

### 8.1 样本生成规则

每条推荐样本固定包含：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `user_id` | integer | 过滤后统一用户索引 |
| `source_items` | list[integer] | 正样本时间戳之前的源域物品序列 |
| `target_items` | list[integer] | 正样本之前的目标域物品序列 |
| `positive_target_item` | integer | 当前目标域监督标签 |
| `timestamp` | integer | 正样本的 Unix 时间戳 |

训练集对 `target[:-2]` 中每个满足条件的目标事件构造一条 next-item 样本；验证集和测试集分别只使用倒数第二、倒数第一条目标事件。源域历史使用严格条件 `source_timestamp < label_timestamp`，不能使用 `<=`。源域历史或目标域历史为空的样本应跳过，并在统计文件中记录跳过数量。

### 8.2 预处理产物契约

`data/processed/sports_to_clothing/` 必须一次性、原子地生成以下文件：

| 文件 | 格式 | 内容 |
| --- | --- | --- |
| `mappings.json` | JSON | 用户和物品双向映射、padding 约定、领域信息 |
| `item_texts.jsonl` | JSONL | 每行一个 `item_id`、`domain` 和 `text` |
| `train.jsonl` | JSONL | 训练推荐样本 |
| `validation.jsonl` | JSONL | 每个有效用户最多一条验证样本 |
| `test.jsonl` | JSONL | 每个有效用户最多一条测试样本 |
| `statistics.json` | JSON | 过滤前后用户、物品、交互和跳过样本数量 |
| `manifest.json` | JSON | 数据指纹、配置、代码产物版本和输出行数 |

JSONL 每行必须是一个独立 UTF-8 JSON 对象，不能使用 pickle。`item_texts.jsonl` 中的 `item_id` 只是 Embedding 缓存关联键，不得拼接进 `text`。推荐样本中的 ID 只承担索引和监督标签作用。

`manifest.json` 至少记录：

- `schema_version: 1`；
- 源域、目标域、阈值、最大文本长度和随机种子；
- 四个原始文件的名称、字节数和 SHA-256；
- 七个输出文件的记录数或对象计数；
- UTC 创建时间。

写入时先生成临时目录，全部文件可解析后再替换正式目录。已有 `manifest.json` 时默认拒绝覆盖，只有显式传入 `--force` 才允许重建，防止不同配置的结果被静默混合。

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

### 13.1 当前可执行

```bash
cd /root/autodl-tmp/SIR-CDR
conda activate sir-cdr
python -m unittest discover -s tests -v
python experiments/run_synthetic.py
```

### 13.2 预处理实现后执行

首次运行由程序下载缺失的原始文件并生成产物：

```bash
python experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml
```

四个 gzip 文件已人工下载并校验时：

```bash
python experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml \
  --skip-download
```

确认需要删除并重建同一输出目录时：

```bash
python experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml \
  --skip-download \
  --force
```

`--skip-download` 缺少任一原始文件时必须以非零状态退出，并一次列出全部缺失路径。下载采用同目录 `.part` 临时文件并支持续传；gzip 完整性校验成功后才能原子替换正式文件。

### 13.3 后续流水线

预处理验收通过后，后续阶段计划提供：

```bash
python experiments/embed_items.py --config configs/amazon_sports_clothing.yaml
python experiments/train_tokenizer.py --config configs/amazon_sports_clothing.yaml
python experiments/train.py --config configs/amazon_sports_clothing.yaml
python experiments/evaluate.py --config configs/amazon_sports_clothing.yaml
```

Embedding、Semantic ID、模型训练和评测脚本仍处于待实现状态。在对应脚本提交前，不应在服务器执行这些命令。预处理阶段不需要加载 `DASHSCOPE_API_KEY`。

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

## 15. 预处理验收清单

预处理实现并运行后，在服务器执行：

```bash
cd /root/autodl-tmp/SIR-CDR
conda activate sir-cdr

python -m unittest \
  tests.test_experiment_config \
  tests.test_amazon2014_io \
  tests.test_amazon_preprocessing \
  tests.test_prepare_amazon_cli -v

python -m unittest discover -s tests -v

find data/processed/sports_to_clothing \
  -maxdepth 1 -type f -printf '%f\n' | sort

python -m json.tool \
  data/processed/sports_to_clothing/statistics.json
python -m json.tool \
  data/processed/sports_to_clothing/manifest.json
wc -l data/processed/sports_to_clothing/*.jsonl
```

验收必须同时满足：

- 四个原始 gzip 文件均非空且通过 `gzip -t`；
- 全部单元测试通过；
- 七个预处理产物全部存在且 JSON/JSONL 可解析；
- 用户在过滤后的两个领域中均至少有 5 次有效交互；
- 物品索引不使用 `0`，且两个领域的物品索引互不冲突；
- 商品文本不包含 ASIN、`reviewText` 或用户标识；
- 验证和测试历史中不存在时间戳晚于或等于标签的源域事件；
- 重复执行未使用 `--force` 时不会覆盖已有结果；
- `git status --short` 不显示任何数据、Embedding 或密钥文件。

## 16. 常见问题与恢复

### 16.1 `prepare_amazon.py` 不存在

说明服务器代码仍停留在只有设计文档的提交。先检查并更新仓库：

```bash
git status --short --branch
git log -3 --oneline
git pull origin main
test -f experiments/prepare_amazon.py
```

最后一条仍失败时，不要自行创建空脚本，等待预处理实现提交合入。

### 16.2 gzip 下载不完整

```bash
cd /root/autodl-tmp/SIR-CDR/data/raw/amazon2014
gzip -t <文件名>.json.gz
```

校验失败时保留或删除对应 `.part` 后重新下载，不要把损坏文件交给解析器。程序自动下载模式只会在 gzip 校验成功后替换最终路径。

### 16.3 使用 `--skip-download` 报文件缺失

检查文件名必须与以下名称完全一致：

```text
reviews_Sports_and_Outdoors_5.json.gz
meta_Sports_and_Outdoors.json.gz
reviews_Clothing_Shoes_and_Jewelry_5.json.gz
meta_Clothing_Shoes_and_Jewelry.json.gz
```

### 16.4 输出目录已存在

先阅读旧的 `manifest.json` 和 `statistics.json`。只有确定旧结果可被替换时才使用 `--force`；正式实验中应保留与论文结果对应的 manifest、Git 提交哈希和配置快照。

### 16.5 API 或网络错误

预处理只需要下载公开数据，不调用千问或 DeepSeek。若日志出现 Embedding API 请求，说明执行了错误阶段，应停止运行并检查命令。任何已暴露的 Key 都必须在供应商控制台撤销后重新创建。
