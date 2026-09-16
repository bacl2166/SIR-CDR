# SIR-CDR 完整代码说明与实验运行手册

本文档对应仓库当前版本，面向第一次接触项目、需要在远程 Linux GPU
服务器上复现实验的研究者。文档按“环境 -> 数据 -> 文本向量 -> Semantic ID ->
模型 -> 训练 -> 推理 -> 结果”的真实调用顺序组织，并在后半部分逐文件解释代码。

“逐代码说明”采用语义行组：连续的 import、dataclass 字段、参数校验和同一个张量公式
作为一个代码段解释；空行、右括号和 `if __name__ == "__main__"` 等 Python 固定样板
不重复占段。所有会改变数据、张量、梯度、文件或控制流的有效语句均在对应文件小节中
说明，这比把每个括号机械翻译成中文更便于定位和维护。

> 当前主实验是 **text-only SIR-CDR v2**。`text_*` 文件构成新主线；
> `formal_*` 文件是已经得到基线结果的 v1；`framework.py` 是早期合成实验框架。
> 三者同时保留是为了复现与对照，但正式新实验应从
> `experiments/run_text_cdr.py` 进入。

## 1. 研究目标与实现边界

SIR-CDR 面向跨域推荐。以 Sports -> Clothing 为例，模型根据同一用户在 Sports
源域和 Clothing 目标域中的历史行为，预测下一件 Clothing 商品。

当前实现遵循以下约束：

1. 只使用商品文本，不读取图片或其他多模态特征。
2. 原始 ASIN 和内部整数 ID 只用于索引、监督标签、去除已交互商品和结果回查，
   不存在可学习的 item-ID embedding。
3. 商品语义来自本地 `Qwen/Qwen3-Embedding-8B`，通过 MRL 导出 768 维离线向量。
4. 训练使用完整目标域 softmax，不做随机负采样。
5. 验证与测试使用完整目标域评测，并屏蔽用户已经交互过的目标域商品。
6. 图只由训练集历史构造，不读取验证集、测试集或当前样本正标签。
7. 连续门控隐推理是本地 PyTorch 神经网络，不在训练时调用大语言模型。
8. 8B 模型只在离线文本向量生成阶段加载；向量生成后训练可完全断网运行。

## 2. 三套代码路径

| 路径 | 入口 | 用途 | 是否作为后续主实验 |
|---|---|---|---|
| text-only v2 | `experiments/run_text_cdr.py` | 图结构、门控隐推理、CPF、Semantic ID 生成与完整目录排序 | 是 |
| formal v1 | `experiments/train_recommender.py` | 已跑通的正式基线与结果复现 | 作为基线 |
| synthetic | `experiments/run_synthetic.py` | 小型随机数据前向传播与接口检查 | 否 |

`cdr_framework/modules.py` 中仍保留 `MultimodalSemanticEncoder`，是旧版兼容代码。 
text-only v2 不实例化它，因此图片向量、属性向量和原始 ID 向量不会进入新模型。

## 3. 环境搭建

### 3.1 推荐硬件与软件

- Linux x86_64；
- Python 3.10；
- NVIDIA RTX 3090 或 4090；
- 当前远程机器已验证：RTX 3090 24 GB、PyTorch 2.6.0+cu124；
- 驱动显示的 CUDA 13.0 是驱动最高兼容版本，PyTorch 自带 CUDA 12.4 runtime，
  二者并不冲突。

### 3.2 创建 Conda 环境

```bash
cd /root/autodl-tmp/SIR-CDR

# 若 conda activate 报 conda init，可先加载 conda shell 脚本。
source /root/miniconda3/etc/profile.d/conda.sh

conda create -n sir-cdr python=3.10 -y
conda activate sir-cdr
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 `requirements.txt` 每一组依赖的作用

- `--extra-index-url .../cu124`：让 pip 能找到 CUDA 12.4 的 PyTorch wheel。
- `torch==2.6.0`：模型、稀疏图、GRU、自动求导、优化器和 checkpoint。
- `numpy`、`pandas`、`scikit-learn`：数值处理、表格处理和实验辅助算法。
- `transformers`、`sentence-transformers`、`huggingface-hub`：下载并本地运行
  Qwen3-Embedding-8B，以及执行 768 维 MRL 向量导出。
- `datasets`、`accelerate`、`peft`、`tiktoken`：保留后续生成模型和参数高效微调接口。
- `PyYAML`：读取两个 YAML 配置文件。
- `requests`、`tqdm`：数据下载、HTTP 工具与进度显示。
- `openai`：通过 OpenAI-compatible 协议调用千问 embedding endpoint。

### 3.4 环境验收

```bash
python -c "import torch, numpy; print(torch.__version__); print(numpy.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

`torch.cuda.is_available()` 必须为 `True`。驱动版本和 `torch.version.cuda` 不需要完全
相同，只要驱动能够向下兼容 PyTorch wheel 所带的 CUDA runtime。

## 4. 完整实验目录约定

```text
SIR-CDR/
├── cdr_framework/                 核心 Python 包
├── configs/                       数据、向量、tokenizer 和推荐配置
├── experiments/                   命令行入口
├── tests/                         单元测试与回归测试
├── docs/                          设计、预处理和本说明文档
├── data/raw/amazon2014/           原始 gzip 数据，不提交 Git
├── data/processed/sports_to_clothing/ 预处理 JSON/JSONL，不提交 Git
├── artifacts/embeddings/          千问向量和断点分块，不提交 Git
├── artifacts/tokenizer/           连续语义向量、码本和 Semantic ID，不提交 Git
├── artifacts/text_cdr/            v2 checkpoint、指标和日志，不提交 Git
└── logs/                           nohup 日志，不提交 Git
```

数据和模型产物体积大且与机器运行相关，GitHub 只保存代码、配置和文档。

## 5. 数据文件契约

预处理目录包含：

| 文件 | 内容 |
|---|---|
| `mappings.json` | 用户映射、商品映射、源域/目标域名称、0 号 padding |
| `item_texts.jsonl` | 每行一个 `item_id`、`domain`、拼接后的商品文本 |
| `train.jsonl` | 训练样本，包含源历史、目标历史、下一目标商品和时间戳 |
| `validation.jsonl` | 每个用户倒数第二个有效目标行为 |
| `test.jsonl` | 每个用户最后一个有效目标行为 |
| `statistics.json` | 原始数、保留数、丢弃数和各 split 行数 |
| `manifest.json` | 参数、源文件哈希、输出计数和 schema 版本 |

一个推荐样本的逻辑形式为：

```json
{
  "user_id": 10,
  "source_items": [2, 9, 18],
  "target_items": [14001, 14008],
  "positive_target_item": 14020,
  "timestamp": 1400000000
}
```

模型不会把 `10`、`2` 或 `14020` 当作数值特征。它们只用于从固定的文本向量表和
Semantic ID 表中取对应行，以及确定交叉熵标签。

## 6. 从零开始的完整运行顺序

### 6.1 第一步：预处理 Amazon 数据

```bash
cd /root/autodl-tmp/SIR-CDR
conda activate sir-cdr
mkdir -p logs

nohup python -u experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml \
  --skip-download \
  > logs/prepare_amazon.log 2>&1 &
echo $! > logs/prepare_amazon.pid
```

完成后检查：

```bash
tail -50 logs/prepare_amazon.log
python -m json.tool data/processed/sports_to_clothing/statistics.json
wc -l data/processed/sports_to_clothing/*.jsonl
```

### 6.2 第二步：下载并配置 Qwen3-Embedding-8B

当前 v2 不再调用 DashScope embedding API，而是在 RTX 3090 上本地运行官方 8B 权重。

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-Embedding-8B",
    local_dir="/root/autodl-tmp/models/Qwen3-Embedding-8B",
)
PY

export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B
```

YAML 固定模型为 `Qwen/Qwen3-Embedding-8B`，使用 BF16、SDPA、batch size 1、最长
8192 tokens，并通过 Matryoshka Representation Learning 截断和重新归一化为 768 维。
如果环境变量未设置，Sentence Transformers 会从 Hugging Face 自动下载公共权重。

先做单商品本地冒烟测试：

```bash
python experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  --smoke-test
```

再运行完整、可断点续传的向量任务：

```bash
nohup python -u experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  > logs/embed_items.log 2>&1 &
echo $! > logs/embed_items.pid
```

### 6.3 第三步：训练 Semantic ID tokenizer

```bash
nohup python -u experiments/train_tokenizer.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda \
  > logs/train_tokenizer.log 2>&1 &
echo $! > logs/train_tokenizer.pid
```

它把 768 维千问向量压缩到 128 维，再使用 4 层 residual K-means、每层 512 个
质心，把每件商品表示为长度 4 的离散 Semantic ID。

### 6.4 第四步：运行 v1 基线（可选）

```bash
nohup python -u experiments/train_recommender.py \
  --config configs/amazon_sports_clothing.yaml \
  --device cuda \
  > logs/train_recommender_v1.log 2>&1 &
```

### 6.5 第五步：运行 text-only v2 主实验

```bash
nohup python -u experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action train \
  --variant full \
  --seeds 42 \
  --device cuda \
  > logs/text_cdr_full_seed42.log 2>&1 &
echo $! > logs/text_cdr_full_seed42.pid
```

查看状态：

```bash
tail -f logs/text_cdr_full_seed42.log
ps -p "$(cat logs/text_cdr_full_seed42.pid)" -o pid,stat,etime,%cpu,%mem,rss,cmd
nvidia-smi
```

相同命令重新执行时，如果输出目录有 `last_checkpoint.pt`，训练从断点继续；如果已有
完整 `manifest.json`，程序验证配置、输入和代码哈希后直接返回已完成结果。

### 6.6 第六步：只评测最佳 checkpoint

验证集用于选超参数：

```bash
python experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action evaluate --variant full --seeds 42 \
  --split validation --mode hybrid --device cuda
```

所有超参数固定后，测试集只运行一次：

```bash
python experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action evaluate --variant full --seeds 42 \
  --split test --mode hybrid --device cuda
```

### 6.7 第七步：消融与多随机种子

```bash
nohup python -u experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action suite \
  --variants full no_graph no_reasoning single_step no_feedback no_semantic no_cpf_loss target_only \
  --seeds 42 43 44 \
  --device cuda \
  > logs/text_cdr_suite.log 2>&1 &
```

汇总验证集均值与样本标准差：

```bash
python experiments/summarize_text_cdr.py \
        --root artifacts/text_cdr/qwen3_8b/sports_to_clothing
```

## 7. 主实验的端到端调用栈

执行 `run_text_cdr.py --action train` 后，调用关系如下：

```text
run_text_cdr.main
  -> TextCDRConfig.from_yaml
  -> variant_config
  -> text_training.train
       -> signature
       -> build_model
            -> load_fixed_catalog
            -> build_training_graphs
            -> TextSIRCDR.__init__
       -> load_rows(train/validation)
       -> DataLoader(collate_text_rows)
       -> TextSIRCDR.forward
            -> item_states
            -> encode
                 -> sequence(source)
                 -> sequence(target)
                 -> UserImplicitReasoner
                 -> CPF prediction/feedback
                 -> query + generation prefix
            -> full-target retrieval loss
            -> Semantic ID generation loss
            -> CPF/alignment/separation losses
       -> backward -> gradient clipping -> AdamW.step
       -> evaluate
            -> TextSIRCDR.rank
            -> full-catalog retrieval
            -> Semantic ID reranking or constrained generation
            -> HR/NDCG/MRR/Coverage
       -> scheduler/early stop/checkpoint/manifest
```

## 8. 张量与符号约定

| 符号 | 当前配置 | 含义 |
|---|---:|---|
| `N` | 26102（含 padding） | 全局商品数 |
| `Nt` | 13044 | 目标域商品数 |
| `B` | 128 训练/16 评测 | batch size |
| `H` | 128 | 隐空间维度 |
| `L` | 4 | Semantic ID 长度 |
| `K` | 512 | 每层码本大小 |
| `T` | 最多 50 | 历史序列长度 |
| `R` | 3 | 每轮门控隐推理步数 |
| `F` | 2 | CPF 外部反馈轮数 |

全局 0 号商品是 padding。源域 ID 和目标域 ID 共用一个全局索引空间，但所属域由
`mappings.json` 决定。

## 9. text-only v2 逐文件、逐代码段说明

### 9.1 `cdr_framework/text_config.py`

- 第 1-5 行导入 dataclass、路径、有限数检查和 YAML 解析。
- 第 7 行复用 v1 的 `RecommendationTrainingConfig`，因此处理目录、tokenizer 目录、
  hidden size、batch size、top-k 等共同字段不重复定义。
- `TextCDRConfig` 是不可变 dataclass。不可变配置能避免训练过程中被意外修改。
- `dropout` 控制序列输入正则化；`graph_layers` 控制稀疏传播层数；`cross_window`
  控制每个用户两个域末尾多少商品建立跨域共现边。
- `feedback_steps` 是 CPF 外循环轮数，`reasoning_steps` 是每轮内部连续门控更新次数。
- `graph_enabled`、`reasoning_enabled`、`semantic_enabled`、`source_enabled` 为消融开关。
- `cpf_weight`、`alignment_weight`、`separation_weight` 是三个辅助目标系数。
- `generation_weight` 和 `retrieval_weight` 只用于推理时混合两类分数。
- `inference_mode` 支持 `retrieval/hybrid/exhaustive/generate`。
- `__post_init__` 先调用父类检查，再逐项检查整数、dropout、非负损失权重、推理模式
  和早停指标，目的是在占用 GPU 前让错误配置尽早失败。
- `from_yaml` 只接受 `text_recommendation` 段，避免误把 v1 的 `recommendation` 段
  当作 v2 参数。

### 9.2 `cdr_framework/text_data.py`

- `TextRow` 保存一条样本；`seen_items` 保存未截断的完整目标历史，用于评测屏蔽。
- `load_text_rows(path, max_length, num_items, target_ids)` 逐行解析 JSONL。
- `targets = set(...)` 把目标域 tensor 转成集合，使域合法性检查接近 O(1)。
- 空行被忽略；源/目标历史为空、正标签不在目标域时立即报错。
- 源历史必须是非零、合法、且不属于目标域；目标历史必须全部属于目标域。
- 写入模型的序列用 `[-max_length:]` 截断；`seen_items` 使用完整历史，因此较早交互
  即使被模型输入截断，也不会重新成为可推荐候选。
- `collate_text_rows` 把 `TextRow` 转成 v1 已验证过的 `FormalBatch`，并单独返回每个
  用户的 seen set。这样训练批处理与全历史屏蔽职责互不混淆。

### 9.3 `cdr_framework/text_graph.py`

- `_integer` 拒绝布尔值、浮点值和小于下限的整数，集中完成参数检查。
- `_normalized_graph` 加入 1 到 `N-1` 的自环；0 号 padding 不加边。随后计算
  `D^(-1/2) A D^(-1/2)`，返回 coalesced sparse COO tensor。
- `build_training_graphs` **只打开 `train.jsonl`**。
- `target_item_ids` 决定域归属，其余非零商品视为源域。
- 内部 `validate_item` 检查范围、padding 规则和所属域。
- `latest` 对每个用户只保留时间戳最大的训练行；同时间戳时后出现的行覆盖前一行。
  这是为了用该用户训练阶段可见的最长前缀构一次静态图，避免重复边处理。
- `positive_target_item` 只校验，不连接到图；所以当前训练标签不会通过边泄漏。
- `connect` 同时插入 `(u,v)` 和 `(v,u)`，构造无向图。
- 源图和目标图连接各自历史中的相邻商品。
- 共享图是源边、目标边和跨域边的并集；跨域边来自两个历史末尾
  `cross_window` 商品的笛卡尔积，表达同一用户近期跨域共现。
- `SparseDualGraphEncoder` 为 shared/source/target 三条路径各建一个无偏置线性投影。
- `forward` 对每条路径计算 `mean(X, AX, ..., A^layers X)` 再投影。它没有
  `nn.Embedding(num_items, ...)`，所以图传播的初始节点状态完全来自商品文本。

### 9.4 `cdr_framework/catalog_generation.py`

- `CatalogTrie` 把目标目录中每件商品的固定长度 Semantic ID 建成前缀树。
- 构造函数验证 item table、token table 的形状和整数类型，排除 padding item 0，
  同时允许 token 值本身为 0。
- `_children[prefix]` 保存某个前缀后允许出现的 token；`_terminal[sequence]` 保存
  完整序列对应的一个或多个商品。
- `next_codes(prefix)` 是 beam search 的词表约束，只返回目录中真实存在的下一 token。
- `terminal_items(sequence)` 解决 Semantic ID 碰撞：一个 token 序列可以映射多件商品。
- `constrained_generate` 对 batch 中每个用户独立做 beam search。
- 每一步只扩展 trie 允许的 token，而不是从整个 `L*K` 词表盲目生成。
- beam 分数是累计对数概率，并使用长度归一化；结束后只接收长度恰好为 `L` 的序列。
- 若多个商品共享序列，使用用户 query 与文本 item vector 的余弦相似度打破平局。
- `seen_items` 在输出商品层屏蔽，保证生成模式也不会推荐历史商品。

### 9.5 `cdr_framework/text_model.py`

#### `TextSIRCDR.__init__`

- `content` 注册 tokenizer 输出的 `[N,H]` 固定连续文本 latent，不作为参数更新。
- `offsets = arange(L) * K` 为不同 Semantic ID 深度划分不重叠 token 区间。
- `tokens = semantic_ids + offsets` 把每层 `[0,K)` 转为统一词表 `[0,L*K)`；
  padding 商品整行强制为 0。
- `target_ids` 保存完整目标目录；`target_index` 建立全局商品 ID 到目标 softmax 列号
  的反向映射，源域位置保持 -1。
- `centroids` 把 `[L,K,H]` 码本展平为 `[L*K,H]`。
- 三张稀疏图作为 non-persistent buffer 随模型移动设备，但不重复写入 checkpoint；
  恢复模型时按输入数据重新构建。
- `content_proj` 和 `code_proj` 把连续 latent 与离散质心分别映射到模型空间。
- 两个 GRU 分别编码源域和目标域历史，避免把域差异强行共享。
- `shared_head` 对两个域复用同一参数；`source_head/target_head` 学习域私有表示。
- `transfer_gate` 决定每一维应该从源域共享信号迁移多少。
- `UserImplicitReasoner` 完成连续门控隐推理；`ContextPredictionFeedback` 提供 CPF
  predictor；`feedback_gate/proj` 把预测反馈给下一轮 context。
- `query_head` 产生检索向量；`prefix_head` 产生生成解码器初始条件。
- `AutoregressiveSemanticDecoder` 逐位置预测 Semantic ID。
- `CatalogTrie` 只包含目标域商品，因此生成天然受目标目录约束。

#### `item_states`

1. `content_proj(content)` 得到 `[N,H]` 文本基础状态。
2. 若开启 Semantic ID，把每件商品 4 个码本质心投影后取平均并加到文本状态。
3. LayerNorm 后把 padding 行乘零。
4. 若开启图，分别获得 shared/source/target `[N,H]` 图状态；否则返回零张量。

#### `sequence`

1. 按历史 ID 从 base、shared、specific 三张表取 `[B,T,H]`。
2. `structural_gate(cat(...))` 输出逐维门值 `g`。
3. 输入为 `base + g*shared + (1-g)*specific`，这使共享与私有结构按用户序列位置自适应融合。
4. `pack_padded_sequence` 避免 GRU 计算 padding。
5. attention 对有效时间步加权池化，并与 GRU 最后状态平均，得到 `[B,H]`。

#### `encode`

1. 分别得到 source、target 用户序列向量。
2. `target_only` 消融把 source 置零。
3. 同一 `shared_head` 得到两域共享表示；两个 private head 得到域私有表示。
4. `transfer=sigmoid(...)` 计算源到目标的连续迁移门。
5. `shared_signal = transfer*source_shared + (1-transfer)*target_shared`。
6. `context = target + transfer*source`，得到隐推理初始上下文。
7. 每个 CPF 外循环调用一次 `UserImplicitReasoner`；若关闭 reasoning，直接使用已有表示。
8. CPF predictor 根据 `[shared+private, context]` 预测正商品文本空间中的向量。
9. 除最后一轮外，`feedback_gate` 决定预测向量的哪些维度写回 context。
10. `query` 用于检索；`prefix` 汇合 shared、private、结构和预测，供生成器使用。

#### `forward`

- 检索分支计算 query 与 **全部目标商品** 文本基础状态的相似度，除以 temperature，
  再用目标目录列号做交叉熵。这就是无随机负采样的完整目标域训练。
- 生成分支把 BOS 和真实 4-token Semantic ID 输入 decoder，目标为 4 个 token 加 EOS。
- CPF 的正目标 `base[positive].detach()` 被停止梯度，防止辅助目标通过移动自己的目标
  得到虚假的低损失。
- CPF 损失包含每轮预测余弦误差、shared/context 对齐和推理状态防坍塌方差项。
- alignment 约束源/目标 shared 接近；separation 约束 shared/private 正交。
- 最终损失为 retrieval + generation + 加权的 CPF/alignment/separation。

#### `sequence_scores` 与 `rank`

- `sequence_scores` 计算候选商品完整 Semantic ID 加 EOS 的平均 log probability。
- `retrieval`：完整目录检索后直接取 Top-K。
- `hybrid`：完整目录检索，取前 `rerank_candidates`，再融合生成分数重排。
- `exhaustive`：对完整目标目录的每个候选都计算生成分数，最严格但最慢。
- `generate`：通过 trie 约束 beam search 直接产生目录合法 Semantic ID。
- 所有模式先屏蔽完整目标历史；不足 K 个合法结果时用 0 padding，不把 0 当推荐结果。

### 9.6 `cdr_framework/modules.py` 中主线使用的三部分

#### `UserImplicitReasoner`：连续门控隐推理

- `init([context, cd_signal, target_sp_signal])` 生成初始隐状态 `s0`。
- 每一步把当前状态、上下文、共享迁移信号和目标私有信号拼接。
- `read=sigmoid(...)` 控制旧状态中多少内容参与候选状态计算。
- `write=sigmoid(...)` 控制候选知识写入多少。
- `forget=sigmoid(...)` 控制旧状态保留多少。
- `candidate=tanh(...)` 生成待写入的新推理内容。
- `state=LayerNorm(forget*state + write*candidate)` 完成连续空间状态更新。
- 循环 `reasoning_steps` 次，保存所有状态供防坍塌损失使用。
- 最终状态与跨域信号产生 shared latent，与目标私有信号产生 private latent。

它叫“隐推理”，因为中间推理状态是不可见的连续向量，不输出自然语言思维链。门参数由
推荐损失反向传播学习；代码中没有 prompt、tokenizer、LLM forward 或 API 请求，
所以它与大模型推理无关。千问只在更早的离线步骤把商品文本映射成初始向量。

#### `DualStructuralFusionPrefix`

把五个 `[B,H]` 信号拼成 `[B,5H]`，线性投影到 `[B,prefix_length*H]`，再 reshape
为 `[B,prefix_length,H]`。这不是自然语言 prompt，而是可学习的连续前缀。

#### `AutoregressiveSemanticDecoder`

- 词表大小为 `L*K`，另加 BOS 和 EOS。
- 把连续 prefix 展平并映射为 GRU 初始状态。
- teacher forcing 训练时一次输入 BOS+真实 tokens。
- `next_log_probs` 在受约束生成时返回下一 token 的对数概率。

文件中其他 multimodal、旧图编码和注入类服务于 synthetic/v1 兼容路径，不被 v2
的 `TextSIRCDR` 实例化。

### 9.7 `cdr_framework/modules_cpf.py`

- `CPFOutput` 统一保存预测向量和命名损失。
- `ContextPredictionFeedback.predictor` 是 Linear -> LayerNorm -> Tanh -> Linear。
- `forward` 的 v1 用法计算预测 MSE、上下文对齐和状态方差，再由 `cpf_total_loss`
  加权。v2 复用其 `predictor`，但使用余弦目标并显式实现多轮反馈。
- CPF 在本项目中解释为 **Context Prediction Feedback**，即上下文预测反馈。

### 9.8 `cdr_framework/text_training.py`

- `VARIANTS` 是允许的消融集合。
- `variant_config` 使用 `dataclasses.replace` 生成不可变配置副本，并为每个
  variant/seed 建独立输出目录，防止 checkpoint 相互覆盖。
- `signature` 对配置、输入数据、tokenizer 产物和关键源码计算哈希。恢复训练和评测
  必须匹配该签名，保证结果确实来自当前代码。
- `build_model` 检查 NaN/Inf、Semantic ID 范围和 latent/码本维度，再构图和实例化模型。
- `evaluate` 在 `torch.no_grad()` 下排序，逐样本计算 HR、NDCG、MRR，并提供论文常用
  别名 `R@K=HR@K`、`N@K=NDCG@K`。`positives_in_seen` 应为 0。
- `train` 固定 Python、CPU Torch 和 CUDA 随机种子。
- AdamW 更新参数；ReduceLROnPlateau 在验证指标长期不升时把学习率乘 0.5。
- DataLoader 每个 epoch 使用 `seed+epoch`，使 shuffle 可复现。
- 每个 batch 执行 zero_grad -> forward -> finite check -> backward -> 梯度裁剪 -> step。
- 第 1 轮、每隔 `evaluation_every` 轮和最后一轮做验证。
- `selection_metric` 改善时写 `best_model.pt`，否则增加 stale；达到 patience 提前停止。
- 每轮原子写入 `last_checkpoint.pt` 和 `history.json`，中断后可恢复。
- 完整结束才写 `manifest.json`；短暂运行或崩溃不会被误认为完成。
- `evaluate_checkpoint` 强制检查签名，只加载最佳验证模型，并把结果写入
  `<split>_<mode>.json`。

### 9.9 `experiments/run_text_cdr.py`

- `ROOT` 由脚本位置计算，不依赖启动时所在目录。
- `sys.path.insert` 允许直接以脚本方式运行本仓库包。
- `--action train` 训练一个 variant；`evaluate` 评测；`suite` 顺序运行多个消融。
- `--seeds` 支持一个或多个随机种子。
- `--output-root` 可把实验产物放到另一块磁盘。
- 第 27-28 行把 YAML 相对路径转换为仓库绝对路径。
- 双重循环依次处理 variant 和 seed，最后输出机器可读 JSON。

### 9.10 `experiments/summarize_text_cdr.py`

- 扫描输出根目录下一层的 `manifest.json`。
- 从目录名的 `_seed` 后缀恢复 variant 名。
- 只读取 `best_validation`，避免利用 test 指标选择模型。
- 对 R@5、N@5、R@10、N@10 计算均值和样本标准差；只有一个 seed 时标准差为 null。

## 10. 数据预处理代码

### 10.1 `cdr_framework/datasets/amazon2014.py`

- `category_files` 根据 Amazon 2014 类别名生成 reviews/meta 两个 gzip 路径。
- `iter_amazon_records` 流式读取 gzip；先尝试 JSON，再兼容 Amazon 旧文件中的
  Python literal 格式，避免一次把数百 MB 文件载入内存。
- `sha256_file` 分块计算哈希；`_validate_gzip` 验证压缩文件完整性。
- `_download_file` 下载到临时文件，完成并验证后再替换正式文件。
- `download_category_files` 负责两个 URL，已存在文件默认复用，`force` 才覆盖。

### 10.2 `cdr_framework/datasets/amazon_preprocessing.py`

- `_TextExtractor` 去除 HTML 标签，只保留可见文本。
- `AmazonEvent` 是标准化交互；`PreparedDomainPair` 是过滤映射后的内存数据；
  `RecommendationSample` 是模型样本；`TemporalSamples` 保存三个 split。
- `_clean_scalar/_flatten/_categories/_sentence` 递归清洗元数据，处理字符串、列表、
  HTML、空值和类别路径。
- `build_product_text` 按 Title、Brand、Categories、Features、Description 的固定顺序
  拼文本，并按字符上限截断。ASIN 不拼进文本。
- `_read_events` 读取 reviewerID、asin、unixReviewTime，统计坏记录。
- `_metadata_texts` 只为确实有交互的商品提取文本。
- `_group` 按用户分组，并按 `(timestamp, asin)` 稳定排序。
- `prepare_domain_pair` 要求用户在两个域都至少有指定交互数，并为保留用户和商品
  建连续整数映射；0 保留给 padding。
- `_sample_for` 只取发生在目标标签时间之前的源行为，并取目标域当前索引之前的行为，
  从源头阻止未来信息进入历史。
- `build_temporal_splits` 把每个用户最后一个目标行为放 test、倒数第二个放 validation、
  更早有效行为放 train。
- `_write_json/_write_jsonl` 负责确定性 UTF-8 输出。
- `write_preprocessed_artifacts` 先写临时目录并逐行回读验证，再原子替换目标目录；
  `manifest` 记录源文件哈希和参数。

### 10.3 `experiments/prepare_amazon.py`

- `build_parser` 定义 `--config/--skip-download/--force`。
- `_repository_path` 把相对路径锚定到仓库根目录。
- `_required_files` 为源域和目标域分别解析原始文件名。
- `_ensure_raw_files` 根据 skip-download 决定只检查还是自动下载。
- `run` 组合 IO、过滤、时间切分和原子写出，并打印关键统计。
- `main` 读取 YAML，发生错误时返回非零退出码，便于 nohup/调度器判断失败。

## 11. 千问 embedding 代码

### 11.1 `cdr_framework/embeddings/providers.py`

定义三个抽象接口：文本向量、图片向量、LLM 语义服务。text-only v2 只使用
`BaseTextEmbeddingProvider.encode_text`。

### 11.2 `cdr_framework/embeddings/qwen3_local.py`

- `Qwen3EmbeddingProvider` 延迟加载官方 `Qwen/Qwen3-Embedding-8B`，空输入不会加载模型。
- `QWEN3_EMBEDDING_MODEL_PATH` 可把模型解析到服务器本地目录，否则使用 Hugging Face ID。
- CUDA 使用 BF16 和 SDPA，tokenizer 使用 left padding，最大序列长度由 YAML 控制。
- `encode_text` 调用 Sentence Transformers，截取前 768 个 MRL 维度并重新 L2 归一化。
- 输出统一搬到 CPU float32，非有限值或错误 shape 会立即终止任务。

### 11.3 `cdr_framework/embeddings/api.py`（旧版兼容）

- `QwenTextEmbeddingProvider` 延迟创建 OpenAI client，只有真正调用 `encode_text` 时
  才读取密钥和联网。
- `_get_client` 检查 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。
- `encode_text` 把文本列表提交给 embedding endpoint，检查返回条数和维度，转成
  float32 tensor。
- `ReservedAPITextEmbeddingProvider` 和 `DeepSeekTextEmbeddingProvider` 是保留接口；
  当前正式配置不使用 API provider。
- `build_text_embedding_provider` 根据配置统一创建 provider；API 默认关闭的旧测试路径
  使用本地 deterministic provider。

### 11.4 `cdr_framework/embeddings/qwen.py`

- `ItemText` 表示待向量化商品；`EmbeddingRunResult` 表示任务结果。
- `sha256_path` 给输入 `item_texts.jsonl` 计算内容身份。
- `load_item_texts` 验证非零且不重复的 item ID、非空 domain 和 text，再按 ID 排序。
- `_write_json_atomic/_save_tensor_atomic` 先写 `.part` 再替换，减少中断损坏。
- `_encode_with_retry` 按 `initial_delay*2^attempt` 指数退避重试。
- `_managed_output` 只允许 `--force` 删除本程序带正确 schema 标记的目录。
- `run_embedding_job` 按 batch 保存 `chunks/batch-xxxxxx.pt`；重启时核对 ID、shape、
  provider、模型、维度和最大序列长度后跳过已完成块；每块更新 `progress.json`。
- 所有块完成后组装 `[max_item_id+1,768]`，0 行保持全零，并写 manifest/index。

### 11.5 `experiments/embed_items.py`

- CLI 读取 YAML embedding 段并将路径绝对化。
- `--smoke-test` 只取一件商品，加载本地 8B 模型但不写正式产物。
- 正式配置创建 `Qwen3EmbeddingProvider`；旧配置仍可创建 `QwenTextEmbeddingProvider`。
- `--force` 只在确认需要从零重建当前 managed output 时使用。

### 11.6 本地与缓存兼容文件

- `embeddings/local.py`：把文本哈希确定性映射成归一化向量，仅用于离线测试。
- `embeddings/cache.py`：按 namespace/key 保存独立 tensor 的通用缓存。
- `embeddings/__init__.py`：集中 re-export 公共类和工厂。

## 12. Semantic ID tokenizer 代码

### 12.1 `cdr_framework/tokenization/tokenizer.py`

`DomainAdaptiveSemanticTokenizer` 包含 universal 投影、domain-specific 路径和门控融合。
输入 `[N,768]` 与 domain ID，输出 128 维融合 latent、通用 latent、域 latent 和 gate。

### 12.2 `cdr_framework/tokenization/residual_kmeans.py`

- `_assign_nearest` 分块计算样本到质心的平方距离，返回最近质心，控制显存。
- `_initial_centroids` 用固定随机种子选择初始点。
- `_fit_level` 反复 assignment/update；空簇从误差最大的样本重新初始化，避免码本空洞。
- `fit_residual_kmeans` 每层对当前 residual 聚类、记录 token、减去选中质心；四层顺序
  逼近连续 latent。
- `quantize_residuals` 用训练好的多层码本把新向量转 token 并重建。

### 12.3 `cdr_framework/tokenization/training.py`

- `TokenizerDataset` 保存 embedding、domain 与 item ID 对齐结果。
- `load_tokenizer_dataset` 核对 embedding manifest、item text 顺序、维度、padding 和 domain。
- `TrainableSemanticTokenizer` 训练连续压缩和域自适应门。
- `_checkpoint` 打包 schema、模型、优化器、epoch、随机状态和配置，用于恢复。
- `train_semantic_tokenizer` 先优化 reconstruction/domain/gate balance，再对最终 latent 拟合
  residual K-means，输出 item_latents、semantic_ids、codebooks、quality report 和 manifest。
- quality gate 检查 collision rate 和每层 utilization；不通过则不能进入正式推荐训练。

### 12.4 兼容 re-export 文件

- `tokenization/__init__.py`：对外暴露 tokenizer、码本、训练函数。
- `tokenization/codebook.py`、`item_index.py`：把旧路径重导向根目录 `codebook.py`，
  保持历史 import 不失效。
- `experiments/train_tokenizer.py`：解析 CLI、加载配置、打印每轮指标和最终质量。

## 13. v1 与合成框架文件

### 13.1 `cdr_framework/formal_recommendation.py`

- `RecommendationRow/FormalBatch` 定义 v1 数据，`FormalBatch.to` 整批搬到设备。
- `FixedCatalog` 校验 item latent、Semantic ID、codebook、目标 ID 的一致性。
- `load_recommendation_rows` 读取并截断 JSONL；`collate_recommendation_rows` padding 序列。
- `SIRCDRRecommender` 用固定文本 latent 和 Semantic ID，不训练 raw ID embedding。
- `encode_user` 用源/目标 GRU、用户解耦器、门控隐推理和结构前缀编码用户。
- `forward` 联合 generation、full-target retrieval、CPF、正交与分离损失。
- `rank_full_catalog` 分块扫描所有目标商品，屏蔽历史，保留候选后按 generation 重排。
  `generation_weight=0` 时直接返回 retrieval 排名，用于诊断。

### 13.2 `cdr_framework/formal_training.py`

- `_config_payload` 把 dataclass 转成可序列化字典。
- `_save_atomic/_write_json` 负责 checkpoint 和 JSON 原子写。
- `_artifact_fingerprints` 对 split 和 tokenizer 产物做 SHA256。
- `load_fixed_catalog` 要求 tokenizer quality passed 且 schema_version=2。
- `evaluate_full_catalog` 逐 batch 做全目标目录 HR/NDCG/MRR/Coverage。
- `run_formal_training` 支持恢复、梯度裁剪、验证早停、best/last checkpoint 和 manifest。
- `evaluate_saved_model` 校验配置与输入哈希后评测验证或测试 split。

### 13.3 旧版入口与诊断工具

- `experiments/train_recommender.py`：v1 训练 CLI。
- `experiments/evaluate_recommender.py`：v1 最佳 checkpoint 全目录评测。
- `experiments/diagnose_recommender.py`：比较 popularity、retrieval-only、generation-reranked，
  并报告 candidate recall 和数据/产物哈希。
- `experiments/sweep_reranking.py`：只在验证集扫描候选数与 retrieval/generation 权重；
  禁止利用测试集调参。
- `cdr_framework/framework.py`：早期端到端 synthetic 模型，验证模块接口和损失能运行。
- `experiments/run_synthetic.py`：生成随机 batch、前向计算损失并展示推荐结果。

## 14. 通用基础文件

- `cdr_framework/config.py`：早期通用 Dataset/Embedding/Model/Training 配置及合法性检查。
- `cdr_framework/experiment_config.py`：Amazon 预处理、千问 embedding、tokenizer、v1 推荐
  四组正式 YAML dataclass；`from_yaml` 分别读取对应段。
- `cdr_framework/data.py`：synthetic 路径的 ItemFeatures、GraphBatch、InteractionBatch、ForwardOutput。
- `cdr_framework/adapters.py`：数据 provider 与 semantic backbone 抽象接口；identity backbone
  用于接口占位。
- `cdr_framework/losses.py`：共享/私有正交、跨域共享对齐、防坍塌方差、私有分离、CPF
  上下文对齐和损失组合。
- `cdr_framework/metrics.py`：通用 HR@K、NDCG@K、MRR@K、Coverage 和输入检查。
- `cdr_framework/ops.py`：序列均值、余弦平局分数和旧版稠密对称邻接构造。
- `cdr_framework/codebook.py`：synthetic/v1 的语义码本、残差量化、item-token 反查和普通 beam search。
- `cdr_framework/graphs/builders.py`：从 padded 序列统计相邻转移边。
- `cdr_framework/graphs/confidence.py`：七维边置信特征及固定顺序 tensor 化。
- `cdr_framework/graphs/__init__.py`：图 API 导出。
- `cdr_framework/datasets/schema.py`：跨 Amazon/Douban 的标准交互、元数据和跨域序列 dataclass。
- `cdr_framework/datasets/splits.py`：通用按用户时间排序 leave-one-out。
- `cdr_framework/datasets/__init__.py`：数据 API 导出。
- `cdr_framework/__init__.py`：包顶层稳定入口，避免调用方依赖内部文件路径。

## 15. 配置文件逐项解释

### 15.1 `configs/amazon_sports_clothing.yaml`

`dataset` 控制类别名、原始/处理目录、每域最少交互数、文本上限和 seed。

`embedding` 固定 provider=qwen3_local、model=Qwen/Qwen3-Embedding-8B、
dimension=768、batch=1、BF16 CUDA 和 8192-token 上限。模型路径由
`QWEN3_EMBEDDING_MODEL_PATH` 覆盖。

`tokenizer` 固定 768 -> 128、4 层 x 512 码本、100 epoch，以及 collision/utilization
质量阈值。

`recommendation` 是 v1 参数，不控制 v2。

### 15.2 `configs/text_sports_clothing.yaml`

- `processed_dir/tokenizer_dir` 复用已完成数据与 Semantic ID。
- `output_dir` 是根目录，程序再追加 `<variant>_seed<seed>`。
- `learning_rate=3e-4`、`weight_decay=1e-4`、`dropout=.1` 比 v1 更保守。
- `evaluation_every=2`、`patience=8`、`scheduler_patience=2` 控制调度与早停。
- `selection_metric=NDCG@10` 明确最佳模型选择规则。
- `inference_mode=hybrid` 先完整检索，再重排 500 个候选。
- `decode_chunk_size=256` 控制生成候选打分显存。
- `beam_size=50` 只在 generate 模式使用。

## 16. 输出文件与读取方法

每个 v2 运行目录，例如 `artifacts/text_cdr/qwen3_8b/sports_to_clothing/full_seed42/`：

| 文件 | 作用 |
|---|---|
| `last_checkpoint.pt` | 最近一轮模型、优化器、scheduler、随机状态，供恢复 |
| `best_model.pt` | 验证选择指标最优的模型 |
| `history.json` | 每轮训练损失、学习率和周期性验证指标 |
| `manifest.json` | 完成标记、最佳验证指标、完整身份签名 |
| `validation_hybrid.json` | 显式验证命令写出的结果 |
| `test_hybrid.json` | 最终测试结果 |

快速读取 R@5、N@5、R@10、N@10：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('artifacts/text_cdr/qwen3_8b/sports_to_clothing/full_seed42/test_hybrid.json')
m = json.loads(p.read_text())['metrics']
for key in ('R@5', 'N@5', 'R@10', 'N@10'):
    print(f'{key}: {m[key]:.6f}')
PY
```

## 17. 测试文件逐一说明

测试名称基本采用“被测行为即测试名”的形式：

- `test_amazon2014_io.py`：gzip JSON/literal 解析、下载、校验和与错误处理。
- `test_amazon_preprocessing.py`：文本清洗、跨域用户过滤、时间切分、无未来泄漏、写出。
- `test_prepare_amazon_cli.py`：离线 CLI、manifest/statistics、缺失文件提示。
- `test_qwen_embedding_job.py`：向量 shape、分块恢复、重试、identity mismatch 和原子输出。
- `test_embedding_providers.py`：本地 provider 确定性、API 默认关闭、保留接口。
- `test_tokenizer_training.py`：连续 tokenizer 训练、恢复、manifest 和质量门。
- `test_residual_kmeans.py`：多层码本独立、重建、利用率与确定性。
- `test_tokenization_boundary.py`：包 re-export 和域自适应融合。
- `test_formal_recommendation.py`：v1 数据校验、无 ID 特征、全目录屏蔽和排序。
- `test_formal_training.py`：v1 训练、恢复、checkpoint 身份和完整目录协议。
- `test_recommendation_diagnostics.py`：诊断输出字段和检索/重排比较。
- `test_reranking_sweep.py`：验证集扫描、最佳组合与禁止 test 调参。
- `test_text_graph.py`：训练图无标签泄漏、域边、稀疏归一化、图编码 shape。
- `test_catalog_generation.py`：trie、受约束 beam、重复 Semantic ID 平局与 seen mask。
- `test_text_pipeline.py`：v2 配置、数据、前向损失、排名、恢复和 identity 检查。
- `test_codebook.py`：早期残差量化、beam 长度归一化和 item-token 映射。
- `test_modules.py`：结构注入与隐推理输出 shape。
- `test_cpf.py`：CPF 预测、对齐和命名损失。
- `test_graph_features.py`：转移边、七维特征顺序和旧图编码器。
- `test_framework_forward.py`：synthetic 端到端损失与推荐。
- `test_config_metrics.py`：配置拒绝非法值及指标公式。
- `test_dataset_schemas.py`：Amazon/Douban 通用 schema 与时间切分。
- `test_experiment_config.py`：YAML dataclass 路径和数值校验。
- `test_package_exports.py`：顶层公共 API 稳定。

测试只证明实现行为和接口符合约定，不证明推荐指标一定提升。正式论文结果必须来自真实
数据、多个随机种子、验证集选参和一次性测试集评测。

## 18. 关键指标定义

对每条样本只有一个正目标商品：

- `R@K` 与 `HR@K` 等价：正商品是否出现在前 K，最后对样本取平均。
- `N@K` 与 `NDCG@K` 等价：命中时贡献 `1/log2(rank+1)`。
- `MRR@K`：命中时贡献 `1/rank`。
- `Coverage`：所有推荐列表中出现过的不同目标商品数 / 目标目录商品总数。

因为使用全目标目录，指标不能与采用 99 个随机负样本的论文数字直接横向比较。论文中
必须明确写出 full-catalog protocol。

## 19. 消融实验含义

| variant | 关闭内容 | 回答的问题 |
|---|---|---|
| `full` | 无 | 完整模型 |
| `no_graph` | 三路图传播 | 结构编码是否有效 |
| `no_reasoning` | 门控隐推理与多轮反馈 | 隐推理整体贡献 |
| `single_step` | 内部推理仅一步 | 多步状态更新贡献 |
| `no_feedback` | CPF 外循环仅一轮 | 预测反馈贡献 |
| `no_semantic` | 商品端码本质心融合 | 结构化编码对表示的贡献 |
| `no_cpf_loss` | CPF 辅助损失权重为 0 | CPF 监督贡献 |
| `target_only` | 源域和图关闭 | 跨域迁移相对单域的贡献 |

## 20. 常见故障排查

### `conda activate` 报 `Run conda init`

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr
```

### 找不到 Qwen3-Embedding-8B 权重

```bash
export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B
test -f "$QWEN3_EMBEDDING_MODEL_PATH/config.json" && echo model_loaded
```

这是公共本地模型，不需要 DashScope API key；不要把数十 GB 权重提交到 Git。

### 训练提示 identity mismatch

表示配置、数据、tokenizer 产物或关键源码与 checkpoint 创建时不同。为了实验可追溯性，
程序拒绝静默混用。保留旧目录并用新的 `--output-root`，不要手工修改 checkpoint。

### CUDA OOM

优先减小 YAML 中 `batch_size`、`evaluation_batch_size` 和 `decode_chunk_size`。
`rerank_candidates` 会显著影响 hybrid 生成重排耗时，但不影响第一阶段完整目录检索。

### 指标低但 loss 下降

依次检查：candidate recall、retrieval-only 与 hybrid 差异、`positives_in_seen`、
训练/验证时间切分、Semantic ID 质量、源域是否产生负迁移。只在验证集调参，不反复看
测试集选择配置。

### 后台任务是否结束

```bash
pid=$(cat logs/text_cdr_full_seed42.pid)
ps -p "$pid" -o pid,stat,etime,%cpu,%mem,rss,cmd
tail -100 logs/text_cdr_full_seed42.log
```

`ps` 不再显示 PID 且日志最后出现结果 JSON 或 manifest 路径，才表示正常结束。

## 21. GitHub 更新与远端同步

代码、配置和文档提交 Git；不要提交 `data/`、`artifacts/`、`logs/`、模型权重或密钥文件。

Windows PowerShell：

```powershell
Set-Location "C:\Users\BACL2\Desktop\实验框架实现\code"
git status --short
git add README.md cdr_framework configs experiments tests docs requirements.txt
git commit -m "Complete text-only SIR-CDR framework and code guide"
git push origin main
```

远程服务器：

```bash
cd /root/autodl-tmp/SIR-CDR
git status --short
git pull --ff-only origin main
git log -1 --oneline
```

`logs/*.log` 是未跟踪文件时不会阻止 fast-forward pull；不要执行会删除服务器数据和
训练产物的 `git clean -fdx`。

## 22. 最终实验检查清单

- [ ] `statistics.json` 行数与预期一致。
- [ ] embedding manifest 为 `Qwen/Qwen3-Embedding-8B`、`qwen3_local`、768 维、26101 items。
- [ ] tokenizer quality `passed=true`，collision rate 与各层 utilization 达标。
- [ ] v2 输出目录按 variant/seed 隔离。
- [ ] 训练日志无 NaN/Inf，GPU 持续工作。
- [ ] `positives_in_seen=0`。
- [ ] 最佳模型由 validation `NDCG@10` 选择。
- [ ] 至少 3 个随机种子并报告 mean ± std。
- [ ] test 只在配置固定后执行一次。
- [ ] 报告 full-target、无随机负采样、文本单模态和 seen-item mask。
- [ ] 模型权重和任何历史 API key 未进入 Git、日志、截图或论文附录。

本文档落盘时执行的本地回归结果为：`Ran 91 tests ... OK`。这组测试覆盖当前仓库的
预处理、embedding job、tokenizer、v1、v2、诊断和生成约束代码；远程正式运行仍应以
服务器上的数据哈希、manifest 和实验日志为最终复现依据。

至此，代码从 Amazon 原始文件、千问文本向量、Semantic ID、训练图、连续门控隐推理、
CPF、完整目标域检索、结构化推荐生成到最终指标汇总形成一条可复现的完整实验链路。
