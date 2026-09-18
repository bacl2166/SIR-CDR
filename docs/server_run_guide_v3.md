# SIR-CDR v3 服务器运行手册（autodl / 24GB GPU）

> 面向远端 Linux 服务器（文档默认路径 `/root/autodl-tmp/SIR-CDR`，GPU 为 24GB
> 显存，conda 环境 `sir-cdr`）。v3 即文本主线接入 prototype 侧解耦 + CD/SP
> 双结构注入 + codebook 摘要前缀 + 新正则的新架构。每个阶段给出命令、验证点
> 与"产物已存在则跳过"的判断方式。

## 0. 前置清单

- 服务器有 GitHub 访问权限（见阶段一）
- 显存 ≥ 24GB（Qwen3-Embedding-8B 以 batch=1 运行）
- 磁盘 ≥ 60GB：原始 gzip 约 3-4GB、8B 模型约 16GB、嵌入与产物约 5-10GB

## 1. 服务器 GitHub 认证（一次性）

任选一种：

**方式 A：SSH 密钥**

```bash
ssh-keygen -t ed25519 -C "$(hostname)" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub   # 复制整行，添加到 GitHub → Settings → SSH and GPG keys
ssh -T git@github.com       # 看到 "successfully authenticated" 即成功
```

**方式 B：HTTPS + PAT**

```bash
git clone https://github.com/bacl2166/SIR-CDR.git /root/autodl-tmp/SIR-CDR
# 后续 push 用个人访问令牌；只读运行也建议 clone 而非 download zip，便于更新
```

## 2. 拉取代码与目录约定

```bash
mkdir -p /root/autodl-tmp/SIR-CDR && cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main        # 已 clone 时；包含 v3 提交 7f4943e
mkdir -p logs data artifacts

# 数据契约（README 有完整说明）：
# data/processed/sports_to_clothing/{train,validation,test}.jsonl + mappings.json
# artifacts/embeddings/qwen3_8b/...  artifacts/tokenizer/qwen3_8b/...  artifacts/text_cdr/qwen3_8b/...
# 原始 gzip 与 8B 权重、checkpoint、日志一律不提交 Git
```

## 3. 环境验收

```bash
conda activate sir-cdr
python -m unittest discover -s tests -v    # 预期 99 tests OK
python experiments/prepare_amazon.py --help
```

## 4. 阶段一：Amazon 数据预处理

```bash
cd /root/autodl-tmp/SIR-CDR
nohup python -u experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml \
  > logs/prepare_amazon.log 2>&1 &
echo $! > logs/prepare_amazon.pid
```

- 首次运行会下载 4 个 Amazon Reviews 2014 gzip；已下载完成后加 `--skip-download`
- 输出 `data/processed/sports_to_clothing/`，**不调用任何 embedding API**
- 验证：`data/processed/sports_to_clothing/mappings.json` 与三个 jsonl 存在且行数 > 0

## 5. 阶段二：Qwen3-Embedding-8B 嵌入

```bash
export QWEN3_EMBEDDING_MODEL_PATH=/root/autodl-tmp/models/Qwen3-Embedding-8B
# 未下载则：huggingface_hub.snapshot_download(repo_id="Qwen/Qwen3-Embedding-8B", local_dir=上述路径)

cd /root/autodl-tmp/SIR-CDR
python experiments/embed_items.py --config configs/amazon_sports_clothing.yaml --smoke-test
# 预期输出 Provider: qwen3_local / Model: Qwen/Qwen3-Embedding-8B / Dimension: 768

nohup python -u experiments/embed_items.py \
  --config configs/amazon_sports_clothing.yaml \
  > logs/embed_qwen3_8b.log 2>&1 &
echo $! > logs/embed_qwen3_8b.pid
```

- 3090 上 batch=1，全量商品需数小时；任务可断点续跑（进度身份校验）
- 验证：`artifacts/embeddings/qwen3_8b/sports_to_clothing/` 下 manifest + tensor
- 若换过 `max_sequence_length` 或换模型目录，必须换新输出目录重跑，不能混用分块

## 6. 阶段三：Semantic ID tokenizer

```bash
cd /root/autodl-tmp/SIR-CDR
nohup python -u experiments/train_tokenizer.py \
  --config configs/amazon_sports_clothing.yaml --device cuda \
  > logs/train_tokenizer_qwen3_8b.log 2>&1 &
echo $! > logs/train_tokenizer_qwen3_8b.pid
```

- 先生成域自适应连续表征，再按 Semantic ID 每个位置拟合独立残差 K-means 码本
- 只有碰撞率与各级码本利用率全部过质量门才标记完成
- **旧共享码本 schema 产物必须重跑一次**：加 `--force`（README 有说明）
- 验证：`artifacts/tokenizer/qwen3_8b/sports_to_clothing/quality_report.json` 为 `{"passed": true}`，
  `item_latents.pt` / `semantic_ids.pt` / `tokenizer.pt` 存在

## 7. 阶段四：v3 主训练（新架构 full）

```bash
cd /root/autodl-tmp/SIR-CDR
nohup python -u experiments/run_text_cdr.py \
  --config configs/text_sports_clothing.yaml \
  --action train --variant full --seeds 42 --device cuda \
  > logs/text_cdr_v3_full_seed42.log 2>&1 &
echo $! > logs/text_cdr_v3_full.pid
tail -f logs/text_cdr_v3_full_seed42.log
```

- 默认启用：`prototype_enabled / cd_injector_enabled / sp_injector_enabled /
  codebook_summary_enabled / contrastive_alignment`，`lsep_weight=0.01`、`proto_orth_weight=0.01`
- 训练日志应出现 `lsep` 与 `proto_orth` 两项损失（>0），验证新组件已生效
- 输出 `artifacts/text_cdr/qwen3_8b/sports_to_clothing/full_seed42/`，独立于旧产物
- 早停 NDCG@10，patience=8；支持断点续跑（同一命令重跑即可）
- 改代码或配置必须换 `--output-root`，禁止覆盖已完成基线

## 8. 阶段五：评测（4 种推理模式 + 验证集融合扫参）

```bash
cd /root/autodl-tmp/SIR-CDR
for m in retrieval hybrid generate exhaustive; do
  python -u experiments/run_text_cdr.py --action evaluate \
    --config configs/text_sports_clothing.yaml \
    --split validation --mode $m --device cuda
done
```

- 四种模式分别命名报告，不混称为同一协议
- 选参只看验证集；确认候选池后可用融合扫参（只读验证集，不改参数）：

```bash
nohup python -u experiments/sweep_text_fusion.py \
  --config configs/text_sports_clothing.yaml \
  --variant full --seed 42 --device cuda \
  --weights 0 0.1 0.25 0.5 1 2 --batch-size 16 \
  > logs/sweep_text_fusion_v3.log 2>&1 &
```

- 选定推理设置后，最后只跑一次测试集：

```bash
python -u experiments/run_text_cdr.py --action evaluate \
  --config configs/text_sports_clothing.yaml \
  --split test --mode hybrid --device cuda   # mode 以验证集选定为准
```

## 9. 阶段六：消融套件与汇总

```bash
cd /root/autodl-tmp/SIR-CDR
nohup python -u experiments/run_text_cdr.py --action suite \
  --config configs/text_sports_clothing.yaml \
  --variants full no_graph no_reasoning single_step no_feedback no_semantic no_cpf_loss \
    target_only no_proto no_cd_inj no_sp_inj no_lsep no_lsh no_csum v2_legacy \
  --seeds 42 43 44 --device cuda \
  > logs/text_cdr_v3_suite.log 2>&1 &
echo $! > logs/text_cdr_v3_suite.pid

python experiments/summarize_text_cdr.py \
  --root artifacts/text_cdr/qwen3_8b/sports_to_clothing
```

- `v2_legacy` = 旧的单门混合接线，作为"无 prototype / 无注入"的天然基线
- 每个变体独立签名哈希，重复运行自动复用已完成结果
- 结果按 R@5/N@5/R@10/N@10 报告，先看验证集，后动测试集

## 10. 常见问题

| 现象 | 处理 |
|---|---|
| `Permission denied (publickey)` | 阶段一未完成认证；重复方式 A |
| CUDA OOM（嵌入） | 保持 batch=1；检查 `max_sequence_length`，换新输出目录重跑 |
| tokenizer 质量门不过 | 查看 `quality_report.json` 具体项；码本坍缩可 `--force` 重跑 |
| 训练报 "Completed run inputs/config/code changed" | 换新 `--output-root` |
| 训练中断 | 原命令重跑即恢复（checkpoint 含 RNG/优化器/调度器） |
| 评测与训练模式不一致 | 报告里必须注明 inference mode（retrieval/hybrid/exhaustive/generate） |

## 11. 论文表述红线

- 单元测试只证明正确性，不证明推荐质量；所有结论以真实验证/测试指标为准
- "隐式推理/双结构注入有效"的表述必须有受控消融支撑（阶段九）
- 多次观察过的测试集不得描述为"全新未接触"
