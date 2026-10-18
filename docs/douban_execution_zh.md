# 第三组 Douban Book → Movie：资产准备与推荐运行

## 当前实现范围

新入口 `experiments/run_douban.py` 单独运行第三组，不调用前两组训练，不使用
Amazon Books–Movies 或 GenCDSR 数据。原始包固定为发布者 Douban Version 1，
SHA256 为 `b1571ebf29c875f5c13bdc1c1eaae092c507c099c2ef62a050f42aab867d10be`。

### 数据划分与前两组对齐

2026-10-18 本地真实数据转换：共有用户 1,859，Book 95,525 个、Movie 34,046 个，
训练 918,465 条、验证 1,855 条、测试 1,858 条。验证重复已见正例 0、测试 144；
这 144 条保留且按既有规则屏蔽，不能据此把测试条数改为 1,714。
Book 文本中 64,359 个有干净训练标签，31,166 个只有 ID 后备文本。

| 规则 | 前两组 Amazon 与本次 Douban |
|---|---|
| 用户 | 两域均至少 5 次有效交互的共有用户 |
| 排序 | 时间、域内原始物品 ID；完全同键时保持原始记录顺序 |
| 留出 | 每用户目标域末次测试、倒数第二次验证 |
| 训练 | 此前所有目标域前缀预测样本；没有删除或下采样训练目标 |
| 源历史 | 严格早于该样本预测时点 |
| 空历史 | 与既有实现相同，跳过该样本并计数，不把留出正例换成其他物品 |
| 重复交互 | 保留，不做 Douban 特有的去重 |
| 已见过滤 | 使用完整目标历史；重复留出正例也不单独放回候选，报告 positives_in_seen |
| 指标 | 沿用原单正例 HR/R 与 NDCG/N 定义，K=5/10/20 |
| 目录 | 完整目标物品目录，包含只出现在留出集的物品；转导式设定 |

测试历史包含验证交互，训练图仍只来自训练样本。编码器继续截取最近 50 条，
已见过滤使用完整历史。Douban 只有日历日期，统一转为 UTC 零点秒数，
同日源交互不视为已经可见。不把缺失日期填成零。

新转换器的样本与现有 Amazon `build_temporal_splits` 做逐字段一致性检查，
覆盖重复交互和重复测试正例。这里只对齐**划分规则**，并未证明论文最终
用户/物品集合已对齐。重复正例被已见过滤屏蔽的规则沿用旧实验，应在比较中
公开其数量；若将来改为重复消费推荐，必须为各组和各方法另立共同协议，
不能只对 Douban 特殊处理或重新解释现有指标。

### UID 映射与文本

原始评论的 user_id/book_id/movie_id 使用目录 UID，先验证一对一映射后转成
原始用户/物品 ID；Book 与 Movie 的同一数字不合并。映射保留原始 ID 和 UID。
错误映射、缺失日期、非共有用户与空历史均有审计清单。

书籍目录只有 book_id/UID，不能假装存在书名或简介。书籍文本为原始目录 ID
加训练样本实际可见源交互的干净 labels；含 U+FFFD 的损坏标签不用，最多保留
30 个标签，无标签时明确为 ID-only 后备。验证/测试才可见的评论不贡献标签。
电影仅取静态 name/director/writer/summary/country/pubdate/language，排除评论、
聚合 tag、评分和评分人数。每物品保存文本来源及训练源行号。

ID-only 文本不等同于真实语义描述，可能限制推荐质量。文件中报告其覆盖率，
不能为了提高指标补入留出评论或编造文本。tokenizer 质量门禁失败会停止任务。

## 执行流程

1. 固定哈希的原始 ZIP → UID 校验 → 时间划分、文本、排除清单与协议指纹。
2. Qwen3-Embedding-8B：优先使用已有缓存，将实际模型 commit 固定进单独回执。
   新资产身份包含模型版本、输入文本和协议 SHA。批次 2，最多 512 token；
   输出仍为 768 维归一化向量。与其他实验的文本长度/批次差异须披露。
3. 域自适应 tokenizer：显式 Book/Movie 域，4 层、每层 512 码、128 维。
   复用现有训练器及碰撞率/使用率门禁，不修改 Amazon 专用配置类。
4. Qwen3-0.6B 用户侧：LoRA、结构投影、两步记忆、CPF 保留。实际批次 8、
   累积 2、验证批次 8，关闭梯度检查点；名义有效批次 16。它是独立训练设置，
   CPF 批统计可能变化，不能声称与批次 4 完全等价。最多 30 轮，patience=5，
   验证 NDCG@10 选 checkpoint；无训练样本抽样。
5. 已完成训练严格复用、未完成训练按原 checkpoint 恢复。随后使用已有目录
   快照评测：固定 CPU FP32 物品目录、GPU FP32 用户评分、验证选参、独立精确
   核验，再锁定一次测试。原 R/N 指标不变，不放宽核验阈值。

每个大阶段独立进程，完成即释放显存。已有资产按身份复用，不反复执行 smoke。
准备阶段只做一次完整 preflight；训练首批的有限损失/梯度检查由现有训练器负责。
运行入口有操作锁，不能重复启动同一任务。源代码/数据/配置身份冲突会拒绝复用，
不会自动删除产物。原始数据、向量和模型留在本地/服务器，不提交 GitHub。

## 服务器命令

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
LOG="logs/douban_$(date +%Y%m%d-%H%M%S).log"
nohup python -u experiments/run_douban.py --action all > "$LOG" 2>&1 &
echo $! > logs/douban.pid
echo "$LOG" > logs/last_douban_log
tail -f "$LOG"
```

仅准备资产：`python -u experiments/run_douban.py --action prepare`。
仅转换原始数据：`python -u experiments/run_douban.py --action convert`。
资产完成后继续训练/评测：`python -u experiments/run_douban.py --action run`。
原始数据已转换但其他资产未完成时，重复 all 会严格复用已完成阶段。

状态摘要：

```bash
python experiments/run_douban.py --action status
pgrep -af '[p]ython.*experiments/'
tail -n 12 "$(cat logs/last_douban_log)"
```

主要产物：

- `data/processed/douban_book_movie/`：mappings、三份 split、item_texts、protocol、statistics、manifest，以及排除和文本来源清单。
- `artifacts/embeddings/qwen3_8b/douban_book_movie/`：向量及进度。
- `artifacts/tokenizer/qwen3_8b/douban_book_movie/`：tokenizer、SID、latents、质量报告。
- `artifacts/llm_cdr_v1/qwen3_0_6b_douban_b8_eb8_gc0/douban_book_movie/seed42/`：训练记录及最终 test.json。
- `artifacts/tuning/douban_book_movie_qwen06_fast_seed42_snapshot_eval_v1/`：验证选参、目录快照与精确核验。

## 与 GenCDR 的比较边界

原论文采用 Douban 和 leave-last-out，但目前没有作者最终筛选/划分资产的验证，
因此标记 `paper_alignment_verified=false`。当前结果只属于本仓库可复现的
共有用户协议；严谨基线需让 GenCDR 在同样映射、样本、文本可见范围、候选目录
与指标下共同重跑。论文表格只能另列作参考。Book→Movie 仅对应 Movies 行；
Books 行需要独立反向实验，不能从本任务推导。

目前本地只完成 CPU 预处理及针对性检查，预训练模型资产、训练与最终推荐指标
须在服务器实际执行后产生，不宣称本地已完成第三组推荐实验。
