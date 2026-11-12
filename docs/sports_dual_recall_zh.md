# Sports ↔ Clothing 双路候选召回诊断

沿用 `sports-clothing-ndcg-v1` 已完成的训练检查点、现有数据划分和语义 ID。首轮复核最近已知的两个验证入围检查点 `sports_to_clothing/c06/seed42` 和 `clothing_to_sports/c03/seed43`。两者各做一次配对验证。

| 对照组 | 每个用户的候选来源 | 重排 |
|---|---|---|
| 检索候选 | 在完整目标域目录中取前 500 个未见物品 | 检查点记录的检索分数与生成分数融合 |
| 双路候选 | 先纳入最多 50 个目标域 SID 约束生成物品，合并去重，再按检索顺序补齐至 500 个 | 同一重排器、融合权重及归一化 |

生成使用现有目标域前缀树，beam size 固定为 50。两组使用同一物品快照、用户编码、已见物品屏蔽和稳定排序。目标标签只用于计算指标。候选缺失的正例仍计入验证分母。报告保存候选召回率、找回和丢失的正例数、NDCG@5、NDCG@10、HR@5、HR@10 及配对差值。耗时为批内模型计算，包含双路额外的生成和合并开销，排除加载和统计。

## 在 L20 服务器运行

先更新诊断代码。独立目录已存在时，可先核对其 Git 版本。以下命令默认诊断目录尚未创建。

```bash
(
set -eu
cd /Data/lihb/SIR-CDR
git fetch origin refs/heads/codex/sports-dual-recall
git worktree add --detach /Data/lihb/SIR-CDR-sports-dual-recall FETCH_HEAD
git -C /Data/lihb/SIR-CDR-sports-dual-recall rev-parse HEAD
)
```

检查 GPU 占用，并按课题组要求确认可以使用的卡。脚本启动时核对指定 UUID，要求无计算进程、利用率不高于 5%、至少 12 GiB 空闲显存，沿用 Sports 调参的 GPU 准入阈值。

```bash
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
```

将 `SIR_GPU_UUID` 设为刚确认的物理 GPU UUID。两个方向在一张卡上依次执行。

```bash
(
set -eu
SIR_GPU_UUID='GPU-替换为已确认空闲卡的UUID'
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd /Data/lihb/SIR-CDR-sports-dual-recall
for sir_job in sports_to_clothing/c06/seed42 clothing_to_sports/c03/seed43
do
  /Data/lihb/envs/sir-cdr/bin/python -u experiments/diagnose_sports_dual_recall.py \
    --root /Data/lihb/SIR-CDR --job "$sir_job" --gpu "$SIR_GPU_UUID"
done
)
```

每个方向只在成功后写出一份 `report.json`。报告位于
`/Data/lihb/SIR-CDR/artifacts/diagnostics/sports-dual-recall-v1/sports-clothing-ndcg-v1/<方向>/<参数组>/<种子>/`。
已有报告时直接读取它。如某个方向未写出报告，只对该方向重试。比较报告中 `dual_minus_retrieval` 和 `positive_candidate_changes`，以确认候选召回的变化是否转化为前 5 和前 10 的排序提升。这轮结果用于决定是否值得继续多种子验证。
