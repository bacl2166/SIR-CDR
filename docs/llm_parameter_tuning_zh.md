# 保持 SIR-CDR 架构的验证集参数调整

本工具只调整已有 model.rank 的候选数、生成/检索分数比例和校准方法，
不增加ItemKNN/RRF模块，不修改模型参数、训练目标、SID或评价指标。
复用现有最佳checkpoint与第一阶段验证缓存。第三组暂缓。

## 搜索预算与四指标保护

第一阶段21设置已经完成。本阶段固定17设置，累计38设置，是顺序验证探索：

- 原运行配置1个作为anchor（Sports为500候选、生成权重0.5、zscore）。
- none校准10个：500/1000候选，生成权重0.1/0.5/1/2/4。
- zscore校准6个：500/1000候选，生成权重0.375/0.625/0.75。
- 检索权重固定1；不搜索推理温度，避免zscore下冗余的正比例缩放。

只有R@5、R@10、N@5、N@10均不低于anchor（仅1e-12舍入余量）的结果才可
入选；其间依次按N@10、R@10、N@5、R@5择优。同分保留anchor，之后偏好
较小候选池和较小生成权重。全部设置均记录，不删除无效或变差结果。
若没有无主指标下降的提升，返回anchor，不承诺参数搜索一定有收益。

单正例R@K=HR@K，N@K=NDCG@K；命中位置r贡献1/log2(r+1)。
原对照表中的0是“差值为0”，R@10实际为0.0226425，并非命中率0。

## 原一致性失败如何处理

原tune_llm_fusion.py及其缓存/报告不修改、不覆盖；其旧GPU model.rank验证
曾报告N@20差异1.782501e-6。此处不声称修复或通过了那次检查。

新工具对缓存和最终推荐统一使用现有原始评分路径：验证时重新加载同一
模型，固定一次目录物品状态，逐用户计算需求与检索/生成分数，然后按原
参数规则排序。没有改变网络结构或评价数学定义。它验证的是这个统一路径
的缓存/重算指标一致性，而非证明旧逐用户重算目录的GPU路径逐位相同。

所有R/N截断仅允许1e-12舍入误差；失败仍阻止测试。审计保存正例变化位置、
两份topK、相关原始分数及旧直接验证报告。ranking_mismatch_rows公开，
指标相同不能称为完整推荐列表完全相同。

## 服务器顺序

先确认没有旧评测/训练进程；尤其不能同时运行两个最终测试入口。

```bash
cd /root/autodl-tmp/SIR-CDR
pgrep -af 'python.*experiments/'
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
```

CPU搜索，无需Qwen前向或重新训练：

```bash
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_sports_clothing.yaml --cache-root artifacts/tuning/llm_sports_seed42 --output artifacts/tuning/llm_sports_parameters_seed42 --action sweep --device cpu
```

结果位于新目录search.json；improved_without_primary_drop表明是否有至少一项
主指标提高且其余主指标不下降。selected.yaml仅记录推理参数，原训练配置不改。

选定设置的直接指标验证（GPU重新评分一次）：

```bash
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_sports_clothing.yaml --cache-root artifacts/tuning/llm_sports_seed42 --output artifacts/tuning/llm_sports_parameters_seed42 --action verify --device cuda
```

verified_all_metrics=true后固定设置，再评测测试集：

```bash
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_sports_clothing.yaml --cache-root artifacts/tuning/llm_sports_seed42 --output artifacts/tuning/llm_sports_parameters_seed42 --action test --device cuda
```

最终测试和验证使用同一评分/参数函数。测试前绑定选中配置、搜索报告哈希及
验证报告哈希，独占预留原模型目录test.json，并写test.lock；后来启动的旧
入口也会因不匹配身份拒绝读取。预留不能阻止已经越过旧检查的进程，因此
必须先确保没有旧测试进程。失败保留failed预留记录，不静默删除测试身份。
失败恢复时只允许相同冻结配置，需先检查日志，不得藉删除记录再调参。

Phones在完成训练并生成自己的第一阶段缓存后，使用对应配置及独立的调参
目录执行同一流程。若本阶段没有提升，保留原设置，之后再设计独立的训练
实验；不能看过测试集后无限扩大搜索范围。
