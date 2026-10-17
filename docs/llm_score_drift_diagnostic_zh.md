# 已训练模型的验证分数漂移：短诊断

2026-10-17 服务器现象：Phones 已完成 15 轮训练，验证调参选定
`pool=1000, weight=0.5, normalization=none`，但重新评分时 12 条验证样本
的正例排名变化。R@10 净差一个样本，最大已展示检索分差约 0.0077。
这不是将显示精度四舍五入即可忽略的问题，最终测试仍被原有校验拦截。

当前尚不能把原因认定为特定 CUDA 算子、关机、缓存错误或训练随机性。
同环境 GPU 计算存在非确定性来源，见
[PyTorch 2.6 重现性说明](https://docs.pytorch.org/docs/2.6/notes/randomness.html)
和 [SDPA 后端说明](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html)。
候选问题需要实际 GPU 分层比较，不能仅提高校验容差让任务通过。

## 只做一次短定位

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
python -u experiments/diagnose_llm_scoring.py --output "logs/scoring_diagnostic_$(date +%Y%m%d-%H%M%S).json"
```

默认读取当前 Phones b4/eb4/gc0 runtime、验证缓存和失败的 verification，
只检查前三条异常样本所在的原始批次，保留原来的批次边界和历史 padding。
不重新训练，不评测整个验证集或测试集，不保存模型，不修改原缓存、参数
搜索报告、测试锁。已有训练身份、库版本、资产及缓存哈希仍须严格匹配。
本次只新增诊断代码/测试/说明，既有训练和调参代码均未改动，不引入身份豁免。

每个独立子进程分别运行：

- **native：** 保持原模型执行精度，比较两次目录表示、固定目录下的用户
  编码及两次完整候选打分。记录 GRU、结构投影、Qwen 前向、记忆和最终
  query/prefix，便于找到最先产生差异的位置。
- **controlled：** 仅对临时模型以 FP32 执行加载后的权重，限定数学 SDPA、
  开启严格确定性检查、禁用 CuDNN 和 TF32，进程启动前设置 cuBLAS 工作区。
  若存在没有确定性实现的操作，保留实际错误，不静默降级。

终端打印每种设置的 `catalog_equal`、`encoding_equal`、`repeated_scores`。
完整合并报告和两个分报告在指定的 logs 目录；贴出两条汇总 JSON 即可先判断。
候选分数按物品 ID 对齐比较，避免候选排序变化造成虚假的“分差”。

## 如何解读

- `repeated_catalog.all_equal=false`：目录/图表示重复计算已产生差异。
- 目录相同、`fixed_catalog_encoding.all_equal=false`：进一步查看逐层张量，
  区分 GRU、结构信号、Qwen 与记忆。
- 编码相同、重复评分不同：继续检查评分路径、候选集合和排序。
- native 相同但与旧缓存不同：不能宣称已确定原因，仍需检查跨进程、
  执行环境和缓存采集上下文差异。
- controlled 重复一致：只表明该小样本下有稳定的候选执行方式；不证明
  全量修复，不把它当作新的正式指标，也不自动认可旧缓存。

确定修复后，应以明确记录的评估策略重建独立验证缓存、重新校验，再执行
锁定的最终测试。不能在两次结果之间挑选更高的一组，也不能把本次验证
数字作为测试指标。训练权重原则上不需要因评估数值漂移而从头训练。
