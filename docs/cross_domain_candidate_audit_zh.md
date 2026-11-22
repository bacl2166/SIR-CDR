# Food–Kitchen 与 Beauty–Electronics 实验规模审计

以当前 Phones–Electronics 的 13,682 个共同用户、151,685 与 58,948 条方向性训练样本为参照，先考察以下两组 Amazon2014 跨域数据。

| 公开 ABXI 基准 | 共同用户 | 两域物品数 | 两域交互数 |
|---|---:|---:|---:|
| Food–Kitchen | 7,144 | 11,837 / 16,258 | 83,663 / 89,885 |
| Beauty–Electronics | 4,474 | 10,379 / 14,188 | 50,329 / 63,800 |

上述数值引用 ABXI 的处理协议。当前 SIR-CDR 训练样本采用每域留出末两项、严格源域时间可见性和训练前缀展开，需要重新计算实际规模。公开交互数与当前训练条目分别列示。

原始分类分别为 Grocery_and_Gourmet_Food、Home_and_Kitchen、Beauty、Electronics。审计工具复用当前 Amazon2014 的 Electronics 原始文件，下载其余缺失的 5-core 评论文件，记录每个文件的 SHA-256、有效共同用户、各方向训练/验证/测试条目、重复正例及跨域重复 ASIN。

```bash
cd /Data/lihb/SIR-CDR-candidate-audit
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
/Data/lihb/envs/sir-cdr/bin/python -u experiments/audit_cross_domain_candidates.py \
  --root /Data/lihb/SIR-CDR --download
```

报告写入原实验根目录下 artifacts/diagnostics/cross-domain-candidate-audit-v1/audit-时间戳.json。

后续接入四个方向 food_to_kitchen、kitchen_to_food、beauty_to_electronics、electronics_to_beauty，先根据实际样本量制定预算，再构建静态物品文本、嵌入和各方向 SID。初始 SIR-CDR 配方使用原 Phones–Electronics 参数，seed42 冒烟验证通过后依次开展 42、43、44 三种子训练与验证融合。选参规则固定为验证 NDCG@5 和 NDCG@10 的等权平均，HR 用于破同分，最终报告全部四项指标。各方向独立选参且使用相同搜索预算。

ABXI 使用 999 个采样负例评估，当前 SIR-CDR 使用完整目标目录。比较算法效果时，需要让参照算法消费同一份划分和候选目录。GenCDR 原论文没有这两个新数据对的对应表值。新增数据上的 SIR-CDR 初始实验与算法基线对照分别记录，避免混用公开表值。

来源

- ABXI 原论文及表 1 https://arxiv.org/html/2501.15118v1
- ABXI 官方代码和处理入口 https://github.com/DiMarzioBian/ABXI
- Amazon2014 原始数据 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html

本次交付包含下载和规模审计入口。真实规模报告、静态文本资产、嵌入、SID、新方向训练和算法基线对照依次在后续阶段执行。
