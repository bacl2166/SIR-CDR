# 新 Amazon 数据对沿用 Phones–Electronics 的处理协议

本次以原实验根目录下 data/processed/gencdr-shared-loo-v1/phones_to_electronics 为实际参考，复用 gencdr_inputs.raw_amazon 与 gencdr_protocol.build_shared。四个新方向分别写入 food_to_kitchen、kitchen_to_food、beauty_to_electronics 和 electronics_to_beauty。

| 处理项 | 沿用的实际规则 |
|---|---|
| 原始数据 | Amazon2014 5-core 评论及同类别元数据 |
| 用户 | 两域均至少 5 条有效交互的共同用户 |
| 交互次序 | 时间戳、原始物品 ID、事件键依次排序，保留重复交互 |
| 留出 | 每域最后一项测试，倒数第二项验证 |
| 源域可见性 | 仅源域训练前缀，时间严格早于目标标签 |
| 训练样本 | 展开全部有合格历史的训练前缀 |
| 商品文本 | 原函数拼接标题、品牌、类别、特征、描述，最多 12,000 字符 |
| 缺元数据 | 保留交互，回退为领域与物品标识文本 |
| 目录 | 保留用户涉及的全目录，含仅出现在留出中的物品 |
| 编码和屏蔽 | 后续编码最近 50 项，排序屏蔽完整已见历史 |

脚本逐项核对新数据 protocol.json 除数据身份外的所有规则，要求与 Phones 参考完全一致。全部评论文件的 SHA-256 必须对应已完成的候选审计，原 Phones–Electronics 的评论和元数据也必须对应原协议记录。转换后核对四方向的用户数、物品数、训练/验证/测试条目和重复正例数。新报告保存参考 manifest、候选审计和脚本哈希。

## 服务器执行

从 GitHub 拉取 codex/cross-domain-candidate-audit 的最新提交并在独立目录检出。确保此前 metadata 下载进程已结束，然后在该目录执行以下命令。

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/Data/lihb/envs/sir-cdr/bin/python -u experiments/prepare_amazon_expansion.py \
  --root /Data/lihb/SIR-CDR \
  --audit /Data/lihb/SIR-CDR/artifacts/diagnostics/cross-domain-candidate-audit-v1/audit-20261123T041006661933Z.json \
  --download-metadata
```

结束时显示 ALIGNMENT_OK 和报告路径。每个新方向包含 train.jsonl、validation.jsonl、test.jsonl、item_texts.jsonl、mappings.json、statistics.json、protocol.json、事件及排除样本审计和 shared_manifest.json。重复执行会核验并复用相同身份的处理结果。

这一步使用 CPU 完成元数据下载、文本构建和样本划分。后续物品向量和 SID 按 Phones 的嵌入模型 revision、768 维输出、512 token 截断、归一化和 tokenizer 配方分别生成。各数据对的共同用户与物品目录不同，因此同名 Electronics 的物品映射与 SID 仍需核对，已有 Phones 的整套 SID 无法直接套用到新目录。

本次四方向的规模依据候选审计，分别为 71,983、64,394、65,418、42,714 条训练样本。真实服务器上的对齐状态以完成报告为准。

## 元数据网络恢复

新增入口的元数据下载使用 60 秒连接超时和 300 秒读取超时，最多尝试 6 次。中断时保存 .part 文件，下次按实际字节数请求续传。服务器忽略 Range 时重新下载该文件，Content-Range 不匹配时停止接受该响应，gzip 完整性通过后才发布正式文件。日志显示每次重试、续传位置及每约 16 MiB 的字节进度。重试耗尽后保留尚存的部分文件，重新执行相同入口继续恢复。
