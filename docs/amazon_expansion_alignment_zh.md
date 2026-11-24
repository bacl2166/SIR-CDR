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

## 对齐成功后的 GPU 资产生成

prepare_amazon_assets.py 读取 Phones 的真实 embedding_model.json、tokenizer_execution.json 和已封存的资产链，将新目录的嵌入模型固定到原 revision，并复用原 embedding、tokenize、seal 函数。新数据对仅在进程内临时注册，原始六方向代码与文件身份继续沿用现有版本。嵌入采用 768 维、归一化、512 token 截断、batch size 2；SID 参数必须与 Phones 实际训练记录一致。

```bash
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/Data/lihb/envs/sir-cdr/bin/python -u experiments/prepare_amazon_assets.py \
  --root /Data/lihb/SIR-CDR --max-workers 2 \
  --alignment /Data/lihb/SIR-CDR/artifacts/diagnostics/amazon-expansion-phones-alignment-v1/alignment-20261123T092552327241Z.json
```

自动筛选无计算进程、利用率至多 5%、至少空闲 24 GiB 的 GPU。每个方向按原协议路径单独生成 embedding 和 tokenizer；两个阶段在不同子进程运行，阶段切换时释放模型。最多并行两个方向，并沿用原调度器的主机内存和 CPU 压力检查。已完成目录通过 verify_ready 后跳过；未完成任务沿用自己的 asset_gpu.json 绑卡以便恢复。没有满足条件的卡时输出 WAIT_GPU。重新运行会刷新空闲卡列表，当前一轮的可用卡列表在启动时确定。

单方向完成时打印 ASSETS_READY，四方向全部封存后打印 AMAZON_ASSETS_COMPLETE。各方向日志在 /Data/lihb/SIR-CDR/logs/amazon-expansion-assets-v1/，控制状态在 artifacts/diagnostics/amazon-expansion-assets-v1/。本地验证覆盖参数绑定、原函数调用和恢复检查；真实 8B 编码及 SID 质量门槛需要服务器执行验证。

每个方向绑定本次 ALIGNMENT_OK 报告中的数据 manifest 哈希，并核对源/目标领域角色。复用已封存资产时会重新核对实际模型 revision 及 tokenizer 训练参数。两个子进程每次启动前重新检查卡占用。GPU 候选池最多传入六张卡以兼容原调度器，并优先保留待续跑任务的原绑卡。

## 阶段切换遇到占卡后的续跑

2026-11-24 的服务器输出显示，Food→Kitchen 已完成 15,265/15,265 批嵌入，并写出最终 manifest，但原调度器在切换到 tokenize 前发现 4 号卡被占用而退出。此次新增 continue_amazon_assets.py 作为续跑入口，保留 prepare_amazon_assets.py 及原模型、数据和执行配方的文件身份。

```bash
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/Data/lihb/envs/sir-cdr/bin/python -u experiments/continue_amazon_assets.py \
  --root /Data/lihb/SIR-CDR --max-workers 2 --poll-seconds 60 \
  --alignment /Data/lihb/SIR-CDR/artifacts/diagnostics/amazon-expansion-phones-alignment-v1/alignment-20261123T092552327241Z.json
```

入口校验完整嵌入的输入、数据及代码身份、模型 revision/config、批次数、矩阵形状、float32 类型、有限值、归一化、零填充行、物品索引和全部缓存分块后，直接进入 SID 训练，不再调用嵌入模型。校验每完成 1,000 个分块显示 VERIFY_EMBEDDING。向量文件 SHA-256 随任务保存，启动 SID 前和封存前复核，并核对 tokenizer 执行收据所记录的输入哈希。缺少最终 manifest 的嵌入继续使用原断点恢复逻辑。已封存方向通过原 verify_ready 校验后跳过。

每次派发重新查询全部 GPU；要求无计算进程、利用率不超过 5%、空闲显存至少 24 GiB，最多并行两个阶段，并检查 CPU 与主机内存。阶段间允许选择不同物理卡；资源不足时每 60 秒重新查询，启动前发现卡已被占用则重新排队。实际子进程失败时保留错误、停止新增任务，并等待正在运行的其他任务结束后退出。

原 asset_gpu.json 和 expansion_recipe.json 保留原内容。每次实际阶段执行在各方向控制目录 stage_executions/ 保存 GPU UUID、时间、原绑卡记录哈希、续跑脚本哈希和执行结果。新的阶段状态在 artifacts/diagnostics/amazon-expansion-assets-v1/continuation_status/；旧 status/ 中的历史失败保留。方向日志继续追加到 logs/amazon-expansion-assets-v1/。此入口需要后台运行时可用 nohup 启动，无需额外 shell 重试循环。
