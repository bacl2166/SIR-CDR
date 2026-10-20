# 三个反向任务的并行资产准备

本次仅准备 Clothing→Sports、Electronics→Phones、Douban Movie→Book 的数据、
向量和 tokenizer 资产。独立输出目录分别为 `clothing_to_sports`、
`electronics_to_phones`、`douban_movie_book`。当前正向 Douban 训练可继续。

## 1. 规则与隔离

三个任务直接使用原 Amazon 的 `build_temporal_splits`：共有用户、两域各至少
5 条有效交互、保留重复交互、按时间/原始物品 ID 排序、末次目标交互测试、
倒数第二次验证，源历史严格早于预测时点。没有通过交换正向 JSON 的两列伪造反向数据。

2026-10-20 本地真实 Douban 反向转换得到：训练 215,651 条、验证 1,841 条、测试 1,849 条，
用户 1,859，Movie 源物品 34,046，Book 目标物品 95,525；各 split 已见正例均为 0。
源历史不足的样本按既有规则跳过并计数，没有挑选新的留出正例。该记录不代表服务器已运行。
反向训练、验证、测试和目标目录重新生成，完整目标历史供已见过滤，编码器仍可截取 50 条。

Amazon 复用原始压缩文件和原元数据文本准入条件；没有新增下载。缺文件会列出路径并停止。
Douban 读取原始固定哈希 ZIP，正确连接 UID；Movie 静态字段与正向相同。
Book 标签只来自反向训练样本可用的目标历史和训练正例，最后两个留出评论不贡献标签。
训练正例是已经观测的监督数据，其评论标签可用；不将“只限训练历史”与这一规则混淆。
损坏标签仍排除，无标签时用明确的原始 ID 后备。重复留出正例保留并计数。

正向处理器、训练/评测核心、配置和原资产不修改，因此不会改变正在运行的正向训练身份。
新处理器及原处理原语、原始文件均有 SHA 身份；已有反向输出只在哈希一致时复用，
不会 force 删除或覆盖身份冲突的目录。协议仍是自建共同协议，未验证为 GenCDR 作者最终划分。

## 2. 向量复用规则

以域、原始物品 ID 和重新生成后的完整文本逐项匹配，不能按新旧行号直接复制。
同时校验旧文本与旧 embedding manifest 的绑定、目录映射和整个旧向量文件 SHA。

- 完整匹配的 Amazon 矩阵在 CPU 上一次性重排复制，不再调用 8B 模型。
- 旧 Amazon 向量可能未保存模型 commit。此时明确记录
  `legacy-frozen-matrix-unrecorded-model-revision`：复用的是同一个冻结向量资产，
  不声称恢复出了原模型版本。只允许整套复制，不能与新版本算出的向量静默混合。
- 有已记录的模型 commit 时，允许相同 commit、维度、归一化和文本截断设置下的部分复用。
  Douban 可复用的电影静态向量及文本完全相同的书籍向量符合此规则。
- 缺失模型版本且文本不完全匹配时，放弃部分复用，整个矩阵交给固定模型版本重新编码。
- 新编码必须先固定模型 commit，保存版本回执，再按身份绑定的 chunks 断点续算。
  不同输入、模型版本或源资产不能混用缓存，最终矩阵保存 SHA。

新编码还绑定当前 provider 代码和 torch/transformers/sentence-transformers 版本，
防止环境变化后把不同实现的新向量混入未完成 chunks；可复用源模型的 config SHA 也会核对。

全套向量复制解决已有资产的逐字节可追溯性，但不能补齐旧模型版本缺失。
论文中应披露该来源限制；若要求从模型权重完全独立重建，需另立固定版本的新向量实验。

## 3. 当前服务器立即执行：三个 CPU 任务并行

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
mkdir -p logs
LOG="logs/reverse_assets_cpu_$(date +%Y%m%d-%H%M%S).log"
nohup python -u experiments/prepare_reverse_assets.py --action cpu --workers 3 --cpu-tokenizer > "$LOG" 2>&1 &
echo $! > logs/reverse_assets_cpu.pid
echo "$LOG" > logs/last_reverse_assets_log
tail -f "$LOG"
```

三个独立子进程各准备一组，CPU 数值库线程限制为 1。原始文件解析、数据切分、
文本审计与向量重排都在 CPU 上完成，`--cpu-tokenizer` 还会为完整复制了向量的任务
在 CPU 上训练自己的 tokenizer。默认语义训练仍最多 100 轮、4 层 512 码、128 维，
保留碰撞率与码本使用率门禁。CPU 与 GPU tokenizer 的设备来源会单独记录。
CPU/磁盘仍与正向任务共享；若正向吞吐明显受影响，下次可降低 `--workers`，不重复启动第二个控制器。

未带 `--cpu-tokenizer` 时，CPU 阶段只完成划分、文本和可直接复用的向量；SID 后续再准备。
某个任务失败不丢弃其他任务已完成结果，父日志与摘要记录每任务退出码。
CPU 阶段结束不等于三组全部资产准备完成，特别是 Douban 新编码和 SID 可能仍待处理。

## 4. 查看三个任务进度

```bash
python experiments/prepare_reverse_assets.py --action status
tail -n 8 logs/reverse_assets/clothing_to_sports_cpu.log
tail -n 8 logs/reverse_assets/electronics_to_phones_cpu.log
tail -n 8 logs/reverse_assets/douban_movie_book_cpu.log
```

`processed=true` 表示已生成完整处理清单；reuse_plan 显示可复制/待编码数量；
embedding.complete 表示完整向量矩阵完成；tokenizer.passed=true 才表示 SID 通过质量门禁。
状态摘要只快速读取完成标记，正式执行阶段才完整核对文件 SHA。

进程退出或服务器重启后，重新执行相同 CPU 命令会复用已完成处理与向量、恢复 tokenizer。
未完成 CPU tokenizer 应继续用 `--cpu-tokenizer` 恢复，不要半途切换 GPU 设备。

## 5. 正向推荐流程结束后，补齐 GPU 阶段

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
LOG="logs/reverse_assets_gpu_$(date +%Y%m%d-%H%M%S).log"
nohup python -u experiments/prepare_reverse_assets.py --action gpu > "$LOG" 2>&1 &
echo $! > logs/reverse_assets_gpu.pid
echo "$LOG" > logs/last_reverse_assets_gpu_log
tail -f "$LOG"
```

该阶段逐组执行，固定 8B 版本补齐缺失向量，再训练相应 tokenizer。完整 CPU tokenizer
会按原身份复用。不会启动 0.6B 推荐训练，也不会评测测试集。
当检测到正向控制器/训练/评测仍在运行，或 GPU 存在其他计算进程时，会安全停止；
它不会自动终止当前任务，也不会排队后自行抢卡。单卡目前优先使用 CPU 命令。

## 6. 产物与验证范围

各 task 使用以下独立位置：

- `data/processed/<task>/`：三份反向划分、映射、文本、协议与 reverse_manifest。
- `artifacts/embeddings/qwen3_8b/<task>/`：reuse_plan、chunks、向量、进度及 manifest。
- `artifacts/tokenizer/qwen3_8b/<task>/`：独立 tokenizer、SID、latents 和质量报告。
- `artifacts/reverse_assets/<task>/`：操作锁、模型回执和 tokenizer 执行身份。
- `logs/reverse_assets/<task>_cpu.log`、`<task>_gpu.log`：逐任务日志。

已做有针对性的 CPU 检查：原始 Amazon 反向留出、Douban 留出与标签隔离、域身份安全
重排、旧版本整套复制与混合隔离、新编码断点恢复、向量哈希核验、并行调度和忙碌 GPU 门禁。
这些检查不代表三个真实服务器任务已完成，不代表 GPU 上的模型编码和 tokenizer 已通过。
