# 第三组以 GenCDR 为基线：Douban 资产与待解决差异

完整后续方案及整体实验可比性结论见
[Douban Book–Movie 资产设计](douban_book_movie_asset_design_zh.md)。

## 已纠正的基线身份

- 目标论文：[From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](https://arxiv.org/html/2511.08006v1)。
- 官方代码：[hupeiyu21/GenCDR](https://github.com/hupeiyu21/GenCDR)，核对版本
  `204b0ce0f624d04f84d0c72c52f4d62512ce4bdf`。
- `Applied-Machine-Learning-Lab/RecSys2026_GenCDSR` 是其他基线；其六个
  Book_Movie JSON 不作为本次 GenCDR 资产。其旧第三组数据、配置、适配器
  和专用测试已删除。第三组注册名、processed、embedding、tokenizer 和
  推荐输出目录统一采用 `douban_book_movie`，不重用旧数据改名冒充。
- 默认资产准备和 Text v4 基准入口现在只运行前两组。它们的处理配置、
  骨干配置、训练代码和已有模型没有更改。默认 preflight 通过仅代表两组。

## 原始数据已下载，最终样本尚未对齐

官方处理器的 `_get_file_paths` 指向 `douban_dataset(text information)`，
需要 bookreviews/books/moviereviews/movies/users 的 cleaned TSV 文件。
论文引用 Zhu 等的 DTCDR / GA-DTCDR 数据研究；其作者发布的
[Douban 原始数据](https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information)
含对应七个文件。来源与字段吻合，但不能据此认定最终筛选快照相同。

本地已下载 Version 1 并完成 ZIP CRC 检查：

```text
data/raw/douban_fengzhu_v1.zip
SHA256 b1571ebf29c875f5c13bdc1c1eaae092c507c099c2ef62a050f42aab867d10be
```

按公开处理器默认规则，仅保留每域至少 5 次交互的用户，得到：

| 域 | 本地用户 | 本地物品 | 本地交互 | 论文用户 | 论文物品 | 论文交互 |
|---|---:|---:|---:|---:|---:|---:|
| Books | 1,888 | 95,737 | 226,537 | 1,713 | 8,601 | 104,295 |
| Movies | 2,630 | 34,892 | 1,278,204 | 2,628 | 20,964 | 1,249,016 |

这是实际差异，不能通过随意采样凑出论文数字。完整审计还记录原始文件
SHA256、无效日期、同日交互、重复留出正例及共有用户数。它不是推荐指标，
也不是可直接训练的 processed 目录。报告始终标记
`paper_alignment_verified=false`、`recommendation_assets_ready=false`；即使
将来统计数字相同，也必须核对具体 ID 和样本内容。

## 服务器按次序操作

无需停止正在进行的前两组 GPU 任务。先同步已发布代码，再执行 CPU 数据审计：

```bash
cd /root/autodl-tmp/SIR-CDR
git config --local core.sshCommand "ssh -i /root/.ssh/id_ed25519_github_sircdr -o IdentitiesOnly=yes"
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
python experiments/prepare_gencdr_douban.py --action cleanup-legacy
mkdir -p logs
python -u experiments/prepare_gencdr_douban.py --action all 2>&1 | tee logs/gencdr_douban_raw.log
python -m json.tool artifacts/gencdr_douban/raw_audit.json | less
```

下载只接受上面固定哈希；网络失败保留 `.part`，再次运行重新下载。
已有完整压缩包可用 `--archive /绝对路径/文件.zip --action audit`。
不需要克隆 GenCDSR，也不调用 Qwen 模型。原始数据和报告不提交 GitHub。
`cleanup-legacy` 仅删除仓库内指定资产位置中以 `gencdsr` 开头的旧数据，
不删除前两组和新 Douban 原始压缩包。Git 更新本身不会删除这些 ignored 数据。
本次未连接服务器执行删除，上述命令用于服务器清理。

## 公平对比仍必须落实的事项

1. **确定最终数据快照。** 向 GenCDR 作者取得论文筛选后的用户/物品 ID、
   时间序列和 train/validation/test；若无法取得，两模型必须在共同协议下
   重跑，标记“统一协议复现”，不能将论文数值当作同条件实验结果。
2. **保持留出规则。** 论文规定末次交互测试、倒数第二次验证。不能为提升
   指标重新挑正例。日期无效、同日事件和重复正例的规则要事先明确并共用。
   当前审计把未知日期排最前仅为检查旧逻辑，不批准其进入正式训练。
3. **统一人群和可见历史。** SIR 当前需要两域历史；若只保留共有用户，
   GenCDR 也必须在同一子集重跑。源域信息只能使用预测时点可见历史。
   保留完整两域人群则需为 SIR 明确定义缺失源域处理，不能暗中删样本。
4. **统一物品身份和候选集。** Book/Movie 相同数字 ID 不足以证明同一物品；
   必须核实映射。明确全目录或采样候选、padding、已见过滤和 SID 碰撞。
   已见过滤不能导致重复正例被屏蔽后仍计为普通未命中而不披露。
5. **审计物品文本。** 公开处理器使用评论 labels 聚合元数据，有纳入留出
   交互信息的潜在风险。正式对比须共同使用合法的静态文本或训练历史内
   派生文本；不能声称使用全量评论标签就一定无泄漏。当前不生成该文本资产。
6. **报告方法差异。** SIR 保留 Qwen3-Embedding-8B 离线物品侧和 Qwen3-0.6B
   用户侧、LoRA、两步记忆隐推理与 CPF；这些是 SIR 的设计，不能称 GenCDR
   原架构复现。统一数据/评估协议，并分别报告模型规模与算力预算。
7. **验证与测试隔离。** 用验证集选择参数，冻结配置后测试；已查看的测试
   结果只能作探索性证据。正式结论至少多个随机种子，报告均值与方差。

GenCDR 论文同时列出 Books 和 Movies 两个目标域结果，完整对照需分别跑
Movies→Books 与 Books→Movies。当前 Sports→Clothing、Phones→Electronics
的试验可以继续，但其样本规模也不同于论文，不能直接宣布公平超过基线。

本次未启动第三组推荐训练，也未更改评价指标或前两组模型权重。
