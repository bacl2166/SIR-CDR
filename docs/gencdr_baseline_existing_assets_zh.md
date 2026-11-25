# 在现有数据资产上运行 GenCDR 基线

## 实验名称与复现边界

输出名称为 **GenCDR-reconstruction-common-assets**。这是基于公开代码和论文方法补全的共同数据协议实现，不能标为作者原版运行结果，也不能直接等同于论文 Table 2 的 GenCDR 数值。

官方仓库：[hupeiyu21/GenCDR](https://github.com/hupeiyu21/GenCDR)，核对版本 `204b0ce0f624d04f84d0c72c52f4d62512ce4bdf`。公开的主要模型代码是 `quantization/rqvae/rqvae.py`、`quantization/rqvae/layers.py`、`quantization_rqvae.py` 和 `quantization_adapter.py`。README 的推荐解码器训练章节只指向 LLaMA-Factory，没有提供完整解码器、用户路由及推荐评估代码。

本实现独立重写模型，不复制 SIR 的图传播、隐推理或 CPF 模块作为 GenCDR。论文中未公开实现的细节均在实验声明中披露。

| 部分 | 实现与依据 | 与原版的关系 |
|---|---|---|
| 通用分词器 | 残差 MLP 编解码、逐层残差量化、EMA 码本、累计承诺损失 | 依据公开 RQ-VAE 计算路径重写 |
| 领域分词器 | 冻结通用编码器与码本，训练两个领域的低秩编码器和 item VIB router | 根据论文补全；公开脚本没有完整动态路由实现 |
| 通用推荐器 | 冻结 Qwen 主干，在 q/v 投影中训练四个 LoRA 专家和输入相关专家门控 | 论文提出多专家，具体混合位置为本实现选择 |
| 领域推荐器 | 冻结通用参数，在通用投影上叠加目标域 LoRA，训练 user VIB router | 根据论文两阶段方法补全 |
| 概率路由 | `(1-gamma) P_universal + gamma P_specific`，以 logaddexp 稳定计算 | 论文 Eq.11 |
| 解码 | 目标域 SID 前缀树、合法分支 masked softmax、beam=50 | 论文 Eq.12；无检索/生成分数融合 |
| 碰撞 | 相同 SID 的物品等分该码概率，分数同分按 item ID 升序 | 预先固定规则，不看验证/测试正例 |

以下差异必须随实验报告披露：

- 使用 SIR 已有的归一化 Qwen 商品嵌入，不运行官方 OpenAI embedding/PCA/数据清洗。
- 默认冻结 `Qwen/Qwen3-0.6B`，revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`，与当前 SIR 主干保持一致。论文主要设置为 Qwen2.5-7B，不能作同模型规模原样复现声明。
- 采用现有共有用户、历史和标签；与作者论文人口规模不同。
- 官方代码未实现论文中 tokenizer 的 MTM，当前重构也未实现 MTM；不能宣称包含论文全部训练损失。
- 输入把源域、目标域已有历史分别编码并添加领域标记。由于现有样本没有逐条历史时间戳，不虚构双域交错顺序。
- 通用推荐阶段只读取正反方向已经存在的训练样本。反向目录的全局 ID 按现有映射在内存中转换，不新增样本或输出数据文件。
- 最佳模型选择采用当前实验的双 NDCG 规则，原论文选择 Recall@10。

## 数据与任务

默认跟随 SIR 批次 `amazon-expansion-dual-ndcg-disk24-v1`：

| 方向 | 随机种子 |
|---|---|
| Food → Kitchen | 42、43、44 |
| Kitchen → Food | 42、43、44 |
| Beauty → Electronics | 42、43、44 |
| Electronics → Beauty | 42、43、44 |

只读输入为 `data/processed/gencdr-shared-loo-v1/<pair>` 中现有文件，以及 `artifacts/embeddings/gencdr-shared-loo-v1/<pair>/text_embeddings.pt` 和 manifest。检查文件哈希、映射和样本数；发现不一致直接报错，不修复或过滤。既有重复正例照常保留，评估时按共同协议屏蔽已见物品并将该例计作未命中，不剔除样本。

**训练 GenCDR 自身的分词器属于模型拟合**，不会重跑商品嵌入、改现有 SID、重新切分数据或处理原始评论。GenCDR 新生成的 SID 只保存在自身 `tokenizer.pt`，SIR 的现有 SID 和所有预处理文件均保持不变。

所有新文件都写到：

```text
/Data/lihb/SIR-CDR/artifacts/gencdr_baseline/gencdr-reconstruction-amazon-v1/
```

## 默认训练与评估

分词器通用训练 100 epoch，领域适配 50 epoch；batch=512；学习率分别为 1e-4 和 5e-5；latent=256，3 层、每层 2048 个码，累计承诺项系数 0.25。两层残差 MLP hidden=[1024,512]。领域编码器 LoRA rank=64，当前缩放 alpha/rank=1；VIB 为标量高斯后验与标准正态的 KL，权重 1e-3。

推荐通用阶段在两个方向既有训练样本上训练 10 epoch；目标领域阶段最多 20 epoch，patience=5。AdamW lr=5e-5，weight_decay=0，batch=8，梯度累积=1，LoRA rank=64、alpha=128，四个通用专家。每个方向每个种子独立拟合，不挑选最好 seed 替代完整报告。每个领域历史最多保留最后 50 个物品，与当前 SIR 配置一致。

只根据验证集选择模型，按以下顺序比较：

1. `(NDCG@5 + NDCG@10) / 2`
2. `(HR@5 + HR@10) / 2`
3. `NDCG@10`
4. 完全同分保留更早轮次。

十二组训练全部完成后绑定验证 checkpoint，再开放测试。候选空间为整个目标域目录，使用前缀树 beam 近似生成，不使用 sampled negatives，不补入正确答案，不用 SIR 的检索重排器。移除该样本全部已见目标域物品；未生成正例计零分。SID 碰撞会影响结果，应报告 `tokenizer_quality.json`。输出 HR/NDCG@5/10/20 及 Recall 别名，单正例时 HR=Recall。

`results.json` 同时包含所有种子指标和三个种子的均值、样本标准差。比较 SIR 时需报告两者同数据、不同 tokenizer/解码器及训练预算；观察到的差异不能归因于单个模块，需另外消融。

## 自动等待、资源与续跑

现在启动调度器即可：没有 SIR 的全部测试收据、验证冻结和最终结果时，只打印 `WAIT_SIR`。控制器退出或 GPU 释放都不能冒充 SIR 实验完成。

通过 SIR 结果绑定核验后，自动选择无计算进程、GPU 利用率不超过 5%、可用显存至少 24 GiB 的卡。默认最多同时运行两组，每组预留 24 GiB 主机 RAM；每次启动检查磁盘至少余 24 GiB，资源不足持续等待。不会终止、暂停或抢占其他用户进程。

已训练任务直接核验后跳过。tokenizer 和推荐模型均按 epoch 保存 optimizer/RNG 断点；中途失败最多从最近完整 epoch 重做当前 epoch，不冒充精确 batch 续跑。只保存 LoRA、SID embedding、路由等新增权重，不重复保存冻结 Qwen。继续执行时保持训练 GPU UUID，避免悄悄改变硬件条件。验证与测试按该 UUID 排队。

目前尚未在 L20 上运行本新基线，因此 24 GiB 显存只是保守准入门槛，**不是已测峰值保证**。OOM/非有限损失将保留日志并停止新的任务准入，等待同批已启动任务结束，不自动减小 batch 或更改科学配置。初次运行应查看首组峰值和日志。

## 服务器操作

代码必须放在独立、干净且已提交的 checkout。不要更新正在运行 SIR 的主目录代码。获取本分支后，在新代码目录执行：

```bash
export HF_HOME=/Data/lihb/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python \
  -m unittest discover -s tests -p 'test_gencdr_baseline*.py' -v

mkdir -p /Data/lihb/SIR-CDR/logs
nohup /Data/lihb/envs/sir-cdr/bin/python -u \
  experiments/run_gencdr_baseline.py \
  --root /Data/lihb/SIR-CDR \
  --source-batch amazon-expansion-dual-ndcg-disk24-v1 \
  --batch-id gencdr-reconstruction-amazon-v1 \
  --max-workers 2 --poll-seconds 60 \
  >> /Data/lihb/SIR-CDR/logs/gencdr-reconstruction-amazon-v1.log 2>&1 </dev/null &
```

该命令不会下载模型；默认 Qwen 主干必须在已有 HF cache 中。不需要再安装 LLaMA-Factory或提供 API key。使用原项目 requirements 环境。初次测试 CPU tiny Qwen，不下载模型、不占用 GPU。

状态与日志：

```bash
CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python \
  experiments/run_gencdr_baseline.py --root /Data/lihb/SIR-CDR --action status
tail -n 40 /Data/lihb/SIR-CDR/logs/gencdr-reconstruction-amazon-v1.log
```

如果需要改变科学参数，在实验启动前提供完整或局部 JSON `--config <path>`；首次 request 会冻结该配置。开始后不能覆盖同一批次声明；必须使用独立 batch ID，并完整报告调参预算。
