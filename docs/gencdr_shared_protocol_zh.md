# GenCDR 共同数据协议与独立训练入口

## 定位：不是论文原表数据的完整复现

上游来源：<https://github.com/hupeiyu21/GenCDR>，公开划分实现
<https://github.com/hupeiyu21/GenCDR/blob/main/dataloader/dataset.py> 的
`AbstractDataset._leave_one_out`。它输出 train=seq[:-2]、val=seq[:-1]、test=seq；
各序列末项作为下一个物品标签。论文说明最终使用验证 Recall@10 最优 checkpoint：
<https://arxiv.org/html/2511.08006v1>。
本次核对的上游 HEAD：`204b0ce0f624d04f84d0c72c52f4d62512ce4bdf`，
协议记录固定提交，不依赖日后移动的 main。

截至本次实现，用户没有作者最终过滤人口、划分及评估文件。本入口实现公开的留一边界，
并把未充分公开的跨域历史、人口和评价细节固定成可审计的**共同协议**。
所有输出 `paper_alignment_verified=false`。不能把运行结果直接与论文表格作公平胜负结论。
需要在这份共同数据上重新运行 GenCDR，再比较双方；本提交不包含 GenCDR 推荐器的训练实现。

## 新旧实验隔离

新入口：`experiments/run_gencdr_shared.py`。

默认三个方向：sports_to_clothing、phones_to_electronics、douban_book_movie。
也支持 clothing_to_sports、electronics_to_phones、douban_movie_book，需显式指定 `--pairs`。

新目录统一包含 `gencdr-shared-loo-v1`：

- `data/processed/gencdr-shared-loo-v1/<pair>/`：新样本、映射、文本、原始事件与审计。
- `artifacts/embeddings/gencdr-shared-loo-v1/<pair>/`：新向量，独立固定模型 revision。
- `artifacts/tokenizer/gencdr-shared-loo-v1/<pair>/`：新训练的 SID 与潜在向量。
- `artifacts/llm_cdr_v1/gencdr-shared-loo-v1/<pair>/seed42/`：新推荐模型和测试结果。
- `configs/runtime/gencdr-shared-loo-v1/<pair>_seed42.yaml`：只指向新资产。
- `artifacts/shared_protocol/gencdr-shared-loo-v1/<pair>/ready.json`：完整资产封存哈希。

原来的 `run_llm_fast.py`、`run_douban.py` 继续代表旧协议，不会自动切换到新数据。
旧结果保留为旧实验，不能重命名为新协议结果。后续新实验必须使用本入口。
不修改现有训练核心的文件身份，也不停止正在运行的远端进程。

## 冻结的共同协议

1. Amazon 使用现有 Amazon2014 评论与静态元数据文件；Douban 使用现有 publisher v1 ZIP，核验固定 SHA256。
   不下载其他 Douban/GenCDSR 数据，不随机采样匹配论文数量。
2. 两域有有效时间交互、且每域至少 5 次交互的共同用户。域名与原始物品 ID 共同定义身份。
3. 保留重复事件；按时间、原始物品 ID、事件键排序；完全重复的事件键拒绝。
4. 每域最后一次交互测试、倒数第二次验证。目标域训练只来自更早前缀。
5. 源域上下文也只来自该域训练前缀 `seq[:-2]`，且时间严格小于目标标签。
   因此反向任务的验证/测试事件也不会在本方向被当作源域训练输入。
6. 默认将目标训练前缀展开成所有有非空历史的 next-item 样本。
   `--training-examples last-prefix` 可改为只监督训练前缀末项，但它是不同的训练配方，
   不能放在相同目录中复用；必须先另设完整实验根目录。公开划分函数本身没有规定下游如何展开训练序列。
7. 历史不足的样本记录在 `excluded_samples.jsonl`；不替换留出目标，不挑选更易预测的测试记录。
8. 全部保留用户涉及的目录物品作为 transductive 目录，包括只在留出中出现的物品。
   缺文本不删除交互，使用明确的 ID 文本回退。Amazon 仅静态商品属性；Movie 静态字段白名单；
   Book 使用 ID 文本，避免旧协议生成的交互标签在新划分中泄漏。不得把聚合评分/评论/测试标签补入文本。
9. 存储完整历史用于已见物品屏蔽，编码器截取最近 50 项。完整目标域目录参与检索，混合模型再对候选重排。
   不采样负样本。重复正例如果已见仍被屏蔽，并报告数量；GenCDR 必须使用相同规则，否则不可公平比较。
10. 一条评测记录一个正例，R@K=HR@K，N@K=NDCG@K。训练选模用 HR@10（等于本任务 Recall@10）。
    推荐器 Qwen3-0.6B、两步隐推理、LoRA 与 CPF 保持现有架构；统一 microbatch8/累积2/eval8，
    最多30轮、patience5、每轮验证。不是对 GenCDR 参数量或训练方案的等价复现。
11. 完成训练后复用独立目录的 CPU float32 物品快照评估器：预设17种融合配置仅看验证集，
    保护四项相对默认方案的指标后按 N@10 等规则选择。它属于预先声明的验证调参预算，
    与训练 checkpoint 的 Recall@10 选择规则不同。新进程精确核验后只进行一次锁定的最终测试。

协议不是保证更高指标的 trick。源域信息减少、Book 缺少静态文本等可能降低性能，必须如实报告。
用户选择、候选集合、屏蔽规则、训练样本展开和文本合法性应在双方复现报告中列出。

## 在服务器上按顺序操作

### 1. 更新代码并生成 CPU 数据划分（可以与旧训练共存）

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
nohup python -u experiments/run_gencdr_shared.py --action convert > logs/gencdr_shared_convert.log 2>&1 &
echo $! > logs/gencdr_shared_convert.pid
```

此阶段需要现有完整原始文件。缺失时明确报错，不回退为旧处理结果。
它会重新解析原始交互，CPU、内存和磁盘有开销；没有启动训练或占用8B GPU模型。

### 2. 当前旧训练和评测全部结束后，准备 GPU 资产

```bash
nohup python -u experiments/run_gencdr_shared.py --action prepare > logs/gencdr_shared_prepare.log 2>&1 &
echo $! > logs/gencdr_shared_prepare.pid
```

顺序处理三组。每组先生成8B Embedding，释放模型进程，再训练 SID，最后下载/校验0.6B骨干并封存资产。
本次不复用未记录 revision 的旧向量，也不复用旧 SID。8B 下载缓存可以复用。
检测到当前正向工作流或其他 GPU 进程时拒绝启动；不要为绕过检查删除旧进程或修改守卫。
中断后重复相同 prepare 命令恢复，不使用 `--force`，不删除现有数据。
若提示 failed test reservation，保留现场审计，不删除 test.lock 重新挑选结果。

### 3. 检查完成情况

```bash
python experiments/run_gencdr_shared.py --action status
tail -n 15 logs/gencdr_shared_prepare.log
```

status 是快速文件存在性展示，不是完整验证。训练入口会实际核验全部处理数据、资产哈希、
Tokenizer 输入链和运行配置，缺一项即拒绝启动。

### 4. 使用新资产开始推荐任务

```bash
nohup python -u experiments/run_gencdr_shared.py --action run > logs/gencdr_shared_run.log 2>&1 &
echo $! > logs/gencdr_shared_run.pid
```

默认逐组训练、验证调参、精确校验、最终测试。`--pairs` 可只运行一组；同一身份中断可恢复。
训练未完成或身份不符不允许进入最终测试。不要将旧 checkpoint 复制到新目录。
后续需要多种子时，先预先声明种子集合，分别使用 `--seed 43` 等重新 prepare 生成对应运行配置，
共享同一份数据/物品资产，推荐模型与测试目录按种子隔离。报告均值与标准差，保留所有运行。
新目录、新种子不代表新的独立盲测集：若与旧协议共享已查看过的留出事件，应披露既往测试接触，
不能宣称这些事件完全未参与研究决策。需要确认性结论时应另设真正未接触的确认数据。

## 将同一划分提供给 GenCDR

处理目录包含 `gencdr_sequence_splits.jsonl`，每个用户包含目标域的 train/val/test 序列，
边界与公开 `_leave_one_out` 一致；`mappings.json` 提供内部 ID 到原始域/物品的映射。
每行的 `accepted_target_positions` 明确哪些位置通过历史可见性检查，空列表表示该 split 不评测该用户。
该文件是序列边界桥接文件，**不能单独代替完整跨域评测协议**。

双方推荐器必须实际消费同一份 `train.jsonl`、`validation.jsonl`、`test.jsonl` 中的
源域历史、目标域历史与标签，以及同一目标目录与 seen-policy。
GenCDR 可以把这些行的内部 ID 映射为自己的 SID，但不得自行重新筛用户、重新划分或扩大历史。
下游 GenCDR 适配器必须先比对逐条身份/历史/标签，再运行其训练；本提交没有宣称已完成该训练器。
单域用户无法直接满足当前 SIR-CDR 的双域非空历史接口，不能伪造另一域历史。

## 作者数据到来后的导入口

`--bundle /absolute/path/to/bundle --pairs <one_pair> --action convert`。
为了核验跨域时间可见性，只有没有时间戳的 `all_item_seqs.json` 不够，不能编造时间。
请先提供下列标准化文件，原始用户 ID 必须真正跨域一致，不接受按类别拼接的局部编号作为共同用户：

- `protocol.json`：`text_policy: "static-or-id-only"`、`source_description`（作者数据版本/来源）。
- `catalog.jsonl`：`{"domain":"Book","original_id":"123","text":"..."}`。
- `events.jsonl`：`{"user":"u","domain":"Book","item":"123","timestamp":100,"key":"unique-event-1"}`。
- `sequence_splits.jsonl`：每个用户/域一行：`user`、`domain`、`train`、`val`、`test`，列表使用原始物品 ID 字符串。

读取器检查每条序列及划分是否与带时间事件严格一致，不允许默默重新排序改变作者的目标。
当前导入口只接受符合本共同协议的双域>=5用户；不支持的人口会拒绝，而不是静默删除。
导入保留外部文件哈希和声明来源，仍不自动将作者论文人口与候选规则标记为已验证。
它适配的是作者**留一序列边界**；若作者最终样本使用不同的源历史、人口或训练展开，
必须先确定共同协议再扩展适配器，不能以本入口擅自改写后宣称原样导入。

## 验证范围

本次仅运行新增的必要 CPU 检查：留一边界、源域留出隔离、未来历史拒绝、重复事件、
外部边界顺序、防混用与新配置隔离。没有在本机生成生产 GPU 向量、训练推荐器或生成指标。
生产资产必须在服务器执行上述 convert/prepare 后才能生成；提交代码不代表服务器资产已就绪。
