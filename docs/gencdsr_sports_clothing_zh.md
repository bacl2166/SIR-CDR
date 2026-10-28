# GenCDSR参考协议上的SIR-CDR双向实验

本实验保留SIR-CDR模型主干，使用重建的Sports/Clothing共同协议。
协议名为`gencdsr-inspired-sc-v1`，不是GenCDSR作者最终资产的逐条复现。
GenCDSR官方仓库固定提交为`bace0d18f53bfeb56ab9bdd564494055b6c38db7`；
核对时其`data/`仅附Book_Movie，未附Sports/Clothing处理后数据。

## 数据与模型的边界

|部分|本实验处理|
|---|---|
|数据源|复用服务器已有Amazon2014 5-core评论与静态元数据文件，记录文件哈希|
|共同人口|按真实原始用户ID求两域候选交集|
|过滤|在共同人口内迭代删除域内不足3次的物品及任一域不足5次的用户，直至稳定|
|计数口径|交互事件次数，保留重复交互；不是不同物品个数|
|过滤顺序|本次明确声明的重建选择；不保证等同作者未公开的过滤器|
|目标留出|各目标域最后一次测试、倒数第二次验证，其余全前缀next-item训练|
|源域输入|仅源域训练前缀，且事件时间严格早于目标标签；不借入源域留出事件|
|缺历史|保留原接口的双域非空要求；排除条目逐条审计，不替换留出正例|
|物品目录|保留人口涉及的所有域限定物品，包含仅在留出中出现的物品|
|物品文本|静态商品属性；缺失时显式ID文本回退，不因文本缺失静默删交互|
|向量与SID|保留SIR向量和tokenizer方案，在新目录重新生成并封存|
|主干|Qwen3-0.6B、LoRA、稀疏图、原型、CD/SP注入、隐推理、CPF及损失公式不变|
|评测|物品级全目标目录检索与重排；完整已见历史屏蔽；重复正例被屏蔽也报告|

原输入已经是5-core，因此在完整原域上重复做item3筛选可能没有作用。
本协议把筛选明确放在共有用户人口内；它可能改变物品目录和用户规模。
筛选使用完整事件集，是本实验声明的transductive预处理；图边只来自训练历史。
数据审计需验证每轮筛选、最终阈值、双向人口/事件对称、标签位置、严格时间边界、
排除账本和文件哈希。不得根据最终推荐指标反复更换过滤顺序。

本实验不复制GenCDSR的混合序列预训练再单域适配训练阶段，不替换为其T5或串并行解码器。
因此方法名称仍为SIR-CDR，结果应写为“GenCDSR参考共同协议上的SIR-CDR”。
未在完全相同资产上运行对方模型时，论文数值只作外部参照；不据此宣称公平超越。

## 与现有实验隔离

- 原`/Data/lihb/SIR-CDR`只提供已有`data/raw/amazon2014`输入，不更新其HEAD或训练文件。
- 新代码建议部署至`/Data/lihb/SIR-CDR-gencdsr-sc`独立工作树。
- 新processed/embedding/tokenizer/control/runtime/model输出都在新工作树的协议命名空间下。
- 旧`gencdr-shared-loo-v1`、`paper-reference-v2`、`sports-clothing-ndcg-v1`结果保持原身份。
- 新目录不意味着独立盲测集。新旧协议可能共享已查看的事件，应披露既往测试接触；
  确认性结论需要适当独立测试数据，不能靠换目录消除既往接触。
- 新资产和新训练签名不兼容旧checkpoint，不能复制旧模型冒充新协议训练。

## 固定首轮训练配方

两个方向，各seed42/43/44。先完成seed42双向pilot，检查数据审计、loss及验证结果，再补43/44。
推荐器microbatch8、累积2、evalbatch8，历史长度50，最多30轮、patience5。
原默认学习率1e-4、检索温度0.1、生成损失1、CPF损失0.1；按验证NDCG@10选模。
固定hybrid候选500、生成分数融合权重0.5、normalization=none；第一轮不搜索融合参数。
报告HR/NDCG/MRR@5/10/20、各种子指标和样本标准差。六项训练及验证完成并冻结后才可测试。
改变数据筛选与物品目录后，指标可能变高或变低，不能把变化全部归因于主干更好。

## 服务器操作顺序

所有源代码先提交GitHub，再由服务器fetch至独立工作树。不要在原活动目录git pull。
代码部署命令使用交付时给出的精确commit；在线连接失败可用校验后的增量Git包。
本地CPU检查不代表已完成服务器数据重建或CUDA训练。

激活并检查计划：

```bash
source /Data/lihb/miniconda3/etc/profile.d/conda.sh
conda activate /Data/lihb/envs/sir-cdr
cd /Data/lihb/SIR-CDR-gencdsr-sc
export HF_HOME=/Data/lihb/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" python experiments/run_gencdsr_sports.py --action plan
```

CPU数据重建与审计：

```bash
CUDA_VISIBLE_DEVICES="" python experiments/run_gencdsr_sports.py \
  --action convert --data-root /Data/lihb/SIR-CDR
CUDA_VISIBLE_DEVICES="" python experiments/run_gencdsr_sports.py --action audit
```

先查看双向统计和排除账本。若过滤后人口/任何split为空，应保持审计现场并调查原始数据，
不能自行降低阈值、替换正例或切回旧processed目录。

再次查看`nvidia-smi`、`free -h`和本人进程，人工确认允许使用的完整GPU UUID。
暂停的旧Douban进程仍占显存，不能把GPU利用率0当作无进程。以下变量不提供默认卡号：

```bash
read -r -p "输入已确认可独占GPU的完整UUID: " APPROVED_GPU_UUID
```

以Sports→Clothing为例，各阶段单独启动进程，释放前一个阶段的模型内存：

```bash
python experiments/run_gencdsr_sports.py --action embedding \
  --pair sports_to_clothing --gpu "$APPROVED_GPU_UUID"
python experiments/run_gencdsr_sports.py --action tokenize \
  --pair sports_to_clothing --gpu "$APPROVED_GPU_UUID"
python experiments/run_gencdsr_sports.py --action seal --pair sports_to_clothing
python experiments/run_gencdsr_sports.py --action runtime \
  --pair sports_to_clothing --seed 42
python experiments/run_gencdsr_sports.py --action smoke \
  --pair sports_to_clothing --seed 42 --gpu "$APPROVED_GPU_UUID"
python experiments/run_gencdsr_sports.py --action train \
  --pair sports_to_clothing --seed 42 --gpu "$APPROVED_GPU_UUID"
python experiments/run_gencdsr_sports.py --action validate \
  --pair sports_to_clothing --seed 42 --gpu "$APPROVED_GPU_UUID"
```

反向把pair替换为`clothing_to_sports`。新种子只重复runtime/smoke/train/validate，不重建同一数据及物品资产。
当前只一张可独占卡时顺序运行两个方向。两张卡都人工确认且空闲时可各跑一个方向，使用各自完整UUID；
最多两任务并行，预留至少48GiB宿主机内存，先以当前可用内存核对，不能共用忙卡。
正式后台命令应由当次卡号、阶段和日志位置确定后再生成，避免误把旧任务恢复或重复启动。

六个模型全部完成验证后：

```bash
CUDA_VISIBLE_DEVICES="" python experiments/run_gencdsr_sports.py --action freeze
```

冻结会绑定六个checkpoint、验证结果、配置、源码和资产。后续test需要相同身份与GPU绑定，
运行前生成不可删除的一次性预约。失败保留现场，不能删test锁后反复挑参数。
先完成上述数据与验证步骤，再根据freeze输出给出最终test命令。

## 上游依据

- GenCDSR论文：https://arxiv.org/html/2607.28659v1
- 固定仓库：https://github.com/Applied-Machine-Learning-Lab/RecSys2026_GenCDSR/tree/bace0d18f53bfeb56ab9bdd564494055b6c38db7
- 数据加载：GenCDSR-TIGER-multi-domain/data.py
- 评测接口：GenCDSR-TIGER-multi-domain/evaluate.py

过滤顺序、共有用户人口、因果源域可见性、已见屏蔽和独立SIR资产都是本协议明确声明的选择，
不是对作者所有实验隐含细节的推定。
