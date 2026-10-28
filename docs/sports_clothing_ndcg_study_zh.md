# Sports↔Clothing：有预算约束的NDCG调参

本工具只进行训练、验证与模型选择，**没有测试集评分入口**。
目标是给定预算与搜索空间内的最佳验证配置，不保证全局最优、固定提升比例或统计显著。
原GenCDR论文数值仍只是外部参照，不用于本次配对显著性结论。

## 隔离与固定项

- 原目录 `/Data/lihb/SIR-CDR` 固定a887bd3，保留现有paper-reference-v2及其暂停记录。
  不git pull、不切HEAD、不改训练源码。已完成的两个seed42原模型不重跑、不覆盖。
- 新控制工具放在 `/Data/lihb/SIR-CDR-sports-hpo` 独立Git工作树，来源为本分支发布提交。
  所有新实验文件仅放在原目录 `artifacts/hparam_search/sports-clothing-ndcg-v1`。
- 工具使用原目录中的原模型、数据、语义ID和固定Qwen版本。数据划分、全训练前缀、
  BF16骨干、有效batch16（8×2）、推理步数、LoRA结构与损失公式不变。
  不引入Douban效率分支的实现变化，避免混淆性能改进与参数改进。
- 原Douban与原调度器当前是SIGSTOP挂起，仍占显存。只使用重新检查后可独占的卡，
  不操作这些旧进程，更不会终止其他用户任务。

## 预先固定的24次新训练预算

每方向以下8组先用seed42训练：

|配置|相对原默认参数的变化（每组都按NDCG@10选模）|
|---|---|
|c00|仅将选模规则从HR@10改为NDCG@10|
|c01|学习率3e-5|
|c02|学习率3e-4|
|c03|检索温度0.05|
|c04|检索温度0.2|
|c05|生成损失权重0.5|
|c06|CPF权重0.03|
|c07|组合：学习率3e-5、温度0.05、生成权重0.5、CPF权重0.03|

默认学习率1e-4、温度0.1、生成权重1、CPF权重0.1；最大30轮、patience5。
选模、学习率调度与早停都跟随NDCG@10，因此c00不是只改保存文件的对照。

1. 两方向交替派发，共16项seed42初筛。每任务先smoke，再完整训练至早停或轮数上限。
2. 按原生验证NDCG@10、HR@10、NDCG@5、HR@5排序，每方向保留前2配置。
3. 每入围配置补seed43/44，共8项新训练。总训练预算24，不无限扩展。
4. 对入围的12个模型收集并精确验证全量验证分数，再各评估同一套48组融合配置：
  pool500/1000、none/zscore、权重0/.125/.25/.375/.5/.625/.75/.875/1/1.25/1.5/2。
5. 每方向联合选择一个训练配置与一个公共融合配置，按三种子平均NDCG@10优先，
  依次以平均HR@10/NDCG@5/HR@5破同分。不得为不同种子各挑一套融合参数。
6. 输出每个种子的指标、样本标准差、全部候选、淘汰和失败记录，并冻结完整来源链。
  本轮最多比较每方向96个“入围训练配置×融合配置”的三种子组合。

原缓存引擎还会为每个入围模型评分17条原融合搜索记录，其中4个配置不在新48组选择网格内。
因此每模型实际有65条评分记录、52个不同配置；12模型共780条评分记录。
原17条仅用于缓存兼容与精确验证，最终选择严格限制在预定48组内。
这些计数不包含训练过程中每轮原生验证，以及本研究前已完成的探索；恢复时确定性复核
也会重算缓存指标，不算新增候选。声明文件同时记录以上次数与选择边界。

这是分阶段筛选：不进入前2的初筛配置不做全三种子验证，因此不能声称穷尽全部配置。
训练轮数与收敛速度变化很大，24次训练不是24轮；按原日志每轮约20～27分钟，若全部
达到30轮上限，训练本身可能需要约240～330GPU小时，另加验证。早停会减少耗时。
不要承诺几小时内完成全部研究。

## 原始探索披露

初始化必须使用受版本管理的
`configs/studies/sports_clothing_prior_observations_20261028.json`。
其中记录原生验证历史、纯检索对照、每方向17条原快照搜索与36次新增评分、
52个不同融合配置和当前已知指标。更早的原生搜索不能从总预算中隐去。
本次三种子仍参与选择，不是独立测试证据；最后测试应在配置冻结后另行规划，
同时尊重原v2尚未完成的测试隔离，不用测试结果返调。
原v2对照目前只有seed42。不能把新三种子平均值与原单种子数值比较后声称配对三种子优势。
若后续需要原HR选模规则的三种子对照，须另行声明补跑两方向seed43/44的4次训练，
不把这4次隐含在本轮24次预算中。

## 部署后先检查计划

先用经过sha256校验的增量Git包获取发布提交，在原目录只fetch对象并创建新工作树。
不要更新原训练树和原控制树。

```bash
source /Data/lihb/miniconda3/etc/profile.d/conda.sh
conda activate /Data/lihb/envs/sir-cdr
cd /Data/lihb/SIR-CDR-sports-hpo
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES="" python experiments/tune_sports_clothing.py --action plan
```

计划与CPU测试不会启动CUDA模型。随后初始化不可覆盖的study声明和48份潜在运行配置
（仅24份会被派发训练，另外的未入围种子配置保持未使用）：

```bash
export HF_HOME=/Data/lihb/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES="" python experiments/tune_sports_clothing.py \
  --action init --root /Data/lihb/SIR-CDR --study-id sports-clothing-ndcg-v1 \
  --prior-observations configs/studies/sports_clothing_prior_observations_20261028.json
```

初始化只验证原资产/本地模型并生成新研究声明，不启动训练，也不下载新模型。
资产指纹检查会读取原数据文件（含测试文件的完整性哈希），测试标签不参与优化或评分。

## 资源检查与启动

启动前提供最新nvidia-smi完整GPU UUID、占用和剩余内存。不能凭旧PID判断空闲，
SIGSTOP进程仍占卡。以下变量必须来自人工核对的允许资源，示例不填任何默认GPU：

```bash
cd /Data/lihb/SIR-CDR-sports-hpo
export HF_HOME=/Data/lihb/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
nohup python -u experiments/tune_sports_clothing.py \
  --action run --root /Data/lihb/SIR-CDR --study-id sports-clothing-ndcg-v1 \
  --gpus "$APPROVED_GPU_UUIDS" --max-workers 2 \
  --host-ram-gib 64 --job-ram-gib 24 \
  >> /Data/lihb/SIR-CDR/artifacts/hparam_search/sports-clothing-ndcg-v1/controller.log 2>&1 &
```

`APPROVED_GPU_UUIDS`为逗号分隔的完整UUID列表，例如从允许且核对过的物理卡映射而来。
如仅使用1卡，初次运行就设max-workers1。第一次run将GPU池、并发和RAM预算写成不可覆盖资源策略，
恢复必须相同，不能静默改池或扩大预算。每任务固定GPU UUID，后续验证/恢复保持一致。
工具只在卡无其他进程、资源满足门槛时派发；检查不替代跨用户资源预约，需协调使用时段。
宿主机可用内存必须覆盖全部活动任务预留及新任务预留，且总预留不超过声明预算。
这是一项保守准入检查，实际已分配内存也可能被重复预留，因此可能暂缓第二项任务。
训练失败后停止新派发、等待已运行兄弟任务结束，保留失败日志。不会发信号抢占任何进程。

## 查看状态、恢复与冻结

```bash
CUDA_VISIBLE_DEVICES="" python /Data/lihb/SIR-CDR-sports-hpo/experiments/tune_sports_clothing.py \
  --action status --root /Data/lihb/SIR-CDR --study-id sports-clothing-ndcg-v1
tail -n 40 /Data/lihb/SIR-CDR/artifacts/hparam_search/sports-clothing-ndcg-v1/controller.log
```

控制器意外退出后，先检查是否还有本研究worker活着；工具拒绝在孤儿worker尚在运行时重复接管。
等其完成，再执行原run命令。部分训练只能从原引擎已保存的整轮检查点恢复；当前未保存轮次可能
重算，不能把进程PID退出当作实验完成。完成的训练和精确评分经过身份核验后跳过。
不手工删除锁/状态、改source哈希、修改声明或补写完成回执。
资源策略与晋级记录各自将内容及摘要原子发布在同一个文件中，避免中途退出遗留缺失摘要。

最终validation_freeze.json是本次新研究的验证选择冻结，与原v2冻结分开。
它不是测试结果，也不表明学术上统计显著。正式比较需冻结后的一次性测试及适当配对统计，
分别说明训练随机性与评估用户抽样不确定性。
