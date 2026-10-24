# 六方向 SIR-CDR 服务器执行手册

本流程运行 Sports↔Clothing、Phones↔Electronics、Douban Book↔Movie，种子固定为 42/43/44。完整模型为固定版本 Qwen3-0.6B。GenCDR 数值仅为论文外部参照，作者最终划分未核验，不能据此声称严格同条件优于 GenCDR。

所有代码通过 GitHub 更新；你在已登录的服务器终端执行以下命令。无需向代理提供 SSH、密码、私钥或令牌。私人代码、数据和缓存放在个人目录中，遵守组内用卡约定。首次执行前人工确认 GPU 白名单，控制器只会在该白名单内选当前空闲卡。

## 1. 更新与检查机器

进入服务器上的仓库（建议位于个人 `/Data/<username>/SIR-CDR`），再执行：

```bash
git status --short
git pull --ff-only origin main
git rev-parse HEAD
nvidia-smi
lscpu
free -h
df -h .
```

工作区有未提交修改时先保留并处理自己的修改；不要使用 reset/clean 覆盖。运行期间固定当前 commit，不执行 git pull。GPU空闲不等于已经获得使用许可，需要先按组内规范确认可用卡。

设置本次会话变量，提示输入个人数据目录和已批准的 GPU 编号：

```bash
read -r -p '个人数据目录的绝对路径（例如 /Data/你的用户名）: ' SIR_DATA
read -r -a SIR_GPUS -p '已获准且当前可用的GPU编号，以空格分隔: '
export SIR_BATCH=paper-reference-v1
export HF_HOME="$SIR_DATA/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p logs
```

激活个人 conda 环境，检查依赖：

```bash
python -c "import torch, transformers, peft, yaml; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s tests -v
python experiments/run_gencdr_six.py --action preflight --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA"
```

预检报告记录 GPU占用、CPU/负载、内存、文件系统、驱动、CUDA runtime、Python/PyTorch、Git commit、人工白名单。`admission.production_ready=true` 才能进入后续执行；预检本身不启动训练。单个忙碌卡会被排除；全忙时停止，之后由你重新执行，而不是后台等待抢卡。初始磁盘门槛为60GiB，正式准备和运行时还应按原始数据、缓存与18份checkpoint的实际占用保留余量。

若机器尚无环境，按仓库 requirements 在自己的 conda 环境安装，不能修改系统 CUDA 或他人环境。当前 requirements 固定 PyTorch 2.6.0/cu124；服务器驱动信息和 PyTorch CUDA runtime 不是同一个版本号，先以实际导入、GPU可用性和烟雾测试结果判断兼容性。

## 2. 原始数据就绪

新协议重新解析原始数据，不能用旧划分替代。预期文件：

```text
data/raw/amazon2014/reviews_Sports_and_Outdoors_5.json.gz
data/raw/amazon2014/meta_Sports_and_Outdoors.json.gz
data/raw/amazon2014/reviews_Clothing_Shoes_and_Jewelry_5.json.gz
data/raw/amazon2014/meta_Clothing_Shoes_and_Jewelry.json.gz
data/raw/amazon2014/reviews_Cell_Phones_and_Accessories_5.json.gz
data/raw/amazon2014/meta_Cell_Phones_and_Accessories.json.gz
data/raw/amazon2014/reviews_Electronics_5.json.gz
data/raw/amazon2014/meta_Electronics.json.gz
data/raw/douban_fengzhu_v1.zip
```

Amazon原始文件名以 `category_files` 解析结果为准。缺失时先通过现有 `prepare_amazon.py --help`、`prepare_gencdr_douban.py --help` 中的下载入口准备原始文件；这可能额外生成旧协议产物，不得将其混入本次结果。Douban ZIP 必须通过仓库固定 SHA256 校验。不要拿 Amazon Books/Movies 替代 Douban Book/Movie。

## 3. CPU转换与GPU资产准备

先转换。默认只开一个原始文件解析进程以控制主存；资源足够可显式设 `--cpu-workers 2` 或3。

```bash
nohup python -u experiments/run_gencdr_six.py --action convert --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA" --cpu-workers 1 > "logs/${SIR_BATCH}_convert.log" 2>&1 &
```

检查父日志和 `conversion.json` 中六方向全部 `returncode=0` 后，运行：

```bash
nohup python -u experiments/run_gencdr_six.py --action prepare --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA" > "logs/${SIR_BATCH}_prepare.log" 2>&1 &
```

准备阶段先统一固定模型revision，再最多三卡准备正向资产，接着准备反向资产。字节完全相同的条目文本与编码参数允许验证后复制embedding，避免重复8B编码；方向特定tokenizer仍独立训练，质量门通过后封存。模型编码和tokenizer使用不同子进程，释放8B模型后再进入下一阶段。已完成且哈希一致的资产复用，中断后重复相同命令恢复；不使用 `--force`。

## 4. 烟雾测试与并行训练/验证

资产完成后执行：

```bash
nohup python -u experiments/run_gencdr_six.py --action smoke --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA" > "logs/${SIR_BATCH}_smoke.log" 2>&1 &
```

六方向均通过真实batch前向/反向、梯度、排序检查后执行：

```bash
nohup python -u experiments/run_gencdr_six.py --action run-validation --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA" > "logs/${SIR_BATCH}_validation.log" 2>&1 &
```

默认最多使用白名单中的六张空闲卡，一卡一个方向；可设 `--max-workers 3` 降低并发。每次新启动都重新检查卡占用和主存/负载。训练显存准入使用烟雾测试峰值×1.25+2GiB余量；保持有效batch16、最多30轮、patience5、HR@10选checkpoint。

六个seed42作业全部训练、验证融合选择和精确验证成功，才运行seed43，再运行seed44。每个任务固定GPU UUID以满足精确评测身份；恢复时需要该卡仍在白名单且空闲，不能随意迁移。所有18个作业完成前不计算最终测试指标。

## 5. 冻结、最终测试、汇总

查看状态：

```bash
python experiments/run_gencdr_six.py --action status --batch-id "$SIR_BATCH"
```

18个作业全部完成后，冻结checkpoint和验证选择：

```bash
python experiments/run_gencdr_six.py --action freeze --batch-id "$SIR_BATCH"
```

只有成功生成并核验 `validation_freeze.json` 后才能运行最终测试：

```bash
nohup python -u experiments/run_gencdr_six.py --action run-test --batch-id "$SIR_BATCH" --gpus "${SIR_GPUS[@]}" --data-root "$SIR_DATA" > "logs/${SIR_BATCH}_test.log" 2>&1 &
```

测试结束后：

```bash
python experiments/run_gencdr_six.py --action summarize --batch-id "$SIR_BATCH"
```

结果位于 `artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/results.json` 与 `results.csv`，包含18份逐种子指标、六方向均值与样本标准差、GenCDR论文参照及明确的不可严格直接比较说明。

## 6. 状态与故障处理

- 父日志：`logs/<batch-id>_<phase>.log`。
- 逐任务日志：`logs/gencdr-shared-loo-v1/<batch-id>/<pair>_seed<seed>_<phase>.log`。
- 状态：`artifacts/experiment_control/gencdr-shared-loo-v1/<batch-id>/status/<phase>/`。
- 数据与资产仍在schema隔离目录，运行配置、推荐器、融合缓存按batch/方向/种子隔离。
- 某一任务失败保留其他已完成结果。资源占用导致未启动时，确认空闲后重复原命令；训练中断由现有checkpoint恢复。
- 已产生最终测试预约而未完成时停止并保留现场。不要删除 `test.lock`、`test.json` 或更换配置重测；先提供脱敏状态和末尾日志进行故障审计。
- CPU压力、主存或磁盘不足时降低并发或等待资源；OOM不能通过删数据或缩小评测目录处理。需要修改microbatch时必须保持有效batch并在新的实验身份下重新验证，不能改完继续复用旧身份。
- 六方向均使用相同的预声明规则；所有种子保留。不能以结果不理想为由舍弃种子或修改测试政策。
- 后续修复仍走本地修改/测试/提交GitHub，再由服务器拉取；已冻结批次更新代码后会拒绝继续。新代码使用新batch，并审查既往测试接触情况。

需要协助时，提供 `git rev-parse HEAD`、脱敏的预检JSON、状态JSON和相关日志末尾；不要提供任何凭据。
