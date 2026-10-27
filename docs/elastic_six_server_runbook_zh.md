# 将正在运行的三并发批次扩容到六个方向队列

适用现有训练目录 `/Data/lihb/SIR-CDR`、训练提交
`a887bd389ff72eb14ec96be0dfb876a61dae9331`、批次 `paper-reference-v2`。
用户已批准调用其余空闲卡，并允许已完成方向继续其他种子，禁止重复已完成的方向/种子。

## 资源与实验规则

- GPU池为 **0、1、4、5、6、7**，每张卡最多一个任务，最多六个方向并行。
- 原在GPU4/5/7等卡运行的训练和验证子进程保留，不发送停止信号。
- 优先同时覆盖六个方向，按42、43、44的顺序选择待运行项，不再设置全体种子的屏障。
  某方向已全部完成时，空位可由其他方向的独立随机种子填充；不会同时重复同一方向/种子。
- 已完成且通过精确验证的方向/种子整项跳过；训练已完成但验证未完成时只继续验证。
- 数据、模型、有效batch、早停和指标不变。全部18项精确验证通过后才冻结并最终测试。

## 为什么使用独立控制工作树

正在运行的训练目录继续保持a887bd3，不在其中执行git pull或修改批次manifest。
新控制器从GitHub的新提交放到独立工作树，所有训练子进程仍调用原目录的原始代码。
新控制器检查当前训练目录和控制工作树均干净，训练目录HEAD必须与原批次一致。

`resource_extension.json` 是不可覆盖的资源修订，记录新增GPU UUID、六并发、方向优先及独立种子调度规则、
原manifest哈希及控制器代码/提交哈希。旧声明保留初始资源，追加修订明确本次扩容。
`extension_freeze.json` 将修订绑定到原18项冻结，测试前同时核验；最终结果另存不可覆盖的来源回执。

## 部署和只读检查

在已激活的原Conda环境中，使用本次发布的新提交SHA替换 `CONTROL_COMMIT`：

```bash
cd /Data/lihb/SIR-CDR
git rev-parse HEAD
git status --short
git fetch origin main
git worktree add --detach /Data/lihb/SIR-CDR-control-six CONTROL_COMMIT
python /Data/lihb/SIR-CDR-control-six/experiments/extend_gencdr_run.py \
  --action inspect --root /Data/lihb/SIR-CDR --batch-id paper-reference-v2 \
  --gpus 0 1 4 5 6 7 --max-workers 6
```

已有同名控制目录时先核对其HEAD，不删除、重置或覆盖它。检查输出包含已完成项、仍在运行的项、
各GPU占用及资源准入信息。此命令只读，不发送信号，也不启动训练。

## 运行与交接

```bash
cd /Data/lihb/SIR-CDR
export HF_HOME=/Data/lihb/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
nohup python -u /Data/lihb/SIR-CDR-control-six/experiments/extend_gencdr_run.py \
  --action run --root /Data/lihb/SIR-CDR --batch-id paper-reference-v2 \
  --gpus 0 1 4 5 6 7 --max-workers 6 \
  >> logs/paper-reference-v2_extension_controller.log 2>&1 &
```

控制器只对经过账号、命令、目录、批次、启动时间校验的原调度主进程发送一次SIGTERM，
使用Linux pidfd固定进程身份，不按进程组发信号，不停止训练/验证子进程。
旧流水线的set-e父shell会因原调度器退出而停止后续阶段；这是交接的一部分。
新控制器取得同一个调度锁后，重新扫描直接worker以及以准确运行配置识别的内部/孤儿进程，
接管这些任务；等待其结束后通过checkpoint和验证回执确认成功，不能仅凭PID退出认为完成。

新增GPU启动前和各阶段启动前均重新检查占用。GPU变忙、资源不足、身份冲突或质量门失败时，
不抢占、不强制迁移，不跳过检查。资源准入每15秒刷新一次；没有活动任务且没有可用卡时停止，
之后用同一命令恢复。初始新增卡已被占用时，检查失败会保留原控制器运行。

## 日志和恢复

```bash
tail -f /Data/lihb/SIR-CDR/logs/paper-reference-v2_extension_controller.log
python /Data/lihb/SIR-CDR/experiments/run_gencdr_six.py --action status --batch-id paper-reference-v2
```

新启动的阶段细节在 `logs/gencdr-shared-loo-v1/paper-reference-v2/<pair>_seed<seed>_extension.log`。
原在运行的训练继续写原 `_run-validation.log`，交接后剩余阶段写 `_extension.log`。
后台任务不会因退出tail或断开终端而停止。`EXTENDED_PIPELINE_COMPLETE` 表示18项验证、冻结、
18项最终测试及结果汇总全部完成。

控制器意外退出后，使用完全相同的控制工作树和run命令恢复。进程发现和启动前写入的意图记录
避免因子进程已启动但PID回执未写入而重复启动。原始GPU affinity保持不变。
已产生测试预约但没有完整完成回执时停止审计，不重测、不补写完成回执。
不要手工删除锁、状态、修订或冻结文件；控制器锁由操作系统随进程退出释放。

## 验证边界

本地使用CPU测试模拟Linux进程与GPU资源；真实Linux进程交接和CUDA执行由服务器上的检查与实际运行验证。
没有通过SSH登录服务器。数据对齐范围仍为公开共同协议，GenCDR论文结果仅为外部参照。

## Conda Python 缺少 pidfd 接口时的修复

实测服务器为 Linux6.12、glibc2.41，Conda Python3.10.22 不暴露
`os.pidfd_open`、`signal.pidfd_send_signal`，但 libc 提供两个函数。
控制器优先使用 Python 接口，缺失时通过 ctypes 调用 libc 的同名函数，显式声明参数和返回类型，
传入零 flags、NULL siginfo，并将 errno 转为异常。始终使用 pidfd，保留账号、目录、命令、批次和
启动时间校验。没有按数字PID发送实际交接信号的回退路径。

`inspect` 和 `run` 都先对自身打开 pidfd 并发送信号0检查能力；信号0不会投递信号。
失败即停止。此检查在新增来源回执或交接之前执行。

第一次失败已经留下旧版 `resource_extension.json` 和 `extension_handoff_intent.json`。
修复版只接受精确的 d857f1c 提交及其已知脚本哈希，且所有其他资源字段必须完全相同。
`inspect` 展示恢复计划；`run` 在独占扩容控制锁内追加不可覆盖的
`extension_pidfd_recovery.json`，记录旧资源哈希、新旧控制器提交和脚本哈希。
不覆盖旧资源声明或交接意图。首次创建恢复记录只允许发生在交接完成和全局冻结之前；
已有匹配的恢复记录可在重启后核验使用。恢复记录的哈希同时绑定到扩容冻结和最终结果来源回执。

当前失败批次申请的是 GPU4/5/6/7、最多4并发，更新后继续使用这一组参数。
GPU0–3在最近的快照中被其他任务占用。更改GPU池或并发上限不属于本次兼容恢复，不能靠改命令覆盖旧声明。

新版本通过已提交到GitHub的增量Git包更新**控制工作树**，训练工作树继续保留a887bd3。
先校验包哈希与Git对象，再检查控制工作树干净，将其切到修复提交；不要在训练工作树执行git pull。
部署后先在控制工作树运行：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -p test_pidfd_compat.py -v
```

其中 Linux 专用测试临时创建两个自己的休眠子进程，强制经过 libc 路径，只终止选中的一个并验证
另一个仍运行，最后清理这两个测试进程。这不访问调度器或训练进程。本地 Windows 跳过该测试，
必须以服务器上的实际通过输出确认兼容调用。随后运行原 `inspect`，确认 pidfd 自检通过、
原训练仍在运行、已完成项不在待执行队列、GPU6仍空闲，再启动交接。
