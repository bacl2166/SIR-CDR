# 当前 Amazon 正式批次的八卡自动扩容

## 范围

针对 `amazon-expansion-dual-ndcg-disk24-v1` 中 Food↔Kitchen、Beauty↔Electronics 四方向及 seed42/43/44 的剩余任务。最多八组并行，自动从物理 GPU 0..7 选择空闲卡。不重跑完成任务，不修改现有数据、SID、科学参数或模型源码。

GenCDR 仍在 SIR 全部验证、测试和汇总完成后运行，不提前抢占 SIR 阶段。新代码允许 GenCDR 启动时指定 `--max-workers 8`；默认值仍为2，旧批次配置不能原地覆盖。

## 交接方式

独立扩容入口 `experiments/extend_amazon_recommendation.py` 导入服务器上 **原来用于启动该批次的干净代码目录**，默认 `/Data/lihb/SIR-CDR-amazon-rec`。原请求、声明、runtime 配置和源码身份必须一致，否则拒绝交接。

扩容器使用 pidfd 核验自己的原控制器，暂时停止其派发，重新扫描全部当前 worker 并记录收据，再只终止原控制器。训练 worker 不接收信号、不重启；新控制器把它们计入八组并发上限，立即在其他空闲卡上启动剩余任务。失败时不再派发，已运行的兄弟任务继续到结束。扩容器异常退出后，可用相同命令重新启动，自动发现仍在运行的 worker。

调度只改变资源策略。原工作目录、原 worker 程序、验证选模、冻结、测试、汇总过程均继续使用原工具。`request.json` 和 `declaration.json` 保持原字节；新的资源声明单独保存至：

```text
artifacts/amazon_recommendation/amazon-expansion-dual-ndcg-disk24-v1/
  resource_extensions/eight-gpus-v1/resources.json
  resource_extensions/eight-gpus-v1/handoff.json
```

控制器退出不等于实验完成；完成仍以原验证绑定、测试收据和 `results.json` 为准，GenCDR 等待条件保持有效。

## 自动资源条件

- 所选卡无计算进程、利用率不超过5%，空闲显存达到原 smoke 推导值且至少12 GiB。
- 已启动任务保持原 GPU UUID；无法在别人的占用卡上强行恢复。
- 包含接管任务在内最多8组；每组预留24 GiB主机内存，按现有主机压力检查准入。
- 每次新增任务前检查 `/Data` 至少24 GiB空闲；不降低原门槛、不清理数据、不终止他人进程。
- 每30秒重查；有合格卡且资源足够就补位。最多8组是上限，不是保证8卡一直满载。

当前快照中的7号卡是可用候选；实际启动前会重新查询，不使用过期快照强制分配。

## 部署

先从 `codex/amazon-eight-gpu` 获取独立代码目录。**不要更新或删除原 `/Data/lihb/SIR-CDR-amazon-rec`**，它仍是所有训练 worker 的源码依据。

在新扩容代码目录，执行：

```bash
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python \
  -m unittest discover -s tests -p test_amazon_eight_gpu.py -v

CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python \
  experiments/extend_amazon_recommendation.py \
  --root /Data/lihb/SIR-CDR \
  --original-tool /Data/lihb/SIR-CDR-amazon-rec --action inspect

nohup /Data/lihb/envs/sir-cdr/bin/python -u \
  experiments/extend_amazon_recommendation.py \
  --root /Data/lihb/SIR-CDR \
  --original-tool /Data/lihb/SIR-CDR-amazon-rec \
  --action run --poll-seconds 30 \
  >> /Data/lihb/SIR-CDR/logs/amazon-eight-gpu-v1.log 2>&1 </dev/null &
```

预期日志先出现 `CONTROLLER_RETIRED_WORKERS_ADOPTED`，随后有 `START ... GPU=7 ...` 等派发记录。若原控制器此前已经结束，则显示 `NO_ORIGINAL_CONTROLLER` 并根据现存 worker/完成结果继续。是否确实派发到7号卡取决于启动时实时占用。

```bash
tail -n 40 -f /Data/lihb/SIR-CDR/logs/amazon-eight-gpu-v1.log
```

本扩容不会自动初始化 GenCDR。若尚未部署，可在新的干净代码目录按 GenCDR 指南启动等待器，并在首次声明时指定 `--max-workers 8`。如果已经存在旧 GenCDR `request.json`，应继续原资源配置，或另建独立批次，不覆盖旧声明。
