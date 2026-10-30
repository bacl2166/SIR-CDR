# L20旧调参研究加入7号卡

适用原研究 `sports-clothing-ndcg-v1`，训练源码a887bd3、原调参工具8644017。新增UUID通过命令行明确指定。原始study.json、resource_policy.json、工具源码、运行配置和已有affinity.json一律不改写；只有独立、版本化资源扩展记录新增卡。并发上限仍为2，主机RAM预算仍为64GiB、每任务24GiB；目标仍为NDCG@10，参数候选和训练/验证预算均不变。

## 接管顺序

1. inspect只验证并展示当前控制器、worker、扩展后的资源池和新增卡的即时准入情况，pidfd只用信号0探测自身能力。
2. run在研究级resource_extension.lock下冻结扩展源码、原研究/资源哈希、初始完成结果和旧控制器身份。
3. 仅用稳定pidfd对本人、准确匹配脚本/root/study/start_ticks的旧控制器发SIGSTOP，防止新派发。训练worker不接收任何控制信号；原父进程仍存活，因此worker可继续运行。
4. 显示DRAINING，等待当前worker自然结束，并按原工具完整校验最佳模型和完成记录。此阶段7号卡还不会启动新任务，也不会占用它的显存。
5. 先封存retiring.json，再只向已停止的旧控制器发SIGTERM和SIGCONT，让它退出并释放原controller.lock。没有数值PID或进程组信号回退。
6. 新调度器取得原controller.lock，核实无活动旧worker，再开始派发。资源池变为旧5/6号卡加7号卡，最多2个worker。已完成任务验证后跳过，旧任务保留原GPU绑定；新未绑定任务可以分配给7号卡。
7. 最终冻结显式引用原资源记录与新增资源历史；不执行测试集评估。

新增卡每次实际派发前仍须满足：无计算进程、利用率不超过5%、至少12GiB空闲显存，以及原RAM准入规则。不能保证等待当前训练结束时7号卡仍空闲；若被其他任务占用，调度器继续等待，不抢占。

## 更新和运行

从GitHub分支 `codex/sports-hpo-gpu-extension` 获取固定提交，在单独的 `/Data/lihb/SIR-CDR-hpo-gpu7` worktree运行。不得更新 `/Data/lihb/SIR-CDR-sports-hpo` 或当前训练源码目录的HEAD。

```bash
source /Data/lihb/miniconda3/etc/profile.d/conda.sh
conda activate /Data/lihb/envs/sir-cdr
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /Data/lihb/SIR-CDR-hpo-gpu7

# 本地Windows测试跳过的真实pidfd用例在L20 Linux上运行。
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -p test_sports_gpu_extension.py -v

CUDA_VISIBLE_DEVICES="" python experiments/extend_sports_hpo.py \
  --action inspect --root /Data/lihb/SIR-CDR \
  --add-gpu GPU-ea76e615-5ac1-0205-d9e3-550860611755

nohup python -u experiments/extend_sports_hpo.py \
  --action run --root /Data/lihb/SIR-CDR \
  --add-gpu GPU-ea76e615-5ac1-0205-d9e3-550860611755 \
  >> /Data/lihb/SIR-CDR/logs/sports-clothing-ndcg-v1_gpu7.log 2>&1 < /dev/null &
```

日志DRAINING表示正在保留当前训练、等待其结束；DISPATCHED附GPU UUID才表示新任务实际派发。status动作与inspect相同，但不需要--add-gpu；原工具status也仍可读完成数量。

## 中断与恢复

若新协调器意外退出，保留全部文件，使用同一版本、同一extension-id `gpu7-v1` 和相同参数重新run。它会核对身份，继续等待存活worker，不重复训练已完成模型。退役后的后续worker若未完成，则记录历史并在原任务锁和GPU绑定下从匹配检查点恢复。已记录为完成的模型文件缺失或哈希改变时拒绝运行，不能悄悄重训覆盖。

若协调器已退出、且退役尚未开始，可执行 `--action resume-old --root /Data/lihb/SIR-CDR` 恢复被暂停的旧控制器。动作需要独占研究级扩展锁；协调器仍运行时会拒绝。retiring.json一旦存在就禁止恢复旧控制器，因为它可能已有待处理SIGTERM；此时应重新run完成接管。恢复旧控制器会将当前extension-id标记取消，之后若再扩展需新的extension-id。

不要删除锁文件、修改冻结JSON、强杀训练worker或同时启动第二个旧控制器。新接管过程只改变计算资源安排，不能据此宣称推荐质量提高或已完成双NDCG新研究。
