# 将空闲GPU加入现有Sports/Clothing调参研究

本轮资源变更：原GPU5、6、7保留，新增GPU0--4；最多六项任务并行，
主机内存预算192GiB，每任务预算24GiB。启动前、每次派发前检查原有GPU准入条件。
6、7号卡被其他用户占用时不会派发到这两张卡。

数据、SID、训练主干、参数组、随机种子、早停与NDCG@10选择规则均不修改。
当前L20研究不是双NDCG平均目标研究。保留所有旧结果和GPU绑定；已完成任务不重跑。
原计划仍为16组初筛、8组多种子确认、12个模型融合评估，不读取测试集调参。

## 部署

从GitHub拉取codex/sports-hpo-idle-pool，在单独控制目录建立detached worktree。
保留训练目录a887bd3、原HPO工具目录8644017、GPU7控制目录47bdc17不变。
在新目录执行 `bash scripts/start_sports_idle_pool.sh`。脚本执行Sports相关测试、
打印当前GPU可用情况并inspect资源计划，然后通过nohup启动新控制器。

脚本使用现有Python /Data/lihb/envs/sir-cdr/bin/python，不安装依赖、不下载模型。
唯一新调度入口为 experiments/expand_sports_hpo.py，扩展记录位于原study的
resource_extensions/idle-gpus-v2，原resource_policy.json不修改。

## 交接和观察

新调度器只暂停经过UID、命令行、脚本路径、启动时间校验的旧调度器，使用pidfd信号。
旧worker不暂停、不终止，会自然完成。日志出现DRAINING表示等待当前worker结束。
随后退休旧调度器，获取原有两个独占锁，按原政策继续调参。

日志 /Data/lihb/SIR-CDR/logs/sports-clothing-ndcg-v1_idle-pool-v2.log：
- DRAINING：等待旧任务或上次控制器退出后留下的worker结束。
- DISPATCHED：已派发某个参数组到指定GPU UUID。
- active/pending：本阶段正在执行/尚待执行的任务数。

只有看到DISPATCHED及新worker才能确认新卡已实际用于训练。
已有affinity的任务保持绑定；因此六项是上限，不承诺始终占满六张卡。
再次运行相同命令可恢复交接或未完成队列；不应改变扩展ID、控制目录或资源参数。
若控制器仍在运行，启动脚本拒绝重复启动。完成冻结后不再启动新研究任务。

状态查询（只读）：

```bash
CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python \
  /Data/lihb/SIR-CDR-hpo-idle/experiments/expand_sports_hpo.py \
  --action status --root /Data/lihb/SIR-CDR
```

## 验证边界

本地测试覆盖资源上限、稳定进程身份、交接先后顺序、未知孤儿worker阻断、
完成组跳过、GPU绑定保留以及原24次训练/12个模型选择流程。
服务器部署脚本还执行既有Linux pidfd集成测试；本机没有远程连接，不能替代服务器实测。
