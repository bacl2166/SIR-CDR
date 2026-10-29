# GenCDSR参考共同协议v2：先完整域过滤，再取共有用户

v1服务器诊断：原始Sports296337条、Clothing278677条，初始共有用户3908人、
共79111条交互。先取交集再迭代双域user5/item3使数据经15轮删除缩至13人、213条。
该规则是我们自行声明的重建选择，实测不适合当前实验。未在这13人资产上训练或选参。
保留v1提交dfafeff9、工作树和所有诊断记录，不能覆盖或当作作者处理结果。

## 修订规则

1. 从原始评论读取完整两个域，先各自在完整域内独立迭代item>=3/user>=5至稳定。
   删除某人在一域的事件，不会连带删除其另一域事件；计数为事件次数，保留重复交互。
2. 在过滤完成的两个域之间按真实用户ID求交集，只取这些用户的完整剩余事件。
   不在交集中再次要求item>=3，因此最终共同子集中某些物品可能只有1或2次交互。
3. 原始输入已经5-core时，第1步完全不删除是合法结果；不能为制造新资产而额外删样本。
   这时共同人口可能回到原来的3908人、Sports13057物品、Clothing13044物品，仍需实测。
4. 保持SIR原有因果双域样本：每域最后两次交互分别作验证、测试，源域只用训练前缀，
   时间严格早于目标标签；空源历史样本记账排除，不换留出正例。
5. 保留静态文本、物品级完整目录、已见屏蔽和全部SIR主干模块。按验证NDCG@10选模，
   先双向seed42，再43/44；微批8/累积2、30轮/patience5，固定hybrid500/0.5/none。

新schema为`gencdsr-inspired-sc-v2`，默认study为`sports-clothing-gencdsr-v2`。
部署至`/Data/lihb/SIR-CDR-gencdsr-sc-v2`独立工作树；原始数据只读自旧SIR-CDR目录。
v2不是GenCDSR作者逐条资产复现，也不复制其混合序列预训练/域适配或解码器。
作者论文表1包含非共有用户且规模不同；不能称完全对齐，论文数值仍只能外部参照。
修订依据为过滤范围和人口定义，不允许根据NDCG重新选过滤阈值或迭代停止点。

## 审计

记录raw/initial_common/每轮完整域过滤/filtered_domains/final，原始文件、读取器和转换器哈希。
CLI `audit`必须用原始文件重新执行完整过滤，核对整个过滤链和最终事件：

```bash
source /Data/lihb/miniconda3/etc/profile.d/conda.sh
conda activate /Data/lihb/envs/sir-cdr
cd /Data/lihb/SIR-CDR-gencdsr-sc-v2
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u experiments/run_gencdsr_sports.py --action convert --data-root /Data/lihb/SIR-CDR
python -u experiments/run_gencdsr_sports.py --action audit --data-root /Data/lihb/SIR-CDR
```

每个方向结果应有`raw_filter_replayed: true`。双向交叉核验附加项只检查两个已冻结输出，
其内层报告`raw_filter_replayed: false`不表示前面的原始审计失败。
代码级不带raw_root的audit只复核输出内部一致性、最终user5及绑定过滤记录，
不能用最终共有子集重新验证完整域item3，不能冒充原始过滤重放。

GPU资源首次使用仍人工确认，显式完整UUID、无计算进程且利用率0、显存>=12GiB。
旧Douban任务已停止，旧共享调度器保持暂停，Phones任务继续运行。不得恢复旧调度器来启动新实验。
先完成新CPU测试和数据审计；再按当时资源状态给GPU5/6双向训练命令。
阶段仍为embedding/tokenize/seal/runtime/smoke/train/validate，六个种子方向完成后freeze/test。
旧v1说明仅作历史，不用于v2路径和默认study。

上游参考：https://arxiv.org/html/2607.28659v1#S4.SS1.SSS1
