# 新 Amazon 数据对的完整推荐实验

入口为 experiments/run_amazon_recommendation.py。它独立等待四个资产 ready.json 全部出现，然后逐项验证原封存文件、Phones 参考配方及 ALIGNMENT_OK 绑定。现有资产准备控制器继续运行；本入口不会重复启动下载、转换、嵌入或 SID 训练。

## 实验声明

Food→Kitchen、Kitchen→Food、Beauty→Electronics、Electronics→Beauty，每个方向运行种子 42、43、44，共 12 次推荐训练。数据采用当前 gencdr-shared-loo-v1，沿用 Phones 的训练配方：Qwen3-0.6B 固定 revision、batch size 8、梯度累积 2、学习率 0.0001、检索温度 0.1、生成和检索损失权重各 1、CPF 权重 0.1、最多 30 轮、patience 5。全部原模型模块、训练样本、验证集和测试集保持原实现。

本次新实验的 checkpoint 选择指标为 (NDCG@5+NDCG@10)/2，学习率调度和早停使用同一标量；完全同分时保留先前轮次。训练记录额外保存 NDCG@5_10_mean，原始 HR/NDCG 数值原样保留。适配器哈希进入 checkpoint identity，已有 Phones 等旧实验不受影响。此处采用用户此前选择的双 NDCG 目标；与旧 HR@10 选模协议的差异必须披露。

验证集使用原有限融合参数网格：运行时锚点加 16 个候选记录（可能与锚点重复），最大候选池 1000，记录所有尝试。选择顺序为双 NDCG 均值、HR@5/@10 均值、NDCG@10，再按较小候选池和较小权重等确定同分顺序。所有指标完整报告，不能仅挑单项最优数值拼成一组。缓存与独立重算必须在原始分数、排序及所有指标上完全一致。

## 执行顺序

1. 持久保存本轮 request；等待所有资产封存，保存实际资产/配置声明。
2. 检查本地固定 revision 的 Qwen 缓存和至少 32 GiB 空闲磁盘。
3. 每方向运行一次正式参数的 smoke，检查梯度、排序及显存峰值，不执行 optimizer.step。
4. 自动运行 12 次训练及各自验证配置选择和精确复核。已完成且身份匹配的训练或验证被复用；部分训练由原 last_checkpoint.pt 恢复。
5. 收齐 12 份验证绑定，固定 checkpoint、runtime、ready、snapshot、cache、search、verification、smoke 和 GPU affinity；写入 validation_freeze.json 后才能评估测试集。
6. 在各自验证时的物理 GPU 上评估测试集，保存完整测试身份和不可覆盖的完成收据。
7. 保存逐种子结果和每方向三种子的均值、样本标准差，输出 AMAZON_RECOMMENDATION_COMPLETE。

不读取测试指标参与早停或配置选择。预检可核对测试文件身份，但模型测试打分在全批次冻结后才释放。失败或未完成的测试预约需要审计，不能删除后静默重测。新数据对尚无同协议 GenCDR 的运行结果，结果文件不宣称严格对齐论文或已经超过 GenCDR。

## 共享服务器调度

最多并行两个工作进程，每个预留 24 GiB 主机内存，另保留 16 GiB 余量。自动选择无计算进程、利用率不超过 5%、空闲显存至少 12 GiB 的卡；正式训练/验证/测试还必须满足 smoke 峰值的原安全预算。每次启动前重新查询 GPU。资源不足每 60 秒重查；运行失败停止新派发并等其他已启动任务结束。CUDA 环境仅暴露选定 UUID 为 cuda:0。

同一个种子的训练、验证缓存、精确复核和测试绑定原 GPU UUID。该卡暂时被占用时等待它释放；其他尚未绑定的种子仍可分配到空闲卡。这个约束保留原缓存中物理设备身份，保证恢复不会静默改变评估数值。

若仅控制器退出而其工作进程仍存活，重启入口会先等待本人同批次的旧工作进程完成，再恢复派发，避免新旧控制器合计突破两任务并发上限。资产的最后一个 ready 文件出现后，还会等待原资产控制器释放写入锁，再验证全部封存记录。

在独立服务器代码目录检出固定发布提交后，可立即 nohup 启动，即使最后一方向资产还没完成。默认 batch-id 是 amazon-expansion-dual-ndcg-v1。所有新记录位于：

```text
/Data/lihb/SIR-CDR/artifacts/amazon_recommendation/amazon-expansion-dual-ndcg-v1/
  request.json
  declaration.json
  jobs/<pair>/seed<seed>/runtime.json
  jobs/<pair>/seed<seed>/training/
  jobs/<pair>/seed<seed>/validation/
  logs/<pair>_seed<seed>_<phase>.log
  status/<phase>/
  validation_freeze.json
  test_completions/
  results.json
```

执行：

```bash
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/Data/lihb/envs/sir-cdr/bin/python -u experiments/run_amazon_recommendation.py \
  --root /Data/lihb/SIR-CDR \
  --alignment /Data/lihb/SIR-CDR/artifacts/diagnostics/amazon-expansion-phones-alignment-v1/alignment-20261123T092552327241Z.json
```

状态：

```bash
CUDA_VISIBLE_DEVICES="" /Data/lihb/envs/sir-cdr/bin/python experiments/run_amazon_recommendation.py \
  --root /Data/lihb/SIR-CDR --action status
```

本地 CPU 小型端到端检查验证策略与不可覆盖测试结果；真实 L20 的正式规模 smoke、显存、耗时和推荐效果需在服务器执行确认。进程通过 nohup 后台运行，断开 SSH 不会因终端关闭而结束；服务器停机或进程退出后需要重新执行同一批次入口恢复。
