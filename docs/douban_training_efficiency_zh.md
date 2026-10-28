# Douban 训练效率改进与验证

适用于共享服务器 `/Data/lihb`。原训练目录 `/Data/lihb/SIR-CDR` 继续固定a887bd3，
原控制工作树 `/Data/lihb/SIR-CDR-control-six` 继续固定363ef0b。
在独立工作树 `/Data/lihb/SIR-CDR-douban-efficiency` 部署本分支。
不在两个原目录执行git pull、切换提交、改配置或删除实验文件。

## 变更与边界

- LLM训练专用加载器仍逐条验证并保留全部训练前缀、原顺序、原标签、原历史截断长度，
  仅不存储训练前向从未使用的完整seen集合，减少Douban重复前缀的Python对象内存。
- 专用训练collate生成与旧路径相同的输入张量，不再复制seen集合。
- 训练历史长度保留在CPU供pack使用；序列mask复制到计算设备，避免GPU→CPU往返。
- 骨干、图、原型、CPF、各损失、优化器、随机打乱、有效batch、验证、早停全部保持原定义。
  不缓存可学习训练图，不跳过非有限损失检查，不改变评估精度/排序/屏蔽规则。
- 默认仍8×2。16×1只作为显式吞吐候选，CPF批内方差/dropout/浮点路径使它不是严格等价替换。
- 新源码哈希自动进入训练身份。旧检查点仍按原身份保护，不添加恢复豁免。
  新YAML只读取原始目录中的数据/tokenizer；输出写入新工作树的douban-efficiency-v1。
  原v2训练不切换版本；不同微批次方案不混成一组三种子结果。

## 先部署和CPU回归

将已发布的Git提交通过正常Git或经过sha256校验的增量bundle导入对象库，
创建独立detached工作树。代码来自同一GitHub提交，不在服务器手工修改源码。
激活既有Conda环境后，进入新工作树，执行：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

CUDA专用梯度一致性测试在这一步跳过。在明确可独占的空闲卡上，再运行
`DoubanEfficiencyTests.test_losses_and_gradients_match_legacy_on_cuda`，
例如从新工作树执行`python -m unittest discover -s tests -p test_douban_efficiency.py -v`；
运行前手工核对GPU UUID与进程，CUDA_VISIBLE_DEVICES仅设置该卡完整UUID。
该测试仅用临时小模型，不读真实验证/测试集，不写原实验输出。

## 吞吐基准：只允许空闲GPU

先运行nvidia-smi，确保选中的物理卡没有其他进程且符合使用规范。
最近提供的快照中全部卡在用，尤其GPU6与Ray混合占用，因此现在不能直接运行基准。
以下`GPU_ID`由空闲检查结果赋值，不提供占用中的默认卡号。

```bash
cd /Data/lihb/SIR-CDR-douban-efficiency
export HF_HOME=/Data/lihb/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
python -u experiments/benchmark_llm_cdr.py \
  --config configs/server/douban_book_movie_efficiency_v1.yaml \
  --device cuda --gpu-id "$GPU_ID" --checkpointing off --batch-sizes 8 \
  --input-pipeline legacy --warmup-updates 10 --measure-updates 100 \
  --output logs/douban-efficiency/legacy-b8-r1.json
python -u experiments/benchmark_llm_cdr.py \
  --config configs/server/douban_book_movie_efficiency_v1.yaml \
  --device cuda --gpu-id "$GPU_ID" --checkpointing off --batch-sizes 8 16 \
  --input-pipeline optimized --warmup-updates 10 --measure-updates 100 \
  --output logs/douban-efficiency/optimized-r1.json
```

每次测量仅使用固定种子从训练集均匀有放回采样的相同样本序列，完整计算损失、梯度裁剪
及优化器更新；模型是临时新权重，不恢复或保存推荐检查点，不访问验证或测试集。
这不是训练子集实验，正式train仍遍历全量训练前缀。记录采样序列哈希、代码/训练身份、
微批次/累积、吞吐、峰值显存和GPU前后快照。基准CPU进程RSS可由ps另行记录。

CLI要求显式物理GPU编号，在CUDA上下文创建前核验空闲并绑定完整UUID；每候选开始/结束核查
占用，发现其他进程后结果标记resource_conflict。检查不是跨用户的原子资源预约，
仍需协调使用时段；短暂出现又退出的占用可能不被两次快照捕获。

两种路径的基准报告训练源码身份相同，但input_pipeline字段不同：legacy是相同模型公式的
旧数据整理/搬运对照，并不冒称运行了原a887bd3的完整训练。

换新文件名交替重复至少3次，报告吞吐中位数与范围，不能挑最好的一次。
先比较旧8×2与优化8×2，再单独比较优化8×2与优化16×1。
基准使用新初始化模型，因此正式采用前还需在独立验证实验确认推荐质量；
16×1须按同一划分、三种子对照，预先固定NDCG@10、HR@10和可接受差异规则。
不根据测试集决定是否采用。无真实服务器测量时不声明实际加速倍数或指标无损。

## 正式实验

提供Book→Movie和Movie→Book两个独立服务器配置。它们保留原HR@10选模，
本次专注实现效率，不同时修改选模指标和训练超参数以免混淆归因。
正式训练仅使用新工作树和新输出，先smoke及验证数据/模型版本，再安排空闲GPU。
原v2的18项冻结/测试流程不受影响；本效率对照不能未经独立实验声明并入v2结果。
