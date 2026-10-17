# Qwen3-0.6B 推荐效率实验

最终评测已改用独立的目录快照及严格评分核验，详见
[评测恢复与实验披露](llm_snapshot_evaluation_zh.md)。下述启动命令仍适用，
完成的训练会复用；旧的原生评分报告不覆盖。

## 已实现的改动

1. **实际批次 4、累积 4、验证批次 4。** 名义有效批次仍为 16。
   用户侧两步隐推理、逐样本记忆、LoRA、CPF、检索/生成损失继续保留。
   实际批次增大后，每 16 个样本的全目录图计算由 16 次减为 4 次；
   这不是整体提速 4 倍的承诺。
2. **快速配置关闭梯度检查点。** 用显存换取减少反向重计算。3090 上的
   完整吞吐和显存需要实际短测，不能用 smoke 或参数量作保证。
3. **验证目录缓存。** LLM 专用 `LLMSIRCDR` 在一次固定权重验证中只计算
   一次物品投影、图表示和原型；退出作用域即释放。禁止在缓存中进入训练、
   启用 autograd 或加载新权重。训练每批仍正常计算可训练图表示和梯度。
4. **设备端损失累计。** 取消逐批逐损失的 Python float 转换；epoch 结束
   统一取回加权均值，保留非有限总损失/梯度检查。日志新增样本吞吐、
   训练耗时和训练峰值 CUDA allocated 显存，便于估算整个实验耗时。
5. **短命令串行入口。** 先 Phones→Electronics，再 Sports→Clothing；
   每组自动训练、验证调参、一致性校验和最终测试。任一步失败即停止。
   验证调参打分原本已缓存物品状态，该机制继续保留。

新增模型、统计辅助代码都纳入训练身份 SHA256。原 `*_v1.yaml`、4B
模板、Text v4 核心源码、物品 embedding、SID、数据切分不变。旧模型文件
保留，但新版训练代码身份已变化，不自动接受旧 checkpoint。没有兼容绕过。

## 学术边界

用户允许非完全等价的效率实验。实际批次变化会改变 CPF 防坍塌方差项
的跨样本统计和 dropout 轨迹，不能称旧实验的精确数值复现。独立目录、
训练签名记录这些差异。不能保证推荐指标一定上升。

保持全部训练/验证/测试样本、历史长度、两步推理、损失权重、学习率、
验证频率、早停和指标定义。没有通过减少测试用户、缩小评价目录或删除
正例提高数字。只用验证集调融合参数，一致性验证通过后冻结设置并测试。
已有测试结果或失败的测试预留仍由原有严格身份/锁机制保护。

第三组注册名仍是 `douban_book_movie`；其最终协议与训练资产尚未完成，
快速入口明确拒绝用其他资产替代。当前结果仍不是 GenCDR 同协议比较结论。

## 服务器操作

先确认旧任务已停止，再更新：

```bash
cd /root/autodl-tmp/SIR-CDR
pgrep -af 'python.*experiments/'
git config --local core.sshCommand "ssh -i /root/.ssh/id_ed25519_github_sircdr -o IdentitiesOnly=yes"
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
```

准备两组 runtime 并做一次必要的资产/梯度检查：

```bash
python -u experiments/run_llm_fast.py --action prepare
```

默认设置实际批次 4、累积 4、验证批次 4、检查点关闭，配置和输出自动
带上 `qwen3_0_6b_fast_b4_eb4_gc0` 标签，与此前输出隔离。
准备成功后后台启动完整两组流程：

```bash
LOG="logs/llm_fast_$(date +%Y%m%d-%H%M%S).log"
nohup python -u experiments/run_llm_fast.py --action run > "$LOG" 2>&1 &
echo $! > logs/llm_fast.pid
echo "$LOG" > logs/last_llm_fast_log
tail -f "$LOG"
```

Ctrl+C 只退出 tail。不要重复启动两份任务。重连后查看：

```bash
cd /root/autodl-tmp/SIR-CDR
ps -fp "$(cat logs/llm_fast.pid)"
pgrep -P "$(cat logs/llm_fast.pid)" -a
tail -n 15 "$(cat logs/last_llm_fast_log)"
```

相同设置再次执行 run 会恢复已保存的完整 epoch；已有完整训练直接
复用。若已有 test.json，则直接进入原有身份校验的测试结果复用入口，
不会重新打开验证调参。未完成或不匹配的测试预留会报错，不覆盖。

最终指标位于：

```text
artifacts/llm_cdr_v1/qwen3_0_6b_fast_b4_eb4_gc0/phones_to_electronics/seed42/test.json
artifacts/llm_cdr_v1/qwen3_0_6b_fast_b4_eb4_gc0/sports_to_clothing/seed42/test.json
```

分别分页查看 `metrics` 中的 R@5、N@5、R@10、N@10：

```bash
python -m json.tool artifacts/llm_cdr_v1/qwen3_0_6b_fast_b4_eb4_gc0/phones_to_electronics/seed42/test.json | less
python -m json.tool artifacts/llm_cdr_v1/qwen3_0_6b_fast_b4_eb4_gc0/sports_to_clothing/seed42/test.json | less
```

日志 `samples` 为真实样本进度，`batch` 为微批次数；Phones 的 batch
总数约从 151685 变成 37922 是合批的结果，没有删除数据。
首次更新包含初始化开销，稳定后 samples_per_second 更适合估计剩余时间。

## 可选：一次短测后再选实际批次

只在无其他 GPU 任务时执行；无需反复测试，不自动启动正式训练：

```bash
python -u experiments/run_llm_fast.py --action benchmark --pairs phones_to_electronics
```

输出 `logs/benchmark_phones_to_electronics_时间.json`，比较实际批次
1/4/8/16 与检查点开关，名义有效批次均为 16。每候选预热 1 次、测量
3 次完整 optimizer update，包含前后向、梯度累积、裁剪和更新。
测速只访问训练样本、使用临时模型、无正式 checkpoint，不评测模型质量；
少量固定样本循环仅用于计时，正式训练仍用全部数据。报告包括代码/资产/
配置身份、设备、吞吐与显存；OOM 候选明确标记失败。

短测的样本前缀不代表所有历史长度。若需要更可靠的测速区间，直接使用
`benchmark_llm_cdr.py --measure-updates` 增加测量，不能宣称一次短测是
完整训练速度保证。

如实测批次 8 更合适，准备和运行时使用同一组参数：

```bash
python experiments/run_llm_fast.py --action prepare --batch-size 8
python -u experiments/run_llm_fast.py --action run --batch-size 8
```

其累积自动变成 2、输出标签变成 `qwen3_0_6b_fast_b8_eb4_gc0`。这组命令
是选择替代配置，不能与默认 run 同时运行。显存不足可选
`--checkpointing on`，准备和运行均需带同样参数，输出标签自动隔离。

## 本地验证范围

已完成真实小型随机 Qwen + PEFT 的 CPU 回归，包括验证缓存排名/指标
一致性、缓存退出和梯度安全、加权损失统计、断点恢复、测试锁以及测速
不写正式产物。另用合成 fixture 验证新入口训练→调参→校验→测试流程。
这些验证不等同于 3090 上的预训练 Qwen3-0.6B 实测提速或推荐指标。
