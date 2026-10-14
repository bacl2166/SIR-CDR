# 用户侧切换到 Qwen3-0.6B

## 本次变更

Sports → Clothing、Phones → Electronics 的 `llm_*_v1.yaml` 默认模板现在使用
`Qwen/Qwen3-0.6B`，固定 Hugging Face 版本
`c1899de289a04d12100db370d81485cdf75e47ca`。
官方配置：https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json

模型为 Qwen3 文本架构，28 层，隐藏维度 1024。现有 QwenLatentReasoner
按 backbone.config.hidden_size 构造投影和记忆，因此无需修改核心计算代码。
继续使用两步连续向量隐推理、LoRA q_proj/v_proj、独立逐样本记忆和 CPF。
这里没有调用聊天模板或生成文字思维链，enable_thinking 不是本框架的参数。

物品侧 Qwen3-Embedding-8B、processed 数据、tokenizer、SID、图模块、损失、
检索与生成评分均沿用原设置。实际批次仍为 1、累积 16，梯度检查点仍开启。
这样先隔离骨干模型变化；批次或检查点优化需要另行短测并记录实验身份。

0.6B 输出位于 `artifacts/llm_cdr_v1/qwen3_0_6b/<pair>/seed42`。
原 4B 模板保留为 `configs/llm_sports_clothing_4b_v1.yaml` 和
`configs/llm_phones_electronics_4b_v1.yaml`。已有 resolved runtime YAML 和产物
均不覆盖；核心代码指纹未因本次模板修改而改变。
第三组任务仍暂缓，本次不修改其模板或启动资产准备。

## 服务器：先跑 Phones → Electronics

先确认此前训练已终止，再同步代码和初始化环境：

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
mkdir -p configs/runtime logs
```

准备固定版本的 0.6B 权重，生成独立 runtime 配置；不会重新计算物品资产：

```bash
python experiments/run_llm_cdr.py --action prepare --config configs/llm_phones_electronics_v1.yaml --resolved-config configs/runtime/llm_phones_electronics_qwen06.yaml
python experiments/run_llm_cdr.py --action preflight --config configs/runtime/llm_phones_electronics_qwen06.yaml
python experiments/run_llm_cdr.py --action smoke --config configs/runtime/llm_phones_electronics_qwen06.yaml --device cuda
```

prepare/preflight/smoke 各步成功后，再正式启动：

```bash
LOG="logs/llm_phones_qwen06_train_$(date +%Y%m%d-%H%M%S).log"
nohup python -u experiments/run_llm_cdr.py --action train --config configs/runtime/llm_phones_electronics_qwen06.yaml --device cuda > "$LOG" 2>&1 &
echo $! > logs/llm_phones_qwen06_train.pid
echo "$LOG" > logs/last_llm_phones_qwen06_train_log
tail -f "$LOG"
```

Ctrl+C 只退出 tail；后台训练继续。不要同时启动两份训练。
相同配置再次执行 train 会从本实验的 last_checkpoint.pt 恢复，保存粒度是完整
epoch；完整 manifest 则直接返回。旧 4B checkpoint 不能作为 0.6B 断点加载。
模型、LoRA、投影和记忆的维度变化，需要新的推荐适配训练，离线物品资产可复用。

重新连接后检查：

```bash
cd /root/autodl-tmp/SIR-CDR
ps -fp "$(cat logs/llm_phones_qwen06_train.pid)"
tail -n 15 "$(cat logs/last_llm_phones_qwen06_train_log)"
nvidia-smi
```

## 训练完成后的验证和推荐

确认新模型目录的 manifest.json 中 complete=true，再进行固定预算的验证集调参。
必须使用新缓存目录；旧 4B 的缓存和 selected.yaml 不能复用。

```bash
python -u experiments/tune_llm_fusion.py --config configs/runtime/llm_phones_electronics_qwen06.yaml --output artifacts/tuning/llm_phones_qwen06_seed42 --action all --device cuda
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_phones_electronics_qwen06.yaml --cache-root artifacts/tuning/llm_phones_qwen06_seed42 --output artifacts/tuning/llm_phones_qwen06_parameters_seed42 --action sweep --device cpu
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_phones_electronics_qwen06.yaml --cache-root artifacts/tuning/llm_phones_qwen06_seed42 --output artifacts/tuning/llm_phones_qwen06_parameters_seed42 --action verify --device cuda
```

只有 verified_all_metrics=true 后，固定选定参数并执行一次最终测试：

```bash
python -u experiments/tune_llm_parameters.py --config configs/runtime/llm_phones_electronics_qwen06.yaml --cache-root artifacts/tuning/llm_phones_qwen06_seed42 --output artifacts/tuning/llm_phones_qwen06_parameters_seed42 --action test --device cuda
```

最终报告：`artifacts/llm_cdr_v1/qwen3_0_6b/phones_to_electronics/seed42/test.json`。
测试后不再根据测试指标调整设置。模型迁移本身不保证提高 R@K/N@K；需要报告
速度、显存、模型版本、训练配置和最终指标，而非仅比较参数量。

Sports 使用对应模板及独立的 llm_sports_clothing_qwen06.yaml、输出和调参目录，
在 Phones 任务之后串行执行。

## 验证边界

### 学术协议与效率记录

- 当前每条评测样本只有一个正例，R@K=HR@K，N@K=NDCG@K；迁移不修改这些定义。
  R 与 N 的差距反映命中位置折扣，不通过缩放或重定义指标使两者接近。
- 验证集选择 checkpoint 和推理参数，完整保留所有搜索记录、失败检查和原始指标。
  不根据测试结果重新调参，也不删除测试锁后重复选择。
- seed42 是单次探索实验；正式结论应在相同协议下使用多个种子，报告均值和标准差。
  比较模型大小时保持资产、推理步数和搜索预算一致；批次优化作为单独变更披露。
- 与 GenCDSR 比较前需要一致的用户群、切分、历史可见范围、候选目录与过滤规则。
  仅跨域名称相同不能证明能直接与论文表格进行公平对比。
- 记录固定模型 SHA、库版本、GPU、batch/accumulation、实际完整轮数、最佳轮次、
  训练/验证耗时和峰值显存。在预热后用相同样本区间计算 samples/s，不能将聊天
  tokens/s 或参数量之比当成本框架的实测提速倍数。
- 本次保留原批次和检查点设置以隔离模型变化；0.6B 预计降低骨干开销，但目录图
  和原型仍需计算，实际吞吐及推荐质量由服务器测量决定。

### 本地测试的范围

新增回归测试使用一层随机初始化的真实 Qwen3 + PEFT，覆盖 0.6B 的非标准
query-head 宽度关系、两步前向、投影/记忆/LoRA 梯度、冻结基座及错误维度的
compact checkpoint 拒绝加载。下载与生产加载的边界用本地小模型替代。
这些测试不代表预训练 0.6B 在 3090 上的实际吞吐、完整训练显存或推荐质量。
真实模型检查和最终指标由服务器上述流程产生。
