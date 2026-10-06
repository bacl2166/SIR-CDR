# SIR-CDR：Qwen 用户侧两步连续隐推理（llm-cdr-v1）

本版本实现用户确认的组合：复用 Qwen3-Embedding-8B 离线物品资产，
引入 Qwen/Qwen3-4B-Instruct-2507，使用 LoRA 和结构投影，维护独立持久记忆，
执行两步语言模型隐推理。训练和实际数据验证在远端服务器进行。

## 实现边界

- 旧 Text v4 模型、配置、训练入口及其代码指纹文件不修改。
- 新模型构造现有 TextSIRCDR，再用 QwenLatentReasoner 替换 reasoner。
- 三个输入是用户上下文、跨域信号、目标私有信号。投影到 Qwen 隐藏维度，
  加入类型向量和 latent seed，形成 4 个初始连续 token。
- 第一步经过真实 Qwen transformer；门控后的 thought 作为第 5 个连续 token，
  再经过 Qwen，得到第二步结果。输出投影回原来的 128 维 shared/private 接口。
- 第一版没有原始商品历史文本输入，也没有 CoT 课程训练。结构空间到 LLM 空间
  的适配依靠投影、LoRA 和推荐损失，因此性能必须验证。
- 使用 AutoModel 的 Qwen3Model，不加载语言词表输出头。SID 解码器仍为原 GRU，
  最终排序仍默认 hybrid。这是用户侧 LLM 隐推理，不是完整纯生成迁移。
- CPF 保留单轮预测监督；首版不再把两步 Qwen 外套两轮 CPF，避免默认四次计算。
- 只实现 BF16 LoRA，不含 4-bit QLoRA。资产生成结束后不同时加载 Embedding-8B。

## 独立持久记忆

令 h 为 Qwen 当前最后位置隐状态，m 为每个样本独立的记忆：

```text
n = LayerNorm(h)
r, f, w = sigmoid(Wr n), sigmoid(Wf n), sigmoid(Ww n)
thought = (1 - f) * h + r * m
m = LayerNorm(m + w * thought)
```

记忆每个 forward 初始化为零，在两步之间保留，不跨用户/批次保留。
第一步 thought 直接作为第二步输入，不 detach。第一步记忆也参与第二步读取。
最后一步更新后的 memory 无后续读取；最终输出来自最后一步 thought。
门采用零权重初始化，read bias=0、forget/write bias=-1；是本实现的保守初始化，
不宣称复制原论文所有超参数。

不使用 KV cache；第二步重算短前缀，避免缓存梯度截断。骨干权重冻结，但前向
不包 no_grad，因此损失能传回 LoRA、结构投影和门控。梯度检查点使用 non-reentrant。

## 文件职责

| 文件 | 职责 |
|---|---|
| cdr_framework/llm_config.py | 独立配置、LoRA 参数、SHA 校验、单样本对比损失保护 |
| cdr_framework/llm_reasoner.py | Qwen 加载、PEFT、结构投影、持久记忆、连续两步 |
| cdr_framework/llm_training.py | 资产/代码/依赖身份、累积梯度、紧凑检查点、恢复、评估 |
| experiments/run_llm_cdr.py | prepare/preflight/smoke/train/evaluate CLI |
| configs/llm_*_v1.yaml | 三个方向模板，输出到 artifacts/llm_cdr_v1 |
| tests/test_llm_*.py | 真实微型 Qwen3/PEFT 的 CPU 离线测试 |

## 训练约束与复现

微批次为 1、累积 16 次。梯度按累积组实际样本数归一化，末尾不足一组仍更新。
累积不会制造批内负样本，因此 contrastive_alignment=false，使用现有余弦对齐；
单样本批次开启批内对比损失会报错。该差异应在对照实验中披露。

检查点只保存非骨干状态和可训练 LoRA，恢复时严格检查完整键集合。
冻结底座按固定 Hugging Face commit SHA 重新加载；必须保留该缓存。
身份包含配置（含 label_smoothing）、资产、依赖版本和代码哈希。训练恢复需完全
匹配；跨操作系统/库版本不保证逐位相同。恢复粒度为已保存的 epoch 边界，
中途退出会从上一完整 epoch 重新开始。

验证集选择 NDCG@10。当前模板固定 zscore + 生成权重 0.5，未经数据优化。
若调参，只在 validation 上进行；最终 test.json 在不同模式之间也共享唯一锁，
评估身份变化则拒绝覆盖。旧 Text v4 融合扫描脚本不能加载此类 compact checkpoint。

## 服务器操作

在已具有三组 processed/tokenizer 资产的服务器操作。先更新：

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
conda activate sir-cdr
```

依赖使用仓库 requirements.txt 的 Transformers/PEFT 组合；不要为了这次改造升级
正在使用的旧实验环境。已有环境先检查：

```bash
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

若缺少依赖，在独立环境按 requirements.txt 安装。测试环境版本与生产版本的差异
需要记录；首次实际模型 smoke 必须通过后再正式训练。

模型下载及 SHA 固定（Sports 示例；下载约为 4B 权重规模）：

```bash
export HF_HOME=/root/autodl-tmp/hf_cache
python experiments/run_llm_cdr.py --action prepare --config configs/llm_sports_clothing_v1.yaml --resolved-config configs/runtime/llm_sports.yaml
```

prepare 仅准备新的 4B 用户侧模型，不重新计算 Embedding-8B 或 tokenizer。
需要联网；支持既有 HF_ENDPOINT 环境设置。遇到网络错误保留缓存后重试。
生成的 YAML 使用服务器绝对资产路径和固定模型 SHA，应与训练结果一并归档。
已有 resolved YAML 内容不一致时拒绝覆盖；新的设置使用新文件和新输出目录。

先检查资产，再执行一批真实 4B 前向/反向（不执行 optimizer.step）：

```bash
python experiments/run_llm_cdr.py --action preflight --config configs/runtime/llm_sports.yaml
python experiments/run_llm_cdr.py --action smoke --config configs/runtime/llm_sports.yaml --device cuda
```

smoke 输出 projection/memory/LoRA 梯度、排名形状和峰值 allocated 显存。
它不是完整训练显存证明：AdamW 首次更新还会分配优化器状态。

先完成一个训练 epoch 和验证，并保存可恢复检查点：

```bash
mkdir -p logs
nohup python -u experiments/run_llm_cdr.py --action train --config configs/runtime/llm_sports.yaml --device cuda --max-epochs-this-run 1 > logs/llm_sports_trial.log 2>&1 &
echo $! > logs/llm_sports.pid
tail -f logs/llm_sports_trial.log
```

确认该进程完成后，继续训练（不可同时启动两个相同输出目录的进程）：

```bash
nohup python -u experiments/run_llm_cdr.py --action train --config configs/runtime/llm_sports.yaml --device cuda > logs/llm_sports_train.log 2>&1 &
echo $! > logs/llm_sports.pid
```

独立验证；冻结所有设置后再运行测试：

```bash
python experiments/run_llm_cdr.py --action evaluate --config configs/runtime/llm_sports.yaml --split validation --device cuda
python experiments/run_llm_cdr.py --action evaluate --config configs/runtime/llm_sports.yaml --split test --device cuda
```

Phones、Books 使用各自 llm_*_v1.yaml 模板和不同 resolved 文件。串行运行，
不要在单卡同时运行三个 LLM 训练任务。

## 验证限度与研究报告

CPU 测试使用随机初始化的小尺寸真实 Qwen3Model 和 PEFT，验证计算路径、梯度、
批次隔离、完整恢复、标签独立排序和检查点保护。模型下载边界用本地 fixture 替代。
这些测试不证明真实 4B 在 3090 上的显存、速度或推荐效果，也不产生真实推荐指标。

2026-10-07 本地验证：完整 145 项测试通过（原 133 项＋新 12 项）；环境为
PyTorch 2.11.0+cpu、Transformers 5.15.1、PEFT 0.20.0。包括 BF16 小模型的
non-reentrant 梯度检查点反向测试。仓库原 requirements.txt 版本未改动，
该较旧依赖组合未在本次本机环境重新安装测试，远端须先执行 smoke。

BF16 4B 权重约 8GB 是权重估算；训练还需 LoRA/门控/其他参数、梯度和优化器。
不得根据该估算保证 24GB 能完成全部任务。Books 的全目录图和损失仍可能较重。

先冻结协议，比较旧模型、LLM 一步、LLM 两步（步数不同必须新输出目录），
再多种子验证。两步相对一步包含额外计算，需同时记录参数量、运行时间和显存。
未实现原 AGCLR 的语言 CoT 课程；只能称 AGCLR-inspired recommendation adaptation。
