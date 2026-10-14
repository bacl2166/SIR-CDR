# 第三组公开 Book/Movie 资产与无需重训的 LLM 调参

默认任务为 Sports→Clothing、Phones→Electronics、GenCDSR Book→Movie。
旧 Amazon 第三组不再进入默认注册表，新的 LLM 模板为
`configs/llm_gencdsr_book_movie_v1.yaml`。前两组配置和训练身份代码文件不变。
旧 Amazon 处理配置仅供历史代码兼容，不属于新默认实验。

## 数据协议与比较边界

来源为 Applied-Machine-Learning-Lab/RecSys2026_GenCDSR，固定提交
`bace0d18f53bfeb56ab9bdd564494055b6c38db7`。公开目录是 Book_Movie，
README 未注明其 Douban 数据版本；对外写 Douban 前须核对论文/作者说明。
转换器记录来源 SHA、输入哈希和自身哈希，不宣称已精确复现论文协议。

- 保留所有公开目录物品，Book/Movie 用不同命名空间；原 ID=0 是真实物品。
- 校验单域序列与混合序列域投影一致。时间戳未公开，`timestamp` 字段仅为
  图接口的序列位置，manifest 明确它不是日期或跨用户全局时间。
- 目标域原序列倒数第二件为验证，最后一件为测试，绝不换成更容易的正例。
- 当前 SIR 要求两域历史非空，显式构造两域重叠用户共同子集；源域只使用
  目标事件之前的源域训练前缀，源域末尾两次交互不进入历史/训练图。
- 当前 SIR 过滤已见物品，因此定义新物品推荐任务：若任一原验证/测试正例
  已在该用户目标历史出现，排除该用户并计数；训练重复正例也排除。不重新
  切分，不静默改变标签。双方比较必须使用同一导出的用户/样本子集。
- 保留物品目录的 28 件 Movie 缺失文本，使用公开计数的固定域标记
  `Movie. Metadata unavailable.`，不捏造标题或属性。
- Qwen3 embedding 与 SIR tokenizer 重新计算；不把作者的 SID 辅助张量
  误作 SIR 的 latent/codebook。推荐模型必须在新的第三组上首次训练。

不能把这个子集与论文完整用户结果直接比较。GenCDSR 的不同阶段切分、
源域信息权限、历史截断、已见过滤和候选目录需要统一，并在同一子集重跑基线。
其他两组当前结果也不能仅凭数据来源相同称为与新基线完全同协议。

本地在上述固定公开提交上转换并通过 SIR 加载校验：保留823用户，训练
52,042行、验证823行、测试823行；Book目录12,426件、Movie目录16,537件。
原单域Movie公开用户为2,213，因此这个823用户子集绝不是原论文完整任务。
manifest记录1090个无双域用户、342个短序列用户、12个验证前无源历史用户、
14个无有效训练行用户以及74个重复评测正例用户的排除。三份导出文件的
positives_in_seen均为0。完整155项本地测试通过，但未运行真实GPU新资产训练。

## 服务器：更新并准备第三组

在没有训练/评测进程写输出时更新。新增脚本不修改旧 checkpoint 指纹涉及的
模块，因此可以继续加载前两组原 LLM 检查点。

```bash
cd /root/autodl-tmp/SIR-CDR
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sir-cdr-llm
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
cd /root/autodl-tmp
git clone https://github.com/Applied-Machine-Learning-Lab/RecSys2026_GenCDSR.git GenCDSR-reference
git -C GenCDSR-reference checkout bace0d18f53bfeb56ab9bdd564494055b6c38db7
cd SIR-CDR
```

先 CPU 转换检查协议与规模，再运行 GPU 资产生成：

```bash
python experiments/prepare_gencdsr_assets.py --reference-root ../GenCDSR-reference --action convert
nohup python -u experiments/prepare_gencdsr_assets.py --reference-root ../GenCDSR-reference --device cuda > logs/prepare_gencdsr_book_movie.log 2>&1 &
echo $! > logs/prepare_gencdsr_book_movie.pid
tail -f logs/prepare_gencdsr_book_movie.log
```

GPU 资产进程结束后运行 `run_gencdr_benchmarks.py --action preflight`。
不要与下面评分缓存任务同时占用单卡。

新第三组模型准备和 smoke：

```bash
python experiments/run_llm_cdr.py --action prepare --config configs/llm_gencdsr_book_movie_v1.yaml --resolved-config configs/runtime/llm_gencdsr_book_movie.yaml
python experiments/run_llm_cdr.py --action smoke --config configs/runtime/llm_gencdsr_book_movie.yaml --device cuda
```

## Sports：一次缓存，21 设置离线扫描

测试尚未评测时使用。本工具发现已有 test.json 则拒绝确认性调参。
仅加载验证样本；原训练身份哈希会读取所有输入指纹，这不等于加载测试标签
选参数。评分矩阵只包含验证候选和验证正例。

```bash
nohup python -u experiments/tune_llm_fusion.py --config configs/runtime/llm_sports_clothing.yaml --output artifacts/tuning/llm_sports_seed42 --action all --device cuda > logs/tune_llm_sports.log 2>&1 &
echo $! > logs/tune_llm_sports.pid
tail -f logs/tune_llm_sports.log
```

每 100 条样本保存一次缓存，输出完成比例。中断后执行同一命令，从缓存的
下一样本继续。完整缓存复用时不再构建 Qwen。固定搜索为 5 个生成权重
`0.1/0.25/0.5/1/2` × 4 个候选数 `100/300/500/1000` 加纯检索，总计21。
原有默认500、0.5组合已包含。原始评分截取后按当前候选池重新 zscore；
不把1000候选的标准化分数直接截取。NDCG@10择优，同分偏好较低生成权重
与较少候选。报告记录全部配置、候选召回和工具/训练/资产/依赖身份。

完整缓存存在后可只用 CPU 复查扫描：

```bash
python experiments/tune_llm_fusion.py --config configs/runtime/llm_sports_clothing.yaml --output artifacts/tuning/llm_sports_seed42 --action sweep --device cpu
```

直接模型验证缓存结果的一致性：

```bash
python experiments/tune_llm_fusion.py --config configs/runtime/llm_sports_clothing.yaml --output artifacts/tuning/llm_sports_seed42 --action verify --device cuda
```

成功生成 verification.json 后冻结 selected.yaml。确认设置与协议后才执行：

```bash
python experiments/run_llm_cdr.py --action evaluate --config artifacts/tuning/llm_sports_seed42/selected.yaml --split test --device cuda
```

测试报告仍写入原模型目录的 test.json，保持原来的唯一测试身份锁。
selected.yaml 只改变允许的推理参数；不要用它续训（manifest 记录的选择策略不同）。
若验证搜索无改善，保留结果；后续可在新输出目录开展独立目标适配/重训实验，
不得根据测试反馈继续搜索。不保证与任何基线达到特定差距。

## 删除不再需要的 Amazon 第三组

按用户要求可删除服务器的旧第三组，先列出精确路径并确认没有相关进程。
不要删除 Book/Movie 新目录，也不要删除整个 data、artifacts 或 HF 缓存。

```bash
du -sh data/processed/books_to_movies artifacts/embeddings/qwen3_8b/books_to_movies artifacts/tokenizer/qwen3_8b/books_to_movies artifacts/llm_cdr_v1/books_to_movies 2>/dev/null
rm -rf -- data/processed/books_to_movies artifacts/embeddings/qwen3_8b/books_to_movies artifacts/tokenizer/qwen3_8b/books_to_movies artifacts/llm_cdr_v1/books_to_movies
```

原始 Amazon Books/Movies gzip 若不用于其他任务，也可单独删除。数据转换和
推理缓存测试不产生真实服务器推荐结果；真实改进须由验证/最终测试证实。
