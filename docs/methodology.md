# 数据、训练与审计方法

## 继续预训练

源数据为固定版本的 OpenWebMath。先准备小样本、验证输入管线，再扩充到约 5B input token。DeepSeek 主线对同一文档选择进行重新分词，并保留分片与来源定位信息。确切版本、文档数、token 数和选择策略见 `data/processed/*/manifest.json`。

所做的数据处理包括：固定源 revision 和文件哈希；确定性采样与预算；文本规范化后的精确 hash 和 URL 去重；固定训练 / 验证划分；将文档文本加 EOS 后连续拼接为 token 流；构造 2,048 token block；保存和检查尾块、训练顺序、epoch 游标与恢复位置。连续拼接采用普通 causal attention，未对不同文档增加隔离 mask。

训练 10B input token 约为 5B 数据的两次遍历。input token 与参与 next-token loss 的 supervised token 分开计数，尾块与预测移位使二者不完全相等。

Qwen 对照中先后测试了原生执行、优化执行、从初始权重长跑、学习率干预及验证集分层 / seen-unseen 分析。DeepSeek 主线从约 0.5B token 观察开始，再扩充数据和训练预算至 10B；历史完整 resume 链保留在对应 setup 目录。

## SFT

输入为 GSM8K 和 MATH 的训练集，固定 512 题验证集，最终训练样本 14,459 条。处理包括 Unicode NFKC、大小写与空白规范化后的题目精确去重；排除精确规范化题目与测试集重叠；保留 rejection 与隔离统计。没有进行语义去重。

GSM8K 使用 `Q: ...\nA:` 格式，移除计算器 `<<...>>` 标记并追加统一最终答案行；MATH 保留原解题过程与 boxed 答案。追加 EOS、不插入 BOS、不做 packing。prompt 与 padding 的 label 为 -100，仅答案和 EOS 参与 loss。最大长度 4,096；这一批数据没有超长训练样本被截断。

训练 3 epoch，共 1,356 steps；10,866,690 input token、7,408,092 supervised token。最佳验证 loss 为第 904 step 的 0.7081，后续 RL 从该检查点开始。

## DAPO 与采样优化

每题 8 个回答，按严格数值答案标记判分，正确 +1、错误 -1；1,536～2,048 token 使用 soft length penalty。筛选同时包含正确和错误回答的组，每轮使用 32 组、256 条回答。population std 归一化组优势，token 加权 PPO objective，clip lower 0.2 / upper 0.28，无 KL loss、critic 或训练 reward model。

每轮先重新计算整批 old log probabilities，然后划分 4 个 64-response minibatch；microbatch 为 16。生成使用当前权重；训练重新前向反向，生成 KV cache 不参与反向。完成回答移出解码 batch，但当前版本没有实现持续补题的完整 continuous batching。

历史路由为每批构造确定性的候选池，对每题历史正确率做 EMA 更新。25% 回答组从池中随机探索，75% 优先选择历史上仍有正确 / 错误分歧的题。被跳过的 prompt proposal 不执行推理，单独计数。

另行实测了按两条已完成正确答案的长度中位数和 batch 已完成长度 p90 取消长回答的策略。取消会强制判错，并记录 raw correctness、人工产生的混合组与取消数。此项没有在当前正式训练中启用。

候选组上限曾为 512，后改为 2,048。每 100 轮保存历史权重并验证；每 10 轮保留滚动完整恢复状态。预算耗尽时，模型尚未执行本轮更新，代码会还原这一轮起点的数据游标、难度历史和 CPU / CUDA RNG，并保存完整权重与 Adam 状态再退出。

## 评测与计数

完整测试使用固定的官方 GSM8K / MATH few-shot prompt、greedy 单答案，公开对比表统一 1,024 token。针对原来截断样本的 2,048 token 补测单独报告；它不是一份独立全量同配置评测。符号判分保留超时与错误，均计入原测试集总分母。

长 RL 的 256 题验证使用 held-out GSM8K、zero-shot、EOS-only、2,048 token，不能与完整 few-shot 测试成绩直接比较。

- `candidate_groups`：实际生成的所有组。
- `effective_groups_found`：筛出的混合对错组，包含同批多出的有效组。
- `effective_groups_used`：实际参与当前权重轨迹更新的组，每个完成轮次 32 个。
- `gradient_updates`：optimizer 更新次数，每个完成轮次 4 次。
- `training_order_cursor`：历史路由启用后指候选 proposal 游标，不再等于实际生成组数。
- `dapo_execution_ledger.json`：记录失败后回退而丢弃的正式更新分支；将其中的废弃计数加到当前轨迹计数，可得到包含这些分支的执行总量。短跑后丢弃的 benchmark 更新另行标注。

## 发布快照

保留每个实验的原始数值、stdout、源代码快照、配置、检查点元数据、图表和审计报告。权重、Adam tensor、数据正文、token 数组、HF / 编译缓存、虚拟环境、第三方 vendor 与原始 rollout 解答不在发布包中。

rollout 导出为每组的样本数、正确性向量、token 长度、奖励、结束原因、接受标志、题目 hash 和来源 ID。逐题评测输出移除题目 / 生成正文与 token ID，保留 hash、长度、类别和正确性。失败分支仅保留相对其恢复点之后的 rollout 指标，避免复制整份主线历史。

训练在快照整理期间继续运行。每个文件按打开时的字节边界读取；`EXPORT_MANIFEST.json` 记录快照时间窗口和原始 / 导出 SHA256。这不是整个文件系统同一瞬间的原子快照。历史 `status=running` 仅表示记录当时的状态。
