# QLoRA Text-to-SQL 微调与执行评测

输入自然语言问题和数据库结构，用 Qwen2.5-Coder-1.5B-Instruct 生成 SQLite 查询。本仓库发布项目源码、测试与配置，供代码阅读与复核；**不是带模型权重、数据与运行账本的一键复现包**。

## 代码导览


请先看 **[详细代码精读：从一条样本到训练、生成与评分](docs/CODE_WALKTHROUGH.md)**。它不是只列文件名，而是按真实调用链提供：

- 固定提交的源码链接与行号，以及每段应先看哪里；
- 一条真实训练样本贯穿schema、聊天模板、301个输入token与14个监督目标；
- 关键变量的来源、类型、张量形状，以及前向、backward、AdamW更新和检查点恢复；
- 推理输入、生成结果切片、SQL后处理、可执行率与EX的实际判分代码；

下方保留概览；详细版只读源码，不需要启动训练或重新评测。


| 路径 | 用途 |
| --- | --- |
| `src/data/` | Spider 下载、按数据库划分与清洗、schema 序列化、训练特征 |
| `src/train/` | assistant-only 损失、按有效答案 token 归一化的梯度累积、QLoRA 训练及检查点 |
| `src/infer/`、`src/eval/` | 生成、SQLite 执行评分与逐题对照 |
| `src/diagnostics/` | 运行诊断、一致性与完整性校验 |
| `configs/` | 数据、提示词、训练、生成和评测配置 |
| `reports/`、`outputs/` | **仅少量脚本**：辅助工具及内容不同于现行源码的历史版本，不是新的运行入口 |

推荐从 [schema 序列化](src/data/schema_serialization.py)、[训练特征](src/data/sft_features.py)、[目标位置损失](src/train/target_loss.py)、[梯度累积](src/train/token_accumulation.py)、[首轮训练 worker](src/train/sft_round1_resource_v3_worker.py) 与 [最终 SQLite 执行器](src/eval/spider_pair_worker_decode_v2.py) 开始。程序和测试中的小型固定样例用于说明行为，不是完整训练/评测数据集。

## 核心代码走读

以“学生表里有多少名学生？”为例。假设表结构是 `students(id, name)`，参考答案是 `SELECT COUNT(*) FROM students`。这是用来读代码的例子，不是实验中的真实预测。

### 1. 输入整道题，只监督 SQL 答案

[训练样本编码，第 100–133 行](https://github.com/Actually61/text2sql-posttrain/blob/8079d64a3428a60183fcaf1e33c2b70c0c44898a/src/data/sft_features.py#L100-L133) 分别编码「问题和表结构」构成的提示词，以及「提示词 + 参考 SQL」构成的完整对话。两段编码的前缀必须一致，才能用 `prompt_length` 找准答案的起点。核心是两行：

```python
labels = [IGNORE_INDEX] * len(input_ids)
labels[prompt_length:eos_position + 1] = input_ids[prompt_length:eos_position + 1]
```

`input_ids` 始终包含整道题。第一行把所有标签设为 `-100`（不计损失），第二行只填回 SQL 的 token 和结束标记 EOS。**不计损失不等于看不见**：模型仍能看见 `students` 的表结构；训练考的是它能否据此写出 SQL，并在答案结束时停下来。

### 2. 一条条算梯度，一组才更新一次

首轮 QLoRA 冻结 4-bit 基座，只更新 LoRA 参数。为了在 8 GB 显存上保留完整样本，每次前向只放一条；通常积累 8 条后更新一次，最后一组是 4 条。

- **算梯度：**[累积函数，第 24–49 行](https://github.com/Actually61/text2sql-posttrain/blob/8079d64a3428a60183fcaf1e33c2b70c0c44898a/src/train/cache_release_core.py#L24-L49)。每条的 `loss` 是其有效答案 token 的平均损失。代码用 `weight = count / total_targets`（本条有效答案 token 数 ÷ 本组总数），再调用 `(loss * weight).backward()`。这样每个答案 token 等权；直接把每条的平均损失除以 8，会让短答案里的单个 token 占更大权重。
- **改参数：**[首轮训练入口，第 311–326 行](https://github.com/Actually61/text2sql-posttrain/blob/8079d64a3428a60183fcaf1e33c2b70c0c44898a/src/train/sft_round1_resource_v3_worker.py#L311-L326)。整组反向传播完成后，程序检查并裁剪梯度。直到 `optimizer.step()` 才用累计梯度更新 LoRA 参数；前面的 `backward()` 不改参数。

### 3. SQL 能运行，不代表答对了

[SQLite 评分，第 187–207 行](https://github.com/Actually61/text2sql-posttrain/blob/8079d64a3428a60183fcaf1e33c2b70c0c44898a/src/eval/spider_pair_worker_decode_v2.py#L187-L207) 把参考 SQL 和预测 SQL 放到同一数据库分别执行。第 191 行判断预测是否成功执行；两条都成功后，第 199 行才用 `result_eq` 比较查询结果，而不是逐字比较 SQL。

假设表里有两名学生 Alice 和 Bob。参考 SQL 返回数字 `2`；如果模型写成 `SELECT name FROM students`，它也能运行，但返回的是两行姓名。这道题就是**可执行，但 EX = 0**。反过来，即使两条 SQL 在这份数据库上结果相同，也不能证明它们在所有可能的数据上都等价。

## 环境和外部材料

- 已记录实验环境：Windows、Python 3.11.9、PyTorch 2.11.0+cu128、Transformers 5.20.2、PEFT 0.21.2、bitsandbytes 0.50.2、8 GB 显存。保留原始 `requirements.lock.txt`；该锁**没有**后来加入的 PEFT。`requirements-sft.txt` 明确补充这个版本，不代表已经在干净环境验证所有安装步骤。CUDA 构建来源、驱动和操作系统应另行适配。
- [Spider 1.0 官方项目](https://yale-lily.github.io/spider)：来源、校验信息见 [data_source.yaml](configs/data_source.yaml)；下载脚本 `src/data/download_spider.py` 区分 `--stage download` 和 `--stage extract`。原始数据库与处理后的样本不再分发。
- [Qwen 模型](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)：固定 revision `2e1fd397ee46e1388853d2af2c993145b0f1098a`，配置见 [model.yaml](configs/model.yaml)；`src/download_model.py` 区分 `--stage metadata`、`--stage weights`。不分发基座或 adapter 权重。
- 外部比较器使用 [taoyds/test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval)，固定提交 `e97acc546ecbee8fa27fa8dbf025ef61493a876c`，上游许可证 Apache-2.0。源码未随本仓库重复分发，可在新的开发副本按以下方式获取，以便在预期目录找到它：

  ```sh
  git clone https://github.com/taoyds/test-suite-sql-eval.git third_party/test-suite-sql-eval
  git -C third_party/test-suite-sql-eval checkout --detach e97acc546ecbee8fa27fa8dbf025ef61493a876c
  ```

训练入口依赖 Windows 进程树、遥测、本机解释器与历史材料的哈希，以及预检/检查点账本；这些账本、模型、划分产物与运行日志不在发布范围。发布副本把一处用户目录改为读取 `LOCALAPPDATA`；历史冻结哈希仍为原实验版本，不应改写它们来伪造验收。跨机器重建时需另立配置、重新生成材料和执行独立预检；本仓库不宣称 Linux 或全新环境直接运行成功。

## 已记录的实验结果（不是此次发布新运行的测试）

按数据库划分，训练样本 6364 条、开发验证集 625 题。开发集首轮 SFT 的 EX 为 73.12%，多表样本加权的第二轮为 72.96%，没有带来总体提升，因此在最终对照前选定第一轮。对项目留出的 Spider 官方 dev（1034 题、20 个数据库），基座和首轮使用相同的单次贪心配置；执行结果匹配率从 57.64% 到 72.44%（+14.80 个百分点）。该 EX 在各题对应的单份原 SQLite 数据库上比较，而不是官方隐藏测试或完整多数据库 test-suite 成绩。最终执行器对异常 UTF-8 的 SQLite TEXT 使用可逆解码，两模型已有预测统一重评，未重生成或择优。
