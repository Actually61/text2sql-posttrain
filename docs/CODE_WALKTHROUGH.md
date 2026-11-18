# 代码精读：从一条 Spider 样本到 SQL 执行评分

这不是重新介绍项目亮点，而是一份**打开源码后能一步步对照阅读**的导读。每一步说明：先打开哪个文件、输入从哪里来、关键语句做什么、输出交给谁。

本文对应源码版本 [`112b5ce`](https://github.com/Actually61/text2sql-posttrain/tree/112b5ce087947820181be80573eeaea0530cb4e8)。源码链接固定到这个提交，避免后续改 README 导致行号漂移；本次只补文档，没有修改训练和评分代码，也没有重新运行实验。

## 0. 怎么使用这份导读

### 先读主线

| 阅读顺序 | 你要弄明白什么 | 跳转 |
| --- | --- | --- |
| 1 | 原始数据是什么，怎么划分、清洗 | [数据与样本](#data) |
| 2 | schema、messages、token 和 labels 怎么接起来 | [数据与样本的第2–5节](#data) |
| 3 | 哪些权重冻结，哪些参数交给 AdamW | [首轮训练的第1–2节](#training) |
| 4 | 一次前向、反向、更新分别在哪里 | [首轮训练的第3–5节](#training) |
| 5 | 新进程如何接着上一次训练 | [首轮训练的第6节](#training) |
| 6 | 模型怎样生成 SQL、如何判断正确 | [推理与评分](#inference) |
| 7 | 第二轮到底改了什么、为什么最终选首轮 | [第二轮补充](#round2) |
| 8 | 代码阅读顺序 | [代码阅读路线](#interview) |

**实际操作：**左边打开本导读，右边打开对应源码链接；每次只看文中指定的几十行，不要从一个几百行的文件第一行硬读到最后。在 VS Code 中可用 `Ctrl+P` 找文件、`Ctrl+G` 跳到行号，或直接点击本文的 GitHub 行链接。

**这是阅读教程，不是启动教程。** 不需要安装依赖、下载模型、运行 `--run`，也不要为了看懂代码绕过哈希和账本检查。公开仓库没有分发完整数据、模型、adapter和运行账本；文中的产物路径用于说明数据流，不代表这些文件都能在 GitHub 点开。

### 先分清框架和项目代码各自负责什么

| 部分 | 已有框架能力 | 本项目如何使用 |
| --- | --- | --- |
| Transformers | tokenizer、聊天模板、模型加载、`generate()` | 固定模型/分词器版本，构造输入，控制单次生成 |
| PEFT | LoRA挂载、adapter保存/加载 | 指定目标层、rank和alpha，只把LoRA参数交给优化器 |
| bitsandbytes | 4-bit量化加载 | 用NF4基座降低权重显存占用 |
| PyTorch | 张量、自动求导、交叉熵、AdamW | 组织assistant-only监督、token归一化累积、参数更新 |
| SQLite与上游SQL比较器 | SQL执行、返回结果比较 | 只读/超时保护、逐题评分和公平对照 |

因此不是从零写 Transformer，也不是只调用一次微调接口：重点在**数据、目标函数、训练控制、保存恢复和评测如何接成一个可核查流程**。

### 文中常见符号和Python写法

- `B`：一次前向的样本数，正式训练为1；不是梯度累积组里的8。
- `L`：当前完整序列长度；`K`：保留的答案预测位置数；`V`：模型词表大小。
- `dict` 是按键取值的字典，例如 `row['question']`；`list` 是有顺序的列表。
- `[f(row) for row in rows]`：对每条记录调用一次 `f`，收集结果；不会自动并行。
- `{key: value.to(device) for ...}`：新建字典，把其中张量搬到指定设备。
- `**inputs`：把字典展开成函数的命名参数；`[:, 1:]`：所有行、从第1列开始的切片。
- `Tensor.shape` 描述张量维度，`[1,301]` 是1条样本、301个位置；标量loss的shape为 `[]`。

<a id="data"></a>
## 一、数据与样本：先理解送进模型的到底是什么

### 贯穿样例：先知道一条样本是什么

本机清洗训练集第一条是真实的 Spider 训练样本（不是模型预测）：

```json
{
  "source_index": 0,
  "db_id": "department_management",
  "project_split": "train",
  "question": "How many heads of the departments are older than 56 ?",
  "query": "SELECT count(*) FROM head WHERE age  >  56"
}
```

这里只摘录要讲的字段。`question` 是自然语言问题；`query` 是人工参考 SQL，训练时作为答案；`db_id` 指定数据库；`source_index` 是它在原始 `train_spider.json` 中的下标，用来跨文件追踪同一道题，**不是训练中的 step 编号**。

把整体流程先看成四种数据形态：

```text
原始题目 row：Python dict，含 question/query/db_id
    ↓ schema 序列化 + build_messages
对话消息：Python list[dict]，含 system/user；训练另加 assistant
    ↓ tokenizer + labels 掩码
单条特征：Python dict，内部是未 padding 的 list[int]
    ↓ collate_sft_batch
训练 batch：三个 torch.long Tensor，shape 都是 [B, L]
```

`B` 是一次前向的样本数，正式训练为 1；`L` 是该 batch 最长完整序列的 token 数，不是固定 4096。

### 1. 先划分数据库，再清洗训练集

#### 打开哪里

- [check_spider.py:302–319，build_artifacts](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/check_spider.py#L302-L319)：划分。
- [check_gold_sql.py:184–226，audit_one / derive_training](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/check_gold_sql.py#L184-L226)：参考 SQL 检查和训练样本取舍。
- [publish_gold_split.py:40–79](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/publish_gold_split.py#L40-L79)：核对后发布清洗产物。

#### 顺着代码看

`data["train"]` 是已读入的官方训练 JSON 数组，即 `list[dict]`。`db_ids = sorted({...})` 先提取不重复数据库名，再固定种子打乱；`validation_ids` 是选中的验证数据库名集合。第 312–313 行按题目的 `db_id` 分流，不是随机抽题。

实际固定划分为：126 个数据库的 6375 条训练候选，14 个数据库的 625 条内部验证题；官方 dev 的 20 个数据库、1034 题另作项目最终留出集。第 318–319 行若发现数据库或精确样本交集就拒绝输出。

清洗阶段，`checks` 是参考 SQL 的逐题执行记录；`lookup[source_index]` 能找到这道题的执行状态。`derive_training()` 的关键是：

```python
key = (row["db_id"], row["question"], row["query"])
```

这是**三个原始字段完全相同**才判重。代码按原始 `source_index` 排序，保留第一条代表样本。不要把前面统计中的问句归一化，说成这里实际用于删题的规则。

执行记录必须是 `status == "ok"` 且 `row_count_complete == True` 才进入 `keep`。未解决的参考 SQL 执行问题进入 `quarantine`，不是自动改 SQL。`empty_result` 没有作为删除条件，所以查询成功但结果为空不等于坏样本。

已保存记录为 `6375 − 8 条重复 − 3 条隔离 = 6364`；验证集保留全部 625 题，不通过删难题改变分母。`keep.append(row)` 保留原始记录字段，不悄悄改 SQL 文本。发布后存于 `data/processed/spider_gold_v1/train_clean.jsonl`，JSONL 是一行一个 JSON 对象；这是外部运行产物，公开仓库没有随附整份数据。

“这里先按数据库划分，再检查参考 SQL，并按数据库、问题和 SQL 的精确三元组去重。保证同一数据库不会同时进入训练和验证，异常样本有记录，不是靠删验证错题提高成绩。”

**自查：按数据库隔离能保证模型从没见过 Spider 吗？** 不能，它保证本项目微调数据的库隔离，不能排除开源基座预训练接触过公开数据。

### 2. schema 怎么变成模型读得懂的文字

打开 [schema_serialization.py:89–122，serialize_schema](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/schema_serialization.py#L89-L122)。

输入 `schema` 是从 `tables.json` 取出的一份数据库元数据 `dict`；输出是字符串 `str`。核心变量：

| 变量 | 来源、类型 | 在这里做什么 |
| --- | --- | --- |
| `tables` | `schema["table_names_original"]`，`list[str]` | 保留数据库真实表名 |
| `columns` | `schema["column_names_original"]`，元素为 `[表编号, 列名]` | 知道每一列属于哪张表 |
| `table_id` | `enumerate(tables)` 的整数下标 | 当前正在渲染哪张表 |
| `column_index` | 原始 columns 数组下标 | 对应列类型和主外键索引 |
| `lines` | 函数新建的 `list[str]` | 逐行收集说明文字，最后换行拼接 |

第 98–103 行遍历每张表及其列；105–109 行附主键；110–118 行把外键索引转换为 `源列 -> 目标表.目标列`。主外键编号仍指向原始 `columns`，不能去掉第 0 项通配符后重新编号。

贯穿样例的实际 schema 包括 `department`、`head`、`management` 三张表。下面只是其中一段，**真实输入没有因为参考答案只用 head 而删掉另外两张表**：

```text
Table "head":
  Columns:
    "head_ID" [number]
    "name" [text]
    "born_state" [text]
    "age" [number]
  Primary key columns: "head_ID"
  Foreign keys: (none recorded)
```

注意这是给模型看的结构说明，不是可直接执行的建表 SQL；`number/text` 是 Spider 元数据标签，也不冒充精确的 SQLite DDL 类型。

“我把表、列、类型和主外键按固定格式转换成文字，保留原始名字和索引关系。输入包含该数据库完整 schema，不用参考 SQL 偷选相关表。”

### 3. 提示词、参考答案为什么分开保存

打开 [prepare_model_inputs.py:126–144](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/prepare_model_inputs.py#L126-L144)，再看 [234–253 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/prepare_model_inputs.py#L234-L253)。

```python
return [{"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user_template"].format(
            schema=schema_text, question=question)}]
```

这是 `build_messages` 的原逻辑，仅换行排版。`prompt` 是从 `configs/prompt.yaml` 读入的字典；`schema_text` 是上一步的字符串；`question` 来自当前题。函数返回 `list[dict[str, str]]`，含 system 和 user 两条消息，**参数中根本没有 gold SQL**。

第237–246行分别构造输入、参考答案和训练消息列表，249–253行再组织各JSONL文件的内容：

| 产物 | 关键字段 | 谁使用 |
| --- | --- | --- |
| `train_inputs.jsonl` / `validation_inputs.jsonl` | `messages` | 模型生成；不含答案 |
| `train_references.jsonl` / `validation_references.jsonl` | `gold_sql` | 评分、分析；不塞进生成提示 |
| `train_sft.jsonl` | `prompt` + `completion` | 训练；`completion` 是 assistant 参考 SQL |
| `*_lengths.jsonl` | prompt/full/SQL 等 token 长度 | 检查长度预算，不是模型答案 |

`token_lengths` 会套同一聊天模板分别计算“生成前缀”和“完整训练对话”的长度，检查完整对话确实以生成前缀开头。它在这一阶段**只统计长度，不会自动生成 labels**。

“冻结提示词和 tokenizer”的落点在 [load_sources:103–122](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/prepare_model_inputs.py#L103-L122)：这一段核对模型revision和分词器来源，并记录提示词等文件的哈希。后续 [prepare_sft_data.py:128–138](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/prepare_sft_data.py#L128-L138) 才与已冻结的prompt、tokenizer文件和依赖版本逐项比对。这里的冻结是实验版本固定，不是说tokenizer需要参加反向传播后再冻结。

“训练需要问题和参考答案，但推理只能输入问题和 schema。所以我把生成输入和评分参考分文件保存，保证评分答案不会被拼进模型输入。”

### 4. 一条消息怎么变成 input_ids、attention_mask、labels

先打开 [prepare_sft_data.py:257–288](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/prepare_sft_data.py#L257-L288) 看调用方，再进入 [sft_features.py:80–136，encode_sft_example](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/sft_features.py#L80-L136)。

前者遍历训练消息，调用后者完成一条编码，再把特征存成 `train_tokens.jsonl`。此处只有 tokenizer，没有加载大模型、没有 backward。正式训练稍后读取这个冻结特征文件，而不是每次更新都重新清洗原数据。

#### 4.1 先用同一个聊天模板编码两遍

第 100–103 行：

```python
prompt_ids = tokenizer.apply_chat_template(
    prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
input_ids = tokenizer.apply_chat_template(
    prompt + [completion], tokenize=True,
    add_generation_prompt=False, return_dict=False)
```

上述摘录省略外层 `_token_list` 校验，模板参数不变。`prompt` 为两条 system/user 消息，`completion` 为 assistant 的一条消息；它们来自 `train_sft.jsonl`，不是临时编的答案。 调用方把每条 `row` 作为参数 `example` 传入；原记录的 `completion` 字段是只含一条消息的列表，函数第95行从 `example["completion"][0]` 复制并校验出字典 `completion`，再用 `[completion]` 与 `prompt` 列表拼接。

- `tokenize=True`：输出 token 编号列表，而不是模板字符串。
- `add_generation_prompt=True`：在问题后加 assistant 的开头，让模型从回答位置继续写。
- 完整训练对话已经有 assistant 答案，因此第二次不再额外加回答开头。
- `prompt_ids`、`input_ids` 都是 `list[int]`。前者只有前缀，后者有前缀、SQL、EOS及模板可能附带的换行。
- 第 104–105 行检查两者前缀完全一致，防止把答案起点算错。

#### 4.2 用边界决定监督哪些位置

第 109–129 行找出答案起点 `prompt_length`、结束标记下标 `eos_position`，再构造标签：

```python
labels = [IGNORE_INDEX] * len(input_ids)
labels[prompt_length:eos_position + 1] = input_ids[prompt_length:eos_position + 1]
```

`IGNORE_INDEX` 在第 11 行定义为 `-100`。第一行先让所有位置不计损失；第二行只把 SQL 和 EOS 的原始 token 编号填回。

| 字段 | 类型和未 padding 长度 | 含义 |
| --- | --- | --- |
| `input_ids` | `list[int]`，长度 `L` | 模型看到的完整训练对话 |
| `attention_mask` | `list[int]`，长度 `L` | 真实输入位置为 1；本阶段还没 padding，所以全为 1 |
| `labels` | `list[int]`，长度 `L` | SQL、EOS处是目标 token；其余为 -100 |
| `prompt_length` | `int` | 前缀长度，也是第一个 SQL token 的下标 |
| `eos_position` | `int` | EOS 在完整序列中的下标 |
| `supervised_token_count` | `int` | SQL token 数＋1个 EOS |

**不计损失不等于看不见。** prompt 的 label 为 -100，但 attention_mask 为 1：模型仍然利用 schema 和问题生成答案。不要把 prompt 的 attention_mask 改成 0。

#### 4.3 用真实样本对照下标

已有 `source_index=0` 特征记录：完整序列 301 token；前缀 286；SQL 占位置 286–298；EOS 在 299；300 是模板换行。按 Python 从 0 计数：

| 位置 | 内容 | labels | attention_mask |
| --- | --- | --- | --- |
| 0–285 | system/user及assistant开头 | -100 | 1 |
| 286–298 | 参考 SQL 的13个 token | 与 input_ids 相同 | 1 |
| 299 | EOS，编号151645 | 151645 | 1 |
| 300 | 模板换行，编号198 | -100 | 1 |

因此有效监督数为 14，不是301。**这些是当前已保存训练样本的真实长度，不是“一个英文单词等于一个 token”的估算。**

这里 `labels` 尚未移动。因果语言模型用前一位置的输出预测下一个 token，所以首个 SQL 标签在286，但对应预测来自位置285。实际自定义 loss 在 `target_loss.py` 里做一次对齐，不能在数据准备时再提前移一次。

“input_ids保留完整上下文，labels只保留SQL和EOS，其余用-100。这样模型看着问题和表结构学习答案，不把复述prompt当训练目标。”

### 5. collate 是什么：把列表变成可训练 batch

打开 [sft_features.py:139–158，collate_sft_batch](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/sft_features.py#L139-L158)。

`features` 是若干条刚才的特征字典；函数先找到当前 batch 最长长度 `longest`，给其他样本右侧补齐，然后生成 CPU 张量。

| 新增 padding 位置 | 填什么 | 原因 |
| --- | --- | --- |
| `input_ids` | `pad_token_id` | 让张量形状一致 |
| `attention_mask` | 0 | 标记这是补齐，不是真实上下文 |
| `labels` | -100 | 不训练模型去预测补齐内容 |

三个输出都是 `torch.long`、shape `[B, longest]`。正式训练 `B=1`，通常不需要额外 padding；如果一条完整序列301 token，张量形状就是 `[1,301]`，**不会每条都补成 `[1,4096]`**。移动到GPU由训练代码完成，此函数返回的是CPU张量。

长度超过上限时，编码函数第106–107行直接报错，不悄悄截断。已保存统计为6364条、最长3761token、82条超过2048、0条超过4096，所以正式数据没有为迁就2048长度而丢弃长样本。

**自查：EOS和padding能一起全部mask吗？** 不能。EOS是要学习的“答案结束”目标；padding才需要忽略。本代码按位置划分监督，不凭token编号一刀切。

<a id="training"></a>
## 二、首轮训练：从三字段batch到LoRA更新

### 先认清：入口、训练循环、算法函数是三层

```text
sft_round1_resource_v3.py                 控制器：预检、资源门槛、分段启动和验收
  └─ sft_round1_resource_v3_worker.py     worker：真正循环读取训练组、反向、更新、保存
       ├─ qlora_stability.fresh_model    加载量化基座并挂新 LoRA
       ├─ sft_features.collate_sft_batch 把一条编码样本变成 batch 张量
       ├─ cache_release_core.backward_cache_group
       │    ├─ target_loss.target_position_loss  前向和答案位置交叉熵
       │    └─ loss.backward()                  累积梯度，不更新权重
       ├─ JournaledAdamW.step()                 一组结束，更新一次 LoRA
       └─ sft_checkpoint.save_checkpoint/load_checkpoint
```

**第一次读时，从 worker 的第 196 行开始，不必先读上百行进程监督。** 前面的身份握手、超时、遥测是安全运行设施，不是微调算法。正式 worker 复用了早期脚本中的函数，因此出现 `qlora_stability`、`smoke` 这些名称，**不代表正式训练只有32条或1步**。

另两个容易看错的文件：`sft_train.py` 中的 `train_segment` 是早期试跑循环；正式 worker 只复用它的学习率和 Adam 步数辅助函数。`token_accumulation.py` 提供原始累积实现和计数函数；正式 worker 实际调用的是带缓存释放观测的 `cache_release_core.py`。

### 1. 先找训练数据、顺序和参数来自哪里

打开 [worker 第 113–131 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L113-L131)：

```python
start, end = config['segment_ranges'][segment - 1]
groups = preflight['groups']
with (ROOT / config['train_path']).open(encoding='utf-8') as stream:
    original = [json.loads(line) for line in stream]
by_id = {row['source_index']: row for row in original}
```

这段不是在生成训练数据，而是**消费已冻结的训练数据与分组计划**。

| 变量 | 来源、类型 | 在这里做什么 |
| --- | --- | --- |
| `config` | `configs/sft_round1_resource_v3.json` 读出的 `dict` | 提供正式运行目录、数据路径、4个分段区间和资源上限 |
| `preflight` | CPU预检报告读出的 `dict` | 保存已经核对的正式训练配置、分组和受保护文件哈希 |
| `original` | `train_tokens.jsonl` 每行解析后得到的 `list[dict]` | 6364条已经包含 token 和标签的训练记录，不是原始 Spider 问题文本列表 |
| `by_id` | `{source_index: row}` 字典 | 根据原始样本编号快速找回一整条记录 |
| `groups` | 冻结调度中的 `list[dict]` | 每组包含 `source_indices`、各样本长度和整组答案token数 |
| `start, end` | 当前分段的更新区间，两个 `int` | 第1段为 `[0, 200)`；第2段为 `[200, 400)`，不是题号范围 |

数据与保存位置见 [resource_v3 配置第62–73行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/configs/sft_round1_resource_v3.json#L62-L73)。参数有两层：运行约束在 [resource_v3 配置第8–35行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/configs/sft_round1_resource_v3.json#L8-L35)，算法配方在 [sft_round1.json 第 33–50 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/configs/sft_round1.json#L33-L50)：batch1、累积8、1轮、796次更新、AdamW 峰值学习率 `1e-4`、`weight_decay=0`、24步 warmup、梯度范数上限1。

`sft_round1.json` 保留了最初规划时的 `cpu_preflight_only` 等历史字段，不能把它误读成项目到现在没有训练。真正的后续启动/结果由正式控制器和独立运行记录负责。

**为什么是796次？** `6364 = 795 × 8 + 4`，最后4条也组成一组更新，既不丢弃，也不补成重复样本。分组生成逻辑在 [sft_round1_plan.py 第 65–96 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_plan.py#L65-L96)：按完整长度进入512/1024/2048/4096四个桶，桶内打乱、组成完整组，余数合并后再分组，最后追加不足8条的尾组。batch1本身没有padding，分桶不是“把每条补到桶上限”，也不是截断。

“训练样本和更新顺序在CPU阶段就冻结了。worker按source_index取完整样本，每条只用一次，最后不足8条的组也正常参与更新。”

### 2. 基座、LoRA、AdamW 是如何接起来的

打开 [worker 第 196–226 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L196-L226)，先看这三句：

```python
model = fresh_model(read(ROOT / formal['model']['inherit_loading_and_lora_from']))
named = adapter_parameters(model)
optimizer = JournaledAdamW(list(named.values()), ledger_path=ledger_path,
    start=start, segment_end=end, total_budget=796,
    **{key: value for key, value in training['optimizer'].items() if key != 'name'})
```

- `formal = preflight['formal_config']`；`training = formal['training']`，不是隐藏的全局超参数。
- `fresh_model(...)` 的参数是 `configs/qlora_smoke.json` 读出的字典，**这里只继承加载和LoRA配置，不继承它的一步冒烟训练限制或冒烟adapter**。
- `model` 是挂载 LoRA 的模型对象。`named` 是 `{参数名: torch.nn.Parameter}` 字典，只含 `requires_grad=True` 的 A/B。`list(named.values())` 把这些参数交给优化器，不含冻结基座。
- `**{...}` 是把字典展开成 `lr=...、betas=...、eps=...、weight_decay=...` 等关键字参数；过滤 `name`，因为它是配置描述而非 AdamW 的构造参数。

进一步打开 [qlora_stability.py 第 115–151 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/qlora_stability.py#L115-L151)：

1. **119–123行：加载。** `BitsAndBytesConfig` 指定NF4 4-bit权重、双重量化和BF16计算；Transformers `from_pretrained` 从本地读基座，并放到 `cuda:0`。
2. **124–126行：训练准备。** 关闭推理用KV cache；PEFT的 `prepare_model_for_kbit_training` 冻结基座并准备量化训练，启用梯度检查点，以重算激活换显存。
3. **144–150行：挂LoRA和检查。** `get_peft_model` 在配置指定的线性层加入A/B，核对可训练参数数为9,232,384、B初始为0。配置目标为注意力的Q/K/V/O和MLP的gate/up/down；`r=8, alpha=16`。

127–143行是项目针对本机冻结审计的额外准备：将量化线性层的bias转换为计算dtype，同时要求数值可无损往返并核对量化结构，**不是训练bias，更不是另一种LoRA算法**。

优化器类在 [sft_update_budget.py 第 22–51 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_update_budget.py#L22-L51)：它继承 `torch.optim.AdamW`。项目额外记录“尝试更新数/完成更新数”，真正的 AdamW 更新仍是第48行 `super().step(closure)`。

“Transformers负责加载基座，bitsandbytes负责量化，PEFT负责挂LoRA。我的训练入口把A/B参数交给PyTorch AdamW，并额外检查更新次数和参数确实变化，不是从零实现AdamW。”

### 3. 一组8条，怎样变成8次前向/反向和1次参数更新

打开 [worker 第 239–253 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L239-L253)：

```python
model.train()
for group_index in range(start, end):
    update = group_index + 1
    group = groups[group_index]
    rows = [by_id[index] for index in group['source_indices']]
    batches = [collate_sft_batch([row], 151643, max_length=4096) for row in rows]
    optimizer.zero_grad(set_to_none=True)
    lr = set_next_lr(optimizer, group_index, training)
```

这是保留关键行的连续流程摘录，中间的审计语句未列出。

- `group_index` 是从0开始的组索引，`update` 是从1开始的更新次数。
- `rows` 是本组8条记录（尾组4条）。`[row]` 外面这一层方括号很重要：每次collate只输入**1条**，不是一次把8条搬到GPU。
- `batches` 是CPU上的列表，每项是字典：`input_ids/attention_mask/labels` 都是 `torch.long` 张量，形状 `[1, L_i]`。不同记录的完整长度 `L_i` 可以不同。
- `151643` 是本项目使用的pad token ID；不是每个模型通用的数字。batch1无需额外补齐，但collate仍做完整性检查。
- `zero_grad(set_to_none=True)` 清的是上一组 `.grad`，不是权重，也不是Adam的历史动量。
- `set_next_lr` 按当前已经完成的组数设置下一次更新的学习率：1–24次从小到大warmup，之后线性衰减。恢复后按保存的游标继续，不重新warmup；见 [sft_train.py 第 19–25 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_train.py#L19-L25) 与 [sft_round1_plan.py 第 54–62 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_plan.py#L54-L62)。

真正计算一整组梯度的是 [worker 第 307–312 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L307-L312) 调用的 `backward_cache_group`。跳转到 [cache_release_core.py 第 25–52 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/cache_release_core.py#L25-L52)：

```python
counts = [supervised_token_count(batch) for batch in batches]
total_targets = sum(counts)
target_device = torch.device(device)
for index, (batch, count) in enumerate(zip(batches, counts)):
    current = {key: value.to(target_device) for key, value in batch.items()}
    loss, metadata = loss_fn(model, current)
    weight = count / total_targets
    (loss * weight).backward()
```

这是按执行顺序抽出的关键行，不是独立可运行脚本。 `device` 来自worker传入的 `"cuda:0"`；`loss_fn` 是worker第307–309行定义并传入的BF16 autocast前向包装，内部调用下一节的 `target_position_loss`。`zip(batches, counts)` 把每条batch与它的目标数配成一对，`enumerate` 再提供从0开始的序号。

| 变量 | 类型/来源 | 意义 |
| --- | --- | --- |
| `counts` | `list[int]`，数每条labels中的有效下一token目标 | 各条SQL答案加结束标记的目标数，不是输入总长度 |
| `total_targets` | `int`，`sum(counts)` | 当前实际累积组的总目标数，包括最后4条组 |
| `current` | `dict[str, Tensor]`，当前CPU batch搬到GPU | 此时仍是三个 `[1, L_i]` 张量 |
| `loss` | 带计算图的0维Tensor | 本条有效答案token的平均交叉熵 |
| `weight` | `float`，本条目标数/本组目标数 | 把本条均值换算为整组token均值的一部分 |
| `metadata` | 普通Python字典 | 前向中实际形状、有效目标数等，供检查和日志使用 |

`backward()` 把梯度加到 A/B 的 `.grad`，此时**权重还没有改变**。组内没有 `zero_grad()`，所以8条梯度会累计；每条平均loss按token数加权后，不再额外除8。否则会把梯度又缩小一次。

`loss.detach().item()` 仅用于记录一个Python浮点数，不参与求导。真正的 `.backward()` 必须用原始、仍连接计算图的 `loss`，不能用日志数值。

“我一次只把一条完整样本放进显存。每条loss按它占整组有效答案token的比例反向，8条梯度累计完成后才更新一次，这样与整组token平均的目标一致。”

### 4. 前向loss到底怎么计算：先对齐，再只算需要的位置

打开 [target_loss.py 第 33–55 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/target_loss.py#L33-L55)：

```python
shift_labels = F.pad(labels, (0, 1), value=-100)[:, 1:]
valid = shift_labels != -100
positions = valid.any(dim=0).nonzero(as_tuple=True)[0]
targets = shift_labels.index_select(1, positions)
output = model(input_ids=input_ids, attention_mask=attention_mask,
               use_cache=False, logits_to_keep=positions)
logits = output.logits
loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), targets.reshape(-1),
                       ignore_index=-100, reduction="mean")
```

这是关键语句摘录，原文件还检查了shape、目标范围和空目标等异常。

**为什么移位？** 因果语言模型在位置`t`预测的是下一个token，因此预测位置`t`应对齐 `labels[t+1]`。第一句先在右侧补一个`-100`，再去掉第一列，得到仍为 `[B,L]` 的 `shift_labels`。本项目准备数据时没有预移位，这里只移动一次。

例如答案的第一个token位于输入位置`p`，用于预测它的是位置`p-1`的输出。**不能直接取答案token自身所在位置的logits，与它自己的token比较。**

| 名称 | 形状（本项目通常B=1） | 内容 |
| --- | --- | --- |
| `input_ids`、`labels`、`attention_mask` | `[B, L]` | 完整上下文；labels只有答案及EOS有效 |
| `shift_labels` | `[B, L]` | 每个预测位置对应的下一个正确token |
| `valid` | `[B, L]`，布尔 | 哪些预测位置需要计算损失 |
| `positions` | `[K]`，整数索引 | 需要输出logits的序列位置；batch>1时是各行位置的并集 |
| `targets` | `[B, K]` | 这些位置应预测的目标token；并集下个别格仍可能是-100 |
| `logits` | `[B, K, V]` | 每个保留位置对词表V个token的分数，不是已经选出的token |
| `loss` | `[]`，0维Tensor | 有效目标的平均交叉熵 |

对前面那条真实训练样本，`L=301`，有效目标在原标签286–299，因此用于预测它们的 `positions` 是285–298，共14个位置。batch1时，`targets` 为 `[1,14]`，`logits` 为 `[1,14,V]`，而不是 `[1,301,V]`。

`logits_to_keep=positions` 只减少输出词表投影 `lm_head` 的计算位置；**Transformer仍接收并处理完整prompt和SQL上下文**。因此这是节省输出层中间张量，不是把问题或schema删掉。由于本函数自己对齐目标并计算交叉熵，调用 `model(...)` 时不传 `labels`，避免模型内部再移位一次。

**为什么训练输入里有参考SQL，不算直接偷看答案？** 训练采用teacher forcing：预测某个答案token时，模型能看问题、schema和它前面的正确token。模型内部的因果注意力限制它不能看未来token；外部 `attention_mask` 的0/1主要标记padding，不等于因果遮挡本身。推理时没有参考SQL，只能把自己刚生成的token接回去继续生成。

“输入保持完整，训练目标是用前面的上下文预测下一个SQL token。我显式对齐一次标签，只对有效答案的预测位置计算词表logits和交叉熵，减少显存，但不改变监督目标。”

### 5. 哪里真正更新参数，怎么确认更新成功

回到 [worker 第 311–344 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L311-L344)，按这个顺序读：

1. **311–315行：**完成整组反向并核对目标数、缓存释放次数；组不完整就停止，不用半组梯度继续更新。
2. **317–322行：**检查梯度有限、全局范数非零，再通过 `clip_grad_norm_(..., 1.0)` 将整组梯度范数限制在1以内。梯度裁剪不是把参数截断。
3. **326行：**`optimizer.step()`，此时AdamW才按照累计梯度更新A/B。`weight_decay=0`，所以这里没有权重衰减项。
4. **328–340行：**检查Adam状态步数是否与更新次数一致；把更新前后adapter对比，要求参数有限且至少有元素发生变化。
5. **341–344行：**准备下一步学习率并清空梯度，准备下一组或保存检查点。

LoRA的B从0开始，所以第一步A的梯度可能为0；代码要求的是**整体有有限且非零的梯度、至少有参数变化**，不是宣称每个矩阵每一步都变化。

每条反向后的 `empty_cache()` 在 [worker 第 281–305 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L281-L305)。项目检查了调用前后累计梯度和Adam状态的存储/版本没有改变。它只释放未使用的CUDA缓存，不清梯度、不清动量，也不能释放正在使用的模型张量；不要把它说成通用训练加速技巧。

“backward只积累梯度，step才改参数。我不仅记录loss，还检查梯度、Adam步数和adapter更新前后的差异，防止出现看起来在训练、实际上参数没动的情况。”

### 6. 为什么保存adapter还不够：正式训练怎样分段恢复

打开 [控制器第 319–333 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3.py#L319-L333)。它依次启动4个新的worker进程，对应更新区间0–200、200–400、400–600、600–796。每段成功保存并通过检查后，才进入下一段；失败不会自动重跑。

worker每段都先构建相同基座和新的LoRA结构；第1段使用新初始化，第2–4段通过 [worker 第 224–231 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L224-L231) 载入**本轮上一段**的检查点，覆盖A/B及优化器状态，继续原训练，不是每段重新训练一轮。

保存入口是 [worker 第 357–367 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L357-L367)。此时整组更新完成，下一步学习率已设置，梯度已清空；随后还会核对冻结基座哈希不变。

打开 [sft_checkpoint.py 第 188–224 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_checkpoint.py#L188-L224)：

| 文件 | 保存什么 | 为什么需要 |
| --- | --- | --- |
| `adapter_model.safetensors`、`adapter_config.json` | A/B权重及LoRA配置 | 恢复微调出的模型变化；不复制整套基座权重 |
| `optimizer.pt` | Adam一阶/二阶矩、step、参数组等 | 只恢复adapter会丢失优化器历史，继续训练的轨迹不同 |
| `rng.pt` | Python、NumPy、Torch CPU/CUDA随机状态 | 恢复后接续随机状态，而不是重新从初始种子开始 |
| `metadata.json` | 完成步数、下一组索引、下一步LR、基座/数据调度/配置/源码哈希绑定 | 不重复读样本、不重启warmup、防止错配版本 |
| `complete.json` | 文件哈希及状态指纹、参数布局 | 区分完整检查点与保存一半的目录 |

实现先写入新的 `.incomplete` 目录，回读核对值及哈希，最后同目录改名发布。只有完整检查点才允许恢复；不会覆盖已有检查点。

恢复流程在 [sft_checkpoint.py 第 230–270 行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_checkpoint.py#L230-L270)：先校验文件和元信息，再恢复adapter与optimizer、验证参数名字/顺序/shape/dtype和状态指纹，**最后**恢复随机状态，避免模型构造等操作消耗本应继续使用的随机序列。

此实现没有单独的 `scheduler.pt`：学习率是根据已完成更新数和冻结公式计算的，所以保存游标、下一步LR和调度配置哈希即可。`rng.pt`使用了pickle形式的状态，代码只用于项目自己生成的可信检查点，哈希一致不等于任意外部文件都安全。

“推理需要原基座和adapter，但不需要优化器状态；继续训练还要恢复Adam状态、学习率进度、数据游标和随机状态。我在完整累积组结束后保存，读回验收，再让下一进程从同一位置继续。”

<a id="inference"></a>
## 三、推理与评分：生成SQL和给SQL判分是两件事

### 先认清这几个入口，避免看错历史文件

| 场景 | 实际文件 | 阅读时需要知道的边界 |
|---|---|---|
| 625 题基座内部验证生成 | `src/infer/baseline_full.py` | 不含微调 adapter；配置是 `configs/baseline_full.json` |
| 最终基座补齐 | `src/infer/final_holdout_resume_v3.py` | 原来已完成的 616 条保留，只补缺少的 418 条，不挑最好答案 |
| 最终首轮 SFT 生成 | `src/infer/final_holdout_resume_v4.py` | **118–120 行强制只接受 `sft`**，虽然顶部继承的说明和部分分支仍提到基座；不要把它理解成可随意重跑两模型的通用入口 |
| 两模型最终统一评分 | `src/eval/evaluate_final_decode_v4.py` | 读取已有预测，CPU 执行 SQL；不加载模型、不重新生成 |
| 一道题真正的 SQLite 执行与比较 | `src/eval/spider_pair_worker_decode_v2.py` | 父评分程序调用它；它接收一题的参考 SQL、预测 SQL 和数据库路径 |

这些是已记录实验的代码阅读入口，**不是下载仓库即可直接运行的命令清单**。它们依赖未发布的数据、模型、adapter、冻结哈希、CPU 预检和运行账本。按下面顺序打开代码，不要启动历史入口。

### 第一步：模型推理时到底拿到了什么？

打开 [`final_holdout_common_v1.py:139–163`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/final_holdout_common_v1.py#L139-L163)，读 `load_prepared()`。

真实关键代码：

```python
encoded = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True,
                                       return_tensors='pt', return_dict=True)
```

- `row`：从准备好的 `inputs.jsonl` 读出的一个 Python 字典。里面有 `source_index`（原题编号）、`db_id`（数据库名）、`messages`（system/user 消息）等；没有参考 SQL。
- `tokenizer`：从冻结模型目录加载的分词器。它不仅切分文本，还按模型规定的聊天格式组织角色和结束标记。
- `row['messages']`：`list[dict]`，这里只允许 `system` 和 `user`，没有训练时的 assistant 标准答案。
- `add_generation_prompt=True`：在对话末尾加上“接下来该 assistant 回答”的起始格式，不是把答案放进去。
- `return_tensors='pt'`：返回 PyTorch 张量，而不是普通整数列表。`return_dict=True` 返回包含多个张量的映射。
- `encoded['input_ids']`：CPU 上的整型张量，形状 `[1, L]`。`1` 是这一批只有一道题，`L` 是这道题提示词的 token 数。
- `encoded['attention_mask']`：形状也是 `[1, L]`，告诉模型哪些位置是真实输入；这里逐题编码、不做 padding，通常全是 1。
- 返回的 `prepared`：`list[tuple]`，每项为 `(row, encoded)`；后面的生成循环逐项取出。

154–157 行检查字段白名单和消息角色，防止把参考答案混入推理；160–161 行只检查长度，超过预设上限立即报错，**不是静默截断**。本函数会核对包括参考文件在内的文件哈希，但不解析参考 SQL 作为模型输入。

“推理只读问题和 schema；聊天模板把它们编码成 `[1, L]` 的输入，标准答案留在评分阶段，不能进入模型的生成输入。”

### 第二步：训练好的 adapter 是怎么用来生成的？

先看 [`final_holdout_resume_v4.py:222–238`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/infer/final_holdout_resume_v4.py#L222-L238)：先加载冻结版本的 4-bit 基座，再执行：

```python
model = PeftModel.from_pretrained(model, ROOT / config['adapter_path'],
    is_trainable=False, local_files_only=True).eval()
```

- 右侧传入的 `model` 是已经加载到 GPU 的原基座；左侧重新赋值后的 `model` 是带有 LoRA adapter 的 PEFT 模型。
- `config['adapter_path']` 来自 `configs/final_holdout_resume_v4.json` 第 8 行，指向首轮 `update_000796` 检查点。
- `is_trainable=False` 表示只用于推理，不为 adapter 开启训练；`.eval()` 进入评估模式。
- 231–235 行检查加载到内存的 adapter 指纹和当前启用的 adapter，236–238 行再断言参数全部冻结、模型不处于训练模式。
- 这里不创建优化器，不调用 `backward()` 或 `optimizer.step()`，所以不是继续微调。

再看 [`final_holdout_resume_v4.py:278–286`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/infer/final_holdout_resume_v4.py#L278-L286)：

```python
inputs = {key: value.to('cuda:0') for key, value in encoded.items()}
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
begun = time.perf_counter()
with torch.inference_mode():
    output = model.generate(**inputs, generation_config=deepcopy(generation_config))
torch.cuda.synchronize()
seconds = time.perf_counter() - begun
new_ids = output[0, encoded['input_ids'].shape[1]:].tolist()
text = tokenizer.decode(new_ids, skip_special_tokens=True)
```

按执行顺序理解：

1. 第一行把 `encoded` 中的 CPU 张量逐个搬到 GPU，得到字典 `inputs`。`**inputs` 是 Python 的字典展开，等价于将 `input_ids=...`、`attention_mask=...` 作为命名参数传给 `generate()`。
2. `generation_config` 来自冻结的基座生成配置。`deepcopy()` 给本次调用独立副本；不是重新采样多个答案。`inference_mode()` 不记录训练用的梯度计算图。
3. CUDA 计算是异步的，前后 `synchronize()` 让记录的 `seconds` 覆盖该次 GPU 生成完成的时间；峰值显存从本题开始重新统计。
4. 对这个 decoder-only 模型，`output` 是形状 `[1, L+G]` 的 token 张量：前 `L` 个是输入，后 `G` 个才是新答案。`output[0, L:]` 取第一题的新增部分；`.tolist()` 变成 Python 整数列表 `new_ids`。
5. `tokenizer.decode()` 把新 token ID 还原成字符串 `text`。`skip_special_tokens=True` 去掉模型控制标记，不是修正 SQL。

生成参数见 [`configs/baseline_full.json:13–24`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/configs/baseline_full.json#L13-L24)：`do_sample=False`、`num_beams=1`、`num_return_sequences=1` 表示每题一次贪心生成；`max_new_tokens=256` 限制**新增答案**长度，不是问题加答案总长度。`temperature/top_p/top_k` 虽保留在配置中，但不用于这里的非采样贪心选择。`use_cache=True` 使用题内 KV Cache；不把不同题的上下文拼起来。

“评测时先加载同一基座，再挂载保存的首轮 adapter，关闭训练与梯度记录。每题贪心生成一次，只截取新生成的 token，解码后保存成 SQL。”

### 第三步：模型输出是怎么保存的，会不会偷偷修答案？

打开 [`baseline_pilot.py:56–59`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/infer/baseline_pilot.py#L56-L59)：最终生成代码复用了这里的 `extract_sql()`，并没有调用这份旧脚本去重跑试验。

```python
def extract_sql(text):
    text = text.strip()
    fenced = re.fullmatch(r"```(?:sql)?[ \t]*\r?\n(.*?)```", text, flags=re.S | re.I)
    return fenced.group(1).strip() if fenced else text
```

- `strip()` 只去掉首尾空白。
- `re.fullmatch()` 要求**整段输出**恰好是一个无语言标记或 `sql` 标记的 Markdown 代码块，才去掉外层围栏；不是从解释文字中搜索一条 SQL。
- 不切分分号来挑选其中一条，不根据执行报错改写 SQL，也不查看标准答案后选择更好的候选。

[`final_holdout_resume_v4.py:289–313`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/infer/final_holdout_resume_v4.py#L289-L313) 把 `source_index`、`db_id`、`predicted_sql`、原始 `generated_text`、含特殊标记的原始输出、`new_token_ids`、生成时长和显存等写成 JSONL。`source_index` 是之后逐题配对的身份，不是按结果排序后的新编号。每题写入后 `flush()`/`fsync()` 落盘；记录 `attempt_number=1`，失败保留已经写好的结果后停止，不自动重试择优。

**问题：SQL 没生成好，评分器会帮它修吗？答案：不会。**生成后处理只去外围包装，模型生成错误要体现在评测里，否则测到的就不只是模型的首次生成能力。

### 第四步：SQL 可执行率和 EX 究竟在哪儿计算？

先看 [`evaluate_final_decode_v4.py:143–186`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_final_decode_v4.py#L143-L186)：`load_generation()` 读取已有预测，并检查模型身份、生成配置、1034 条唯一题号、无重试以及文件哈希。它不调用模型。

接着看 [`evaluate_final_decode_v4.py:214–222`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_final_decode_v4.py#L214-L222)：调度器按 `source_index` 找到对应预测，构造请求 `request`：

- `db_path`：根据该题 `db_id` 拼出的 SQLite 数据库绝对路径，字符串。
- `gold_sql`：从单独的参考文件中读出的标准 SQL，字符串。
- `predicted_sql`：已经生成并保存的模型 SQL，字符串。
- `query_timeout_seconds` / `comparison_timeout_seconds`：执行和比较的预算；不是生成模型的超时。

随后 `client.call(request)` 把请求交给独立 CPU 评分子进程。真正看判分，打开 [`spider_pair_worker_decode_v2.py:187–207`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/spider_pair_worker_decode_v2.py#L187-L207)：

```python
response["order_matters"] = "order by" in gold_sql.lower()
gold, gold_rows = _execute_query(path, gold_sql, query_budget)
predicted, predicted_rows = _execute_query(path, predicted_sql, query_budget)
response["gold"], response["prediction"] = gold, predicted
response["prediction_executable"] = predicted["status"] == "ok"
```

`_execute_query()` 返回两个对象：

| 对象 | 类型 | 内容 |
|---|---|---|
| `gold` / `predicted` | `dict` | 执行状态、行数、列数、耗时、错误类型等元数据 |
| `gold_rows` / `predicted_rows` | `list[tuple]` | SQLite 真正返回的行；例如 `[(2,)]` 表示一行一列、值为 2 |
| `prediction_executable` | `bool` | 预测 SQL 是否执行成功，与答案是否正确分开 |
| `matched` | `bool` 或未评分时的 `None` | 查询结果是否按固定比较器判为一致 |

如果两边都执行成功，199 行才调用：

```python
matched = official_exec_eval.result_eq(gold_rows, predicted_rows, response["order_matters"])
```

第207行将 `bool(matched)` 写回结果，并设 `status="scored"`。**`scored` 只表示比较成功完成，不表示答对；只有 `matched=True` 才计入EX正确数。**

比较的是**查询返回结果**，不是 SQL 字符串。`official_exec_eval` 来自冻结版本的外部 `test-suite-sql-eval/exec_eval.py`；本仓库负责调用，不是自己重新发明 SQL 等价判断算法。此项目只在每题对应的一份原始 SQLite 数据库上执行，并没有跑完整的官方多数据库测试套件。

例如同一张表中，标准 SQL 查询人数得到 `[(2,)]`，预测 SQL 查询姓名得到 `[('Alice',), ('Bob',)]`：预测执行成功，但结果不匹配，EX 计错。这是教学示例，不是声称某条真实实验预测。

最终比率在 [`evaluate_spider.py:261–268`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_spider.py#L261-L268) 汇总：

- `correct = sum(r['matched'] is True for r in results)`，EX = `correct / count`。
- `valid = sum(r['prediction_executable'] is True for r in results)`，可执行率 = `valid / count`。
- 分母是整个固定评测集合，不是只留下能运行的 SQL。预测 SQL 错误仍占分母；数据库、标准答案或评分基础设施错误则导致正式验收停止，不能悄悄把这种错误算成模型失误后宣称评测完成。

“先分别执行参考和预测 SQL，再比较返回结果。能执行只说明语法和表列引用等能通过，不代表答对；EX 才检查查询结果，但单份数据库上的相同结果也不能证明所有数据情况下的语义等价。”

### 第五步：评分安全和最终修复应该怎么解释？

[`spider_pair_worker_decode_v2.py:64–105`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/spider_pair_worker_decode_v2.py#L64-L105) 的核心有三点：

1. `mode=ro` 和 `PRAGMA query_only=ON` 只读打开数据库；授权回调允许查询、读操作和受限制的函数，拒绝其他操作。不是仅靠提示词写“只生成查询”。
2. `set_progress_handler()` 定期检查查询时间；父进程还通过 [`evaluate_spider.py:216–258`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_spider.py#L216-L258) 设硬截止时间，超时终止子进程。不能把 SQLite 的连接 `timeout=` 当成完整查询时间限制，它主要关联锁等待。
3. `text_factory` 只对查询返回的 SQLite TEXT 用 `raw.decode('utf-8', errors='surrogateescape')` 解码；非法字节可逆保留，不使用忽略或替换字符，BLOB 仍是 `bytes`。这修复的是结果读取问题，不改 SQL，也不改 `result_eq`。

最终两模型已有的全部预测使用同一新版评分；没有重新让模型回答。完整政策与限制写在 [`evaluate_final_decode_v4.py:57–66`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_final_decode_v4.py#L57-L66)。注意这是离线数据库保护，不能据此声称已经实现线上按用户划分的表、列、行权限。

### 第六步：从两份分数得到“改善/退化”而不只报总分

看 [`evaluate_final_decode_v4.py:275–344`](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/evaluate_final_decode_v4.py#L275-L344) 的 `compare_saved()`。这个函数只读已有分数，连 SQLite 都不会重新执行：

1. `scores[arm]` 将每个模型的逐题记录变成 `dict[int, dict]`，用 `source_index` 作键；不会按生成速度或答案内容对齐。
2. 295–301 行比较两边提示词、生成配置字节和 EX 协议，303–311 行核对题号、数据库、难度、问题和 gold SQL 一致，再组成配对列表 `pairs`。
3. 317–320 行：基座错、SFT 对计 `improved`；基座对、SFT 错计 `regressed`。强制检查 `改善数 - 退化数 = SFT 对题数 - 基座对题数`，避免配错样本后只看百分比。
4. 输出总体、各难度、失败类型及改善/退化题号到 `comparison.json`。它是分析产物，不是给模型下一轮自动调参的输入。

“我不是只比较两个平均准确率，而是先确认同一题、同一协议，再逐题统计改善和退化。这样既能看到收益，也能看到微调损害了哪些原本会做的题。”

<a id="round2"></a>
## 四、第二轮只补看这一处：样本权重如何进入loss

第一轮主线读懂后，再打开 [sft_round2_weight_core_v1.py:41–61](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round2_weight_core_v1.py#L41-L61) 和 [75–99行](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round2_weight_core_v1.py#L75-L99)。

首轮：`本条loss系数 = 本条有效答案token数 / 整组有效答案token数`。

第二轮：`本条loss系数 = 本条样本权重 × 本条有效答案token数 / 整组加权有效答案token数`。

代码中的 `weights` 是冻结的1或2样本权重；`counts` 是各条有效答案token数；`denominator` 是 `Σ(w_i × n_i)`；`coefficients` 是每条平均loss真正用于反向传播的系数。不要把 `weight=2` 理解成自动复制两条样本、翻倍训练轮数，或把整组梯度再除8。

这轮对多表类训练样本增加权重，从原基座新建LoRA，保持训练数据、顺序、学习率和一轮更新计划等其他条件一致。它不是接着首轮adapter再训练，也没有把验证题答案加入训练。

内部625题上，首轮EX为73.12%，第二轮72.96%；逐题是16条改善、17条退化。**这次实验没有带来总体提升，所以保留负结果，最终留出对照选择首轮。**不能将一次实验解释成“加权永远无效”，也不能编出没有测过的提升。

“第二轮改变的是训练样本对目标函数的相对贡献，而不是模型结构。加权后部分题改善、部分题退化，总体没有收益，所以最终保留首轮。”

<a id="interview"></a>
## 五、代码阅读路线



> “项目基于开源Qwen模型和Transformers、PEFT实现。我先把问题和完整schema转成输入，构造只监督SQL及结束标记的标签；训练时一次处理一条，按整组答案token数累积梯度，再用AdamW更新LoRA；评测时只给问题和schema，生成一次SQL后执行，并和参考查询结果比较。”

### 打开这五处

| 顺序 | 点击这里 | 代码要点 | 说明 |
| --- | --- | --- | --- |
| 1 | [sft_features.py:100–133](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/data/sft_features.py#L100-L133) | 指出prompt边界、EOS位置和两行labels赋值 | prompt的label为-100，但mask为1，因此看得见、不计损失 |
| 2 | [worker:213–220](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L213-L220) → [cache_release_core.py:25–48](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/cache_release_core.py#L25-L48) | LoRA参数给优化器；每条答案token数决定反向系数 | `backward()`累计梯度，不更新权重；不能把日志浮点数拿去求导 |
| 3 | [target_loss.py:33–54](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/target_loss.py#L33-L54) → [worker:317–344](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_round1_resource_v3_worker.py#L317-L344) | 前一位置预测下一token；整组结束后clip和step | 只减少答案预测位置的词表logits，不删完整上下文；step才更新A/B |
| 4 | [final_holdout_resume_v4.py:278–291](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/infer/final_holdout_resume_v4.py#L278-L291) | generate输入只有问题和schema；按L切出新增token | 训练有gold用于监督，推理没有gold；每题单次贪心，不挑最好结果 |
| 5 | [spider_pair_worker_decode_v2.py:187–207](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/eval/spider_pair_worker_decode_v2.py#L187-L207) | 分别执行gold/predicted，再用result_eq比结果 | 可执行不等于答对，单份数据库EX也不等于证明语义恒等 |

打开 [sft_checkpoint.py:188–224](https://github.com/Actually61/text2sql-posttrain/blob/112b5ce087947820181be80573eeaea0530cb4e8/src/train/sft_checkpoint.py#L188-L224)：“推理需要原基座和adapter；继续训练还需Adam状态、学习率游标、数据顺序和RNG”。

### 读完后的六个检查点（附答案）

1. **`input_ids`里为什么有参考SQL，不是泄露吗？** 这是训练用的teacher forcing；模型用因果mask在每个位置预测下一个token，看不到未来答案。推理输入没有这段参考SQL。
2. **为什么301个输入位置，只有14个监督目标？** 示例前286个是prompt，末尾还有模板换行；真正监督的是13个SQL token和1个EOS。
3. **8次backward为什么只step一次？** 8条梯度累积到同一组参数的`.grad`，按整组token平均后统一更新；期间不清梯度。
4. **哪里能证明基座没有被训练？** 加载准备冻结基座，`adapter_parameters`限定可训练参数，优化器只接收这些参数，段末再核对基座哈希。不是仅靠口头说“使用了LoRA”。
5. **推理时怎么知道加载的是796步adapter？** 生成配置绑定该检查点，代码加载后核对adapter指纹；不是从文件夹名猜，也不是拿冒烟adapter代替。
6. **loss下降是否就说明SQL正确率上升？** 不是。loss衡量参考答案token的预测误差；最终还需在隔离数据上按固定协议生成、执行和评分。

### 暂时不用优先读的内容

- `src/diagnostics/`、Windows Job、PID握手、遥测与互斥锁：属于可靠运行和失败追踪，算法主线读懂后再补读。
- `reports/`、`outputs/`中的少量历史Python脚本：不是默认训练入口，不要看到相似文件名就切换实现。
- 最终评测不同resume版本：用于保留当时失败和修复记录；了解历史边界即可，不需要重放运行。

**数据怎么变成监督目标；梯度怎么变成LoRA参数更新；预测怎么变成可信的评测结果。**
