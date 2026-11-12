"""只读检查 Spider 原始数据，并按数据库固定训练/内部验证划分。"""

import argparse
import hashlib
import json
import math
import random
import sqlite3
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def normalized_question(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def quoted_identifier(value):
    return '"' + value.replace('"', '""') + '"'


def within_root(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"配置路径必须位于项目内：{relative}")
    return path


def summarize_duplicates(records):
    exact = Counter((row["db_id"], row["question"], row["query"]) for row in records)
    questions = defaultdict(list)
    for row in records:
        questions[normalized_question(row["question"])].append(row)
    groups = [rows for rows in questions.values() if len(rows) > 1]
    return {
        "exact_duplicate_groups": sum(count > 1 for count in exact.values()),
        "exact_redundant_rows": sum(count - 1 for count in exact.values() if count > 1),
        "normalized_question_duplicate_groups": len(groups),
        "normalized_question_redundant_rows": sum(len(rows) - 1 for rows in groups),
        "normalized_question_groups_across_databases": sum(len({row['db_id'] for row in rows}) > 1 for rows in groups),
        "note": "只统计，不删除。相同问句不等于相同schema或相同答案。",
    }


def cross_duplicates(left, right):
    left_exact = {(row["db_id"], row["question"], row["query"]) for row in left}
    right_exact = {(row["db_id"], row["question"], row["query"]) for row in right}
    left_q = defaultdict(set)
    right_q = defaultdict(set)
    for row in left:
        left_q[normalized_question(row["question"])].add(row["db_id"])
    for row in right:
        right_q[normalized_question(row["question"])].add(row["db_id"])
    common = sorted(set(left_q) & set(right_q))
    return {
        "db_id_overlap": sorted({row["db_id"] for row in left} & {row["db_id"] for row in right}),
        "exact_record_overlap": len(left_exact & right_exact),
        "normalized_question_overlap_groups": len(common),
        "same_db_normalized_question_overlap_groups": sum(bool(left_q[q] & right_q[q]) for q in common),
        "question_fingerprint_examples": [
            {"normalized_question_sha256": hashlib.sha256(q.encode("utf-8")).hexdigest(),
             "left_db_ids": sorted(left_q[q]), "right_db_ids": sorted(right_q[q])}
            for q in common[:20]
        ],
        "note": "仅列问句指纹与db_id，不展示最终评测题目或SQL；跨库通用问句需解释，不能默认删除或称为泄漏。",
    }


def inspect_schema(schema, fatal):
    db_id = schema.get("db_id", "<unknown>")
    tables = schema.get("table_names_original")
    columns = schema.get("column_names_original")
    normalized_tables = schema.get("table_names")
    normalized_columns = schema.get("column_names")
    types = schema.get("column_types")
    if not isinstance(tables, list) or not tables or any(not isinstance(name, str) or not name for name in tables):
        fatal.append(f"schema {db_id}: table_names_original无效")
        return False
    if len({name.casefold() for name in tables}) != len(tables):
        fatal.append(f"schema {db_id}: 原始表名大小写归一后重复")
    if not isinstance(normalized_tables, list) or len(normalized_tables) != len(tables):
        fatal.append(f"schema {db_id}: table_names长度不匹配")
    if not isinstance(columns, list) or not columns or columns[0] != [-1, "*"]:
        fatal.append(f"schema {db_id}: column_names_original第0项不是[-1, '*']")
        return False
    if not isinstance(normalized_columns, list) or len(normalized_columns) != len(columns):
        fatal.append(f"schema {db_id}: column_names长度不匹配")
    elif any(not isinstance(normalized, list) or len(normalized) != 2 or normalized[0] != original[0]
             for original, normalized in zip(columns, normalized_columns)
             if isinstance(original, list) and len(original) == 2):
        fatal.append(f"schema {db_id}: column_names与原始列的表索引不一致")
    if not isinstance(types, list) or len(types) != len(columns):
        fatal.append(f"schema {db_id}: column_types长度不匹配")
    for index, column in enumerate(columns[1:], 1):
        if (not isinstance(column, list) or len(column) != 2 or type(column[0]) is not int
                or not 0 <= column[0] < len(tables) or not isinstance(column[1], str) or not column[1]):
            fatal.append(f"schema {db_id}: 第{index}项列元数据无效")
    primary = schema.get("primary_keys")
    foreign = schema.get("foreign_keys")
    if not isinstance(primary, list) or any(type(index) is not int or not 0 < index < len(columns) for index in primary):
        fatal.append(f"schema {db_id}: primary_keys原数组索引无效")
    if not isinstance(foreign, list) or any(not isinstance(pair, list) or len(pair) != 2 or any(
            type(index) is not int or not 0 < index < len(columns) for index in pair) for pair in foreign):
        fatal.append(f"schema {db_id}: foreign_keys原数组索引无效")
    return True


def inspect_database(path, schema):
    """只执行完整性与schema元数据查询，不读取数据行或执行gold。"""
    result = {"db_id": schema["db_id"], "quick_check": None, "missing_tables": [], "missing_columns": [],
              "extra_tables": [], "extra_columns": [], "error": None}
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + 30
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        checks = [row[0] for row in connection.execute("PRAGMA quick_check").fetchall()]
        result["quick_check"] = checks
        actual = {name.casefold(): name for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        expected_tables = schema["table_names_original"]
        result["missing_tables"] = [name for name in expected_tables if name.casefold() not in actual]
        result["extra_tables"] = [name for key, name in actual.items() if key not in {table.casefold() for table in expected_tables}]
        for table_index, expected_table in enumerate(expected_tables):
            if expected_table.casefold() not in actual:
                continue
            columns = {row[1].casefold(): row[1] for row in connection.execute(
                f"PRAGMA table_info({quoted_identifier(actual[expected_table.casefold()])})")}
            expected_columns = [column[1] for column in schema["column_names_original"][1:] if column[0] == table_index]
            for column in expected_columns:
                if column.casefold() not in columns:
                    result["missing_columns"].append({"table": expected_table, "column": column})
            for key, column in columns.items():
                if key not in {name.casefold() for name in expected_columns}:
                    result["extra_columns"].append({"table": expected_table, "column": column})
    except sqlite3.Error as error:
        result["error"] = str(error)
    finally:
        connection.close()
    return result


def check_data(config, root=PROJECT_ROOT):
    raw = within_root(root, config["raw_dir"])
    fatal, warnings = [], []
    loaded, source_hashes = {}, {}
    provenance = {"enabled": bool(config.get("download_manifest")), "matched_file_count": 0}
    downloaded_files = {}
    if provenance["enabled"]:
        try:
            manifest_path = within_root(root, config["download_manifest"])
            manifest_bytes = manifest_path.read_bytes()
            download_manifest = json.loads(manifest_bytes)
            if download_manifest.get("stage") != "extracted" or download_manifest.get("raw_dir") != config["raw_dir"]:
                raise ValueError("下载清单未完成解压或原始目录不一致")
            downloaded_files = {entry["path"]: entry for entry in download_manifest["files"]}
            if len(downloaded_files) != len(download_manifest["files"]):
                raise ValueError("下载清单包含重复文件名")
            provenance.update(manifest_path=config["download_manifest"],
                              manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                              archive_sha256=download_manifest["archive_sha256"],
                              archive_path=download_manifest["archive_path"],
                              source_url=download_manifest["source_url"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            fatal.append(f"下载清单不可验证：{error}")
    for key in ("train_file", "final_file", "schema_file"):
        path = raw / config[key]
        if not path.is_file():
            fatal.append(f"缺少原始文件：{path.relative_to(root).as_posix()}")
            continue
        source_hashes[path.relative_to(root).as_posix()] = sha256(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, list):
                raise ValueError("顶层不是数组")
            loaded[key] = value
        except (ValueError, UnicodeError) as error:
            fatal.append(f"{config[key]}不能按JSON数组读取：{error}")
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "download_integrity_and_split_only",
              "raw_dir": config["raw_dir"], "fatal_errors": fatal, "warnings": warnings,
              "source_sha256": source_hashes, "gold_sql_executed": False,
              "download_provenance": provenance,
              "final_labels_used_for_training_or_tuning": False}
    if len(loaded) != 3:
        report.update(passed=False, raw_unchanged=True)
        return report, None
    train, final, schemas = loaded["train_file"], loaded["final_file"], loaded["schema_file"]
    for name, rows, expected in (("official_train_spider", train, config["expected_train_examples"]),
                                 ("project_final_from_official_dev", final, config["expected_final_examples"])):
        if len(rows) != expected:
            fatal.append(f"{name}条数{len(rows)}，与配置期望{expected}不同；请核对来源，不自动接受")
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip()
                                                for key in ("db_id", "question", "query"))
                    or not isinstance(row.get("sql"), dict) or not row["sql"]):
                fatal.append(f"{name}第{index}条必需字段无效；未打印内容")
    schema_map = {}
    for index, schema in enumerate(schemas):
        if not isinstance(schema, dict) or not isinstance(schema.get("db_id"), str) or not schema["db_id"]:
            fatal.append(f"schema第{index}项db_id无效")
            continue
        if schema["db_id"] in schema_map:
            fatal.append(f"schema db_id重复：{schema['db_id']}")
        if inspect_schema(schema, fatal):
            schema_map[schema["db_id"]] = schema
    if fatal:
        report.update(passed=False, raw_unchanged=all(sha256(root / name) == value for name, value in source_hashes.items()))
        return report, None
    train_db = {row["db_id"] for row in train}
    final_db = {row["db_id"] for row in final}
    if train_db & final_db:
        fatal.append(f"官方train与项目final的db_id有交集：{sorted(train_db & final_db)}")
    needed_db = sorted(train_db | final_db)
    database_reports = []
    for index, db_id in enumerate(needed_db, 1):
        if Path(db_id).name != db_id or db_id in (".", "..") or ":" in db_id:
            fatal.append(f"无效db_id路径：{db_id}")
            continue
        if db_id not in schema_map:
            fatal.append(f"所需数据库缺少schema：{db_id}")
            continue
        path = raw / config["database_dir"] / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            fatal.append(f"所需SQLite文件缺失：{path.relative_to(root).as_posix()}")
            continue
        source_hashes[path.relative_to(root).as_posix()] = sha256(path)
        try:
            inspected = inspect_database(path, schema_map[db_id])
        except sqlite3.Error as error:
            inspected = {"db_id": db_id, "error": str(error), "quick_check": None,
                         "missing_tables": [], "missing_columns": [], "extra_tables": [], "extra_columns": []}
        database_reports.append(inspected)
        if inspected["error"] or inspected["quick_check"] != ["ok"]:
            fatal.append(f"SQLite完整性检查失败：{db_id}；{inspected['error'] or inspected['quick_check']}")
        if inspected["missing_tables"] or inspected["missing_columns"]:
            fatal.append(f"SQLite与schema有缺失项：{db_id}，详见database_checks")
        if inspected["extra_tables"] or inspected["extra_columns"]:
            warnings.append(f"SQLite含schema之外的表/列：{db_id}，未修改原数据")
        if index % 20 == 0 or index == len(needed_db):
            print(f"只读检查数据库：{index}/{len(needed_db)}", flush=True)
    extras = {"train_others": {"included_in_project_splits": False, "present": False}}
    other_path = raw / "train_others.json"
    if other_path.is_file():
        other_rows = json.loads(other_path.read_text(encoding="utf-8-sig"))
        other_ids = {row["db_id"] for row in other_rows}
        extras["train_others"].update(present=True, example_count=len(other_rows), database_count=len(other_ids),
                                     db_ids=sorted(other_ids), train_db_overlap=sorted(other_ids & train_db),
                                     final_db_overlap=sorted(other_ids & final_db), sha256=sha256(other_path))
    extras["test_files_inventory_only"] = [{"path": path.relative_to(raw).as_posix(), "bytes": path.stat().st_size}
                                           for path in sorted(raw.rglob("*")) if path.is_file() and
                                           path.suffix.lower() in (".json", ".sql", ".txt") and "test" in path.name.lower()]
    cross = cross_duplicates(train, final)
    if cross["normalized_question_overlap_groups"]:
        warnings.append("官方train与final存在归一化相同问句；见匿名指纹统计，未自动删除或作为调参依据")
    current_hashes = {name: sha256(root / name) for name in source_hashes}
    unchanged = current_hashes == source_hashes
    if not unchanged:
        fatal.append("原始文件在检查期间发生变化，停止发布划分")
    if provenance["enabled"] and downloaded_files:
        for relative, actual_hash in source_hashes.items():
            path = root / relative
            raw_relative = path.relative_to(raw).as_posix()
            expected = downloaded_files.get(raw_relative)
            if not expected or expected["sha256"] != actual_hash or expected["size"] != path.stat().st_size:
                fatal.append(f"原始文件与ZIP解压清单不一致：{raw_relative}")
            else:
                provenance["matched_file_count"] += 1
        if sha256(manifest_path) != provenance["manifest_sha256"]:
            fatal.append("下载清单在检查期间改变，停止发布划分")
        provenance["all_selected_files_match"] = provenance["matched_file_count"] == len(source_hashes)
    report.update(
        counts={"official_train_examples": len(train), "official_train_databases": len(train_db),
                "official_dev_examples": len(final), "official_dev_databases": len(final_db),
                "schema_entries": len(schemas), "selected_databases_required": len(needed_db),
                "selected_databases_checked": len(database_reports),
                "sqlite_files_in_database_dir": len(list((raw / config["database_dir"]).rglob("*.sqlite")))},
        database_checks=database_reports, extra_data=extras,
        duplicate_summary={"official_train": summarize_duplicates(train), "official_dev": summarize_duplicates(final),
                           "official_train_vs_final": cross},
        raw_unchanged=unchanged, passed=not fatal,
        final_holdout_policy="官方dev作为本项目最终留出集；本步仅字段、数量、schema/文件和匿名重复检查，不执行其gold或模型预测。",
    )
    return report, {"train": train, "final": final, "schemas": schema_map} if not fatal else None


def build_artifacts(config, report, data, root=PROJECT_ROOT):
    seed, fraction = config["split_seed"], config["validation_db_fraction"]
    if type(seed) is not int or not 0 < fraction < 1:
        raise ValueError("split_seed须为整数，validation_db_fraction须在0和1之间")
    db_ids = sorted({row["db_id"] for row in data["train"]})
    count = max(1, math.ceil(len(db_ids) * fraction))
    if count >= len(db_ids):
        raise ValueError("内部验证分配会用完所有训练数据库")
    random.Random(seed).shuffle(db_ids)
    validation_ids = set(db_ids[:count])
    train = [(index, row) for index, row in enumerate(data["train"]) if row["db_id"] not in validation_ids]
    validation = [(index, row) for index, row in enumerate(data["train"]) if row["db_id"] in validation_ids]
    train_rows, validation_rows = [row for _, row in train], [row for _, row in validation]
    cross = {"train_vs_validation": cross_duplicates(train_rows, validation_rows),
             "train_vs_final": cross_duplicates(train_rows, data["final"]),
             "validation_vs_final": cross_duplicates(validation_rows, data["final"])}
    if any(value["db_id_overlap"] or value["exact_record_overlap"] for value in cross.values()):
        raise ValueError("划分后出现数据库或精确记录交集，拒绝输出")
    source_train = (Path(config["raw_dir"]) / config["train_file"]).as_posix()
    source_final = (Path(config["raw_dir"]) / config["final_file"]).as_posix()
    manifests = {
        "train": {"example_count": len(train), "db_ids": sorted({row["db_id"] for row in train_rows}),
                  "role": "模型训练、训练题上的候选/偏好构造；不得包含validation/final题或标签"},
        "validation": {"example_count": len(validation), "db_ids": sorted(validation_ids),
                       "role": "选择checkpoint、超参数、提示与错误分析；不做梯度训练；dev_small只能从这里取"},
        "final_eval": {"example_count": len(data["final"]), "db_ids": sorted({row["db_id"] for row in data["final"]}),
                       "role": "方案冻结后统一最终评测；来源是official dev，不是official test"},
    }
    for value in manifests.values():
        value["database_count"] = len(value["db_ids"])
    manifest = {"dataset": "Spider 1.0", "version": "project_db_split_v1", "seed": seed,
                "algorithm": "sorted unique train db_id -> random.Random(seed).shuffle -> first ceil(n*fraction) validation",
                "validation_database_fraction": fraction, "splits": manifests,
                "source_sha256": report["source_sha256"], "cross_split_checks": cross,
                "download_provenance": report["download_provenance"],
                "extra_train_others_used": False, "official_test_used": False,
                "python_version": sys.version.split()[0],
                "normalization": "NFKC + casefold + whitespace collapse; punctuation retained",
                "note": "比例按数据库而非题数；不根据模型成绩重抽seed；此划分不提供对预训练污染的保证。"}
    final_manifest = {"source_file": source_final, "sha256": report["source_sha256"][source_final],
                      **manifests["final_eval"], "labels_copied_to_processed": False, "official_test": False}
    stats = {"splits": manifests, "cross_split_checks": cross, "raw_counts": report["counts"],
             "duplicate_summary": {"train": summarize_duplicates(train_rows), "validation": summarize_duplicates(validation_rows),
                                   "final_eval": summarize_duplicates(data["final"])},
             "schema_table_count_by_db": {key: len(value["table_names_original"]) for key, value in data["schemas"].items()},
             "gold_execution_checked": False, "token_lengths_checked": False,
             "next_steps": ["训练/内部验证gold SQL自检", "schema渲染与token长度", "正式评测工具与控制实验"]}
    artifacts = {"split_manifest.json": json_text(manifest), "final_eval_manifest.json": json_text(final_manifest),
                 "stats.json": json_text(stats)}
    for split, rows in (("train", train), ("validation", validation)):
        lines = []
        for index, row in rows:
            if set(row) & {"source_file", "source_index", "project_split"}:
                raise ValueError("原始数据包含将被覆盖的追溯字段")
            record = {**row, "source_file": source_train, "source_index": index, "project_split": split}
            lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
        artifacts[split + ".jsonl"] = "\n".join(lines) + "\n"
    return artifacts, manifest


def publish_immutable(artifacts, target):
    """先检查全部冲突，再写入；内容相同则复用，绝不覆盖不同划分。"""
    for name, value in artifacts.items():
        path = target / name
        if path.exists() and path.read_bytes() != value.encode("utf-8"):
            raise ValueError(f"已有产物与本次不同，拒绝覆盖：{path}；请先审查数据来源或划分变更")
    target.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        path = target / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(value.encode("utf-8"))


def report_markdown(report):
    lines = ["# Spider 1.0 下载与划分检查", "", f"检查通过：{report['passed']}；原始文件未变：{report['raw_unchanged']}", "",
             "本步没有执行gold SQL，没有模型评测，没有展示最终dev问题或答案。", ""]
    if "counts" in report:
        lines += ["## 文件与数据库", "", "| 项目 | 数量 |", "| --- | ---: |"]
        lines += [f"| {key} | {value} |" for key, value in report["counts"].items()]
    if "split_summary" in report:
        lines += ["", "## 固定划分", "", "| 用途 | 题数 | 数据库数 |", "| --- | ---: | ---: |"]
        lines += [f"| {name} | {value['example_count']} | {value['database_count']} |" for name, value in report["split_summary"].items()]
        lines += ["", "官方train_spider按db_id切成训练与内部验证；官方dev整体作本项目final_eval。official test与train_others均未用于本次实验。"]
    lines += ["", "## 致命问题", ""] + (report["fatal_errors"] or ["无。"])
    lines += ["", "## 提醒", ""] + (report["warnings"] or ["无额外提醒。"])
    if "duplicate_summary" in report:
        lines += ["", "## 重复统计", "", "```json", json_text(report["duplicate_summary"]), "```",
                  "", "相同通用问句可以出现在不同schema。上述重复统计不自动等于训练泄漏，也没有据此删掉样本；最终标签未用于训练、调参或模型错误分析。"]
    lines += ["", "## 本步边界", "", "SQLite quick_check和schema对照验证文件与结构；不能代替gold SQL执行检查、语义正确性检查或正式模型指标。", "",
              "第0列[-1, '*']是占位；跳过它渲染DDL，但主键/外键仍使用原数组索引，不普遍减1。",
              "", "完整数据库检查、哈希、附加数据清单与警告见同名JSON报告。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--check-only", action="store_true", help="只检查原始数据，不生成训练/验证划分")
    args = parser.parse_args()
    config = yaml.safe_load(within_root(PROJECT_ROOT, args.config).read_text(encoding="utf-8"))
    report, data = check_data(config)
    if report["passed"] and not args.check_only:
        try:
            artifacts, manifest = build_artifacts(config, report, data)
            publish_immutable(artifacts, within_root(PROJECT_ROOT, config["processed_dir"]))
            report["split_summary"] = manifest["splits"]
            report["split_cross_checks"] = manifest["cross_split_checks"]
            report["processed_dir"] = config["processed_dir"]
            report["processed_sha256"] = {name: hashlib.sha256(value.encode("utf-8")).hexdigest() for name, value in artifacts.items()}
        except ValueError as error:
            report["fatal_errors"].append(str(error))
            report["passed"] = False
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "spider_data_check.json").write_text(json_text(report), encoding="utf-8")
    (reports / "spider_data_check.md").write_text(report_markdown(report), encoding="utf-8")
    if not report["passed"]:
        print("SPIDER_DATA_CHECK: FAIL；详见reports/spider_data_check.json。", flush=True)
        for error in report["fatal_errors"][:12]:
            print(error, flush=True)
        raise SystemExit(1)
    counts = report["counts"]
    print(f"官方train_spider：{counts['official_train_examples']}条/{counts['official_train_databases']}库；官方dev：{counts['official_dev_examples']}条/{counts['official_dev_databases']}库。", flush=True)
    print(f"所需SQLite检查：{counts['selected_databases_checked']}/{counts['selected_databases_required']}；原始文件哈希未变。", flush=True)
    if "split_summary" in report:
        for name, value in report["split_summary"].items():
            print(f"{name}: {value['example_count']}条/{value['database_count']}库", flush=True)
        print("三份划分db_id交集为0；归一化问句重复另列报告，不自动删样本。", flush=True)
    print(f"SPIDER_DATA_CHECK: PASS；提醒 {len(report['warnings'])} 项。", flush=True)
    print(f"报告：{reports / 'spider_data_check.md'}", flush=True)


if __name__ == "__main__":
    main()
