"""下载官网链接的 Spider 1.0 ZIP，固定本地哈希并安全解压；不安装新依赖。"""

import argparse
import hashlib
import http.cookiejar
import json
import os
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs/data_source.yaml"
MANIFEST = ROOT / "reports/spider_download_manifest.json"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_manifest(value):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.inputs = {}
        self.in_form = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self.action = attrs.get("action")
            self.in_form = True
        elif tag == "input" and self.in_form and "name" in attrs:
            self.inputs[attrs["name"]] = attrs.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.in_form = False


def open_download(opener, file_id, position):
    url = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode({"id": file_id, "export": "download"})
    headers = {"Accept-Encoding": "identity"}
    if position:
        headers["Range"] = f"bytes={position}-"
    response = opener.open(urllib.request.Request(url, headers=headers), timeout=30)
    if "text/html" in response.headers.get("Content-Type", ""):
        with response:
            html = response.read(128 * 1024).decode("utf-8")
        form = ConfirmationForm()
        form.feed(html)
        action = urllib.parse.urlsplit(form.action or "")
        if (action.scheme != "https" or action.netloc != "drive.usercontent.google.com"
                or action.path != "/download" or form.inputs.get("id") != file_id):
            raise RuntimeError("Google Drive 返回的不是预期公开文件确认页，停止下载。")
        url = form.action + "?" + urllib.parse.urlencode(form.inputs)
        response = opener.open(urllib.request.Request(url, headers=headers), timeout=30)
    if "text/html" in response.headers.get("Content-Type", ""):
        response.close()
        raise RuntimeError("仍收到网页而非 ZIP，可能有下载配额或访问限制。")
    return response


def download(config):
    archive = ROOT / config["archive_path"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    expected = config["archive_sha256"]
    if archive.exists():
        actual = digest(archive)
        if not expected or actual != expected:
            raise RuntimeError("已有压缩包与固定哈希不一致，请先检查来源，不能自动替换。")
        print("已有 ZIP 与固定 SHA256 一致，复用。", flush=True)
        return
    partial = archive.with_suffix(".zip.part")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    size = partial.stat().st_size if partial.exists() else 0
    for attempt in range(1, 4):
        try:
            with open_download(opener, config["google_drive_file_id"], size) as response:
                content_range = response.headers.get("Content-Range")
                if size and response.status == 206:
                    if not content_range or not content_range.startswith(f"bytes {size}-"):
                        raise RuntimeError("续传字节范围不一致。")
                    mode = "ab"
                elif response.status == 200:
                    if size:
                        print("服务端未提供续传，重新接收完整 ZIP。", flush=True)
                    size, mode = 0, "wb"
                else:
                    raise RuntimeError(f"不支持的下载响应状态：{response.status}")
                length = response.headers.get("Content-Length")
                total = size + int(length) if length else None
                if total and total > config["max_archive_bytes"]:
                    raise RuntimeError("压缩包超过本步大小限制。")
                print(f"官方文件：spider_data.zip；响应总大小：{total if total else '未提供'} 字节", flush=True)
                next_progress = size + 16 * 1024**2
                with partial.open(mode) as handle:
                    while chunk := response.read(1024**2):
                        if size == 0 and not chunk.startswith(b"PK\x03\x04"):
                            raise RuntimeError("响应不是预期的 ZIP 文件。")
                        handle.write(chunk)
                        size += len(chunk)
                        if size > config["max_archive_bytes"]:
                            raise RuntimeError("下载超过大小限制，已停止。")
                        if size >= next_progress:
                            handle.flush()
                            print(f"已下载 {size / 1024**2:.1f} MiB", flush=True)
                            next_progress += 16 * 1024**2
                if total is not None and size != total:
                    raise OSError("响应提前结束。")
                break
        except (OSError, urllib.error.URLError) as error:
            size = partial.stat().st_size if partial.exists() else 0
            if attempt == 3:
                raise RuntimeError(f"下载失败，保留临时文件；错误类型：{type(error).__name__}") from None
            print(f"网络重试 {attempt}/3，已接收 {size} 字节。", flush=True)
            time.sleep(attempt)
    if not zipfile.is_zipfile(partial):
        raise RuntimeError("ZIP 目录损坏或下载不完整。")
    actual = digest(partial)
    if expected and actual != expected:
        raise RuntimeError("新的下载内容与已固定 SHA256 不一致，保留临时文件供检查。")
    partial.replace(archive)
    config["archive_sha256"] = actual
    SOURCE_CONFIG.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    save_manifest({"dataset": config["dataset"], "source_page": config["source_page"], "source_url": config["source_url"],
                   "archive_path": config["archive_path"], "archive_size": archive.stat().st_size,
                   "archive_sha256": actual, "checksum_source": "locally_computed_on_first_download_not_publisher_checksum",
                   "license": config["license"], "license_url": config["license_url"],
                   "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "stage": "downloaded"})
    print(f"SPIDER_DOWNLOAD: PASS；SHA256={actual}", flush=True)


def extract(config):
    archive = ROOT / config["archive_path"]
    if not config["archive_sha256"] or digest(archive) != config["archive_sha256"]:
        raise RuntimeError("请先完成 ZIP 下载与本地哈希固定。")
    destination = (ROOT / config["raw_dir"]).resolve()
    if not destination.is_relative_to(ROOT.resolve()):
        raise RuntimeError("原始数据目录必须在项目内。")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    with zipfile.ZipFile(archive) as zipped:
        infos = zipped.infolist()
        roots = [PurePosixPath(info.filename).parent for info in infos
                 if PurePosixPath(info.filename).name == "train_spider.json" and "__MACOSX" not in info.filename]
        if len(roots) != 1:
            raise RuntimeError(f"无法唯一定位原始 train_spider.json，候选目录数：{len(roots)}")
        prefix = roots[0]
        if sum(info.file_size for info in infos) > config["max_extracted_bytes"]:
            raise RuntimeError("解压总大小超过本步限制。")
        bad_member = zipped.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP CRC 校验失败：{bad_member}")
        selected = []
        skipped_entries = []
        targets = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16)):
                raise RuntimeError("压缩包包含不安全路径或符号链接。")
            if info.is_dir() or not path.is_relative_to(prefix):
                continue
            relative = path.relative_to(prefix)
            if relative.name in (".DS_Store",) or "__MACOSX" in relative.parts:
                continue
            if ":" in info.filename:
                # 官方 test 附带两个 macOS 时间命名 CSV；Windows 不能原名落盘。
                # 仅跳过这些辅助 CSV，原始字节仍在 ZIP 内；JSON/SQLite 不允许跳过。
                if (relative.parts[0] == "test_database" and "data_csv" in relative.parts
                        and relative.suffix.lower() == ".csv"):
                    skipped_entries.append({"path": info.filename, "size": info.file_size,
                        "reason": "auxiliary_test_csv_windows_invalid_colon_preserved_in_archive"})
                    continue
                raise RuntimeError(f"Windows 无法原样保存这个数据文件名：{relative}")
            target = (destination / Path(*relative.parts)).resolve()
            if not target.is_relative_to(destination) or str(target).casefold() in targets:
                raise RuntimeError("解压路径越界或发生文件名冲突。")
            targets.add(str(target).casefold())
            selected.append((info, relative, target))
        for required in ("train_spider.json", "dev.json", "tables.json"):
            if not any(str(relative) == required for _, relative, _ in selected):
                raise RuntimeError(f"压缩包缺少 {required}")
        files = []
        for index, (info, relative, target) in enumerate(selected, 1):
            # 已有原始文件只做校验，绝不静默覆盖可能发生的修改。
            target.parent.mkdir(parents=True, exist_ok=True)
            entry_hash = hashlib.sha256()
            with zipped.open(info) as source:
                if target.exists():
                    for chunk in iter(lambda: source.read(1024**2), b""):
                        entry_hash.update(chunk)
                    if target.stat().st_size != info.file_size or digest(target) != entry_hash.hexdigest():
                        raise RuntimeError(f"已有原始文件与 ZIP 不同：{relative}")
                else:
                    with target.open("xb") as output:
                        for chunk in iter(lambda: source.read(1024**2), b""):
                            entry_hash.update(chunk)
                            output.write(chunk)
            files.append({"path": relative.as_posix(), "size": info.file_size, "sha256": entry_hash.hexdigest()})
            if index % 100 == 0:
                print(f"已解压/核对 {index}/{len(selected)} 个文件", flush=True)
        extraction = {"stage": "extracted", "raw_dir": config["raw_dir"], "archive_dataset_prefix": prefix.as_posix(),
                      "zip_entry_count": len(infos), "extracted_file_count": len(files),
                      "skipped_auxiliary_entries": skipped_entries,
                      "extracted_size": sum(entry["size"] for entry in files), "files": files}
        if manifest.get("stage") == "extracted":
            if any(manifest.get(key) != value for key, value in extraction.items()):
                raise RuntimeError("本次提取清单与已固定版本不同，拒绝覆盖来源记录。")
            print("全部原始文件与已有清单一致，保留原清单及其哈希。", flush=True)
        else:
            manifest.update(extraction, extracted_at_utc=datetime.now(timezone.utc).isoformat())
            save_manifest(manifest)
    print(f"SPIDER_EXTRACT: PASS；原始目录 {destination}；文件数 {len(files)}", flush=True)
    print("这里只校验归档和原始文件；问题数量、schema 和 SQLite 覆盖由 check_spider.py 检查。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("download", "extract"))
    args = parser.parse_args()
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    (download if args.stage == "download" else extract)(config)


if __name__ == "__main__":
    main()
