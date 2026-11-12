"""分步下载官方模型，并固定仓库版本、核验文件大小和哈希。"""

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["HF_XET_CACHE"] = str(HF_CACHE / "xet")
# 本次 Xet 传输长时间无可见进度；禁用该后端，小文件走官方库 HTTP 路径。
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

import yaml
from huggingface_hub import HfApi, hf_hub_url, snapshot_download

SMALL_FILES = {
    "config.json", "generation_config.json", "tokenizer_config.json",
    "tokenizer.json", "merges.txt", "vocab.json", "added_tokens.json",
    "special_tokens_map.json", "chat_template.jinja", "README.md", "LICENSE",
    "model.safetensors.index.json",
}


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_file(model_dir, entry):
    path = model_dir / entry["name"]
    if not path.is_file() or path.stat().st_size != entry["size"]:
        raise RuntimeError(f"文件不存在或大小不符：{path}")
    expected = entry["sha256"] or entry["git_blob_sha1"]
    if not expected:
        raise RuntimeError(f"官方元数据未提供文件哈希：{entry['name']}")
    algorithm = "sha256" if entry["sha256"] else "sha1"
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        # 普通 Git 文件的 blob 哈希包含对象头；LFS 权重使用文件 SHA256。
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"文件哈希不符：{entry['name']}")
    return {"size": path.stat().st_size, "algorithm": algorithm, "hash": actual}


def download_weight_http(config, model_dir, entry):
    """从官方 URL 分段下载；临时文件始终是连续前缀，完成后再验证 SHA256。"""
    destination = model_dir / entry["name"]
    if destination.is_file():
        verify_file(model_dir, entry)
        print(f"已有完整权重：{entry['name']}", flush=True)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".http.part")
    total = entry["size"]
    position = partial.stat().st_size if partial.exists() else 0
    if position > total:
        raise RuntimeError("HTTP 临时文件大于官方文件，请检查下载目录。")
    url = hf_hub_url(config["model_id"], entry["name"], revision=config["revision"], endpoint=config["endpoint"])
    failures = 0
    with partial.open("ab") as target:
        while position < total:
            end = min(position + 64 * 1024**2, total) - 1
            request = urllib.request.Request(url, headers={"Range": f"bytes={position}-{end}", "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    expected_range = f"bytes {position}-{end}/{total}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise RuntimeError("服务端没有返回所请求的连续字节区间。")
                    while position <= end:
                        chunk = response.read(min(1024**2, end + 1 - position))
                        if not chunk:
                            raise OSError("下载响应提前结束。")
                        target.write(chunk)
                        position += len(chunk)
                    target.flush()
                failures = 0
                print(f"权重进度：{position / 1e9:.3f}/{total / 1e9:.3f} GB ({100 * position / total:.1f}%)", flush=True)
            except (OSError, TimeoutError) as error:
                target.flush()
                failures += 1
                if failures >= 3:
                    raise RuntimeError(f"HTTP 下载连续失败，已保留 {position} 字节供续传；类型 {type(error).__name__}") from None
                print(f"HTTP 重试 {failures}/3，已保存 {position} 字节；类型 {type(error).__name__}", flush=True)
                time.sleep(failures)
    with partial.open("rb") as downloaded:
        digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError("下载权重 SHA256 不符；临时文件保留用于排查，未发布为模型。")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["metadata", "weights"], required=True)
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "configs" / "model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_dir = (PROJECT_ROOT / config["local_dir"]).resolve()
    if not model_dir.is_relative_to(PROJECT_ROOT.resolve()):
        raise RuntimeError("模型目录必须位于本项目中。")
    manifest_path = PROJECT_ROOT / "reports" / "model_manifest.json"

    if args.stage == "metadata":
        api = HfApi(endpoint=config["endpoint"], token=False)
        info = api.model_info(
            config["model_id"], revision=config["revision"],
            files_metadata=True, timeout=30, token=False,
        )
        if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
            raise RuntimeError("未能取得确定的模型仓库 commit。")
        files = []
        for sibling in info.siblings:
            name = sibling.rfilename
            is_weight = bool(re.fullmatch(r"model(?:-\d+-of-\d+)?\.safetensors", name))
            if name not in SMALL_FILES and not is_weight:
                continue
            lfs = sibling.lfs
            sha256 = getattr(lfs, "sha256", None) if lfs else None
            if isinstance(lfs, dict):
                sha256 = lfs.get("sha256")
            if sibling.size is None:
                raise RuntimeError(f"元数据缺少文件大小：{name}")
            files.append({
                "name": name, "size": sibling.size, "sha256": sha256,
                "git_blob_sha1": sibling.blob_id, "is_weight": is_weight,
            })
        names = {entry["name"] for entry in files}
        if not {"config.json", "tokenizer_config.json", "tokenizer.json"}.issubset(names):
            raise RuntimeError("官方仓库缺少当前加载方案所需的配置或分词器文件。")
        if not any(entry["is_weight"] for entry in files):
            raise RuntimeError("未找到官方 safetensors 权重。")
        config["revision"] = info.sha
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        manifest = {
            "model_id": config["model_id"], "revision": info.sha,
            "endpoint": config["endpoint"], "local_dir": config["local_dir"],
            "source_url": f"{config['endpoint']}/{config['model_id']}/tree/{info.sha}",
            "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
            "files": files, "verified_files": {},
        }
        write_json(manifest_path, manifest)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["revision"] != config["revision"]
                or manifest["model_id"] != config["model_id"]
                or manifest["local_dir"] != config["local_dir"]):
            raise RuntimeError("模型配置与已固定的文件清单不一致。")
        files = manifest["files"]

    selected = [entry for entry in files if args.stage == "weights" or not entry["is_weight"]]
    print(f"模型：{config['model_id']}", flush=True)
    print(f"固定 commit：{config['revision']}", flush=True)
    print(f"权重总大小：{sum(entry['size'] for entry in files if entry['is_weight']) / 1e9:.3f} GB", flush=True)
    print(f"本步文件：{len(selected)} 个，清单大小 {sum(entry['size'] for entry in selected) / 1e6:.1f} MB", flush=True)
    print(f"保存位置：{model_dir}", flush=True)
    small_selected = [entry for entry in selected if not entry["is_weight"]]
    snapshot_download(
        repo_id=config["model_id"], revision=config["revision"],
        endpoint=config["endpoint"], local_dir=model_dir,
        cache_dir=HF_CACHE / "hub", token=False,
        allow_patterns=[entry["name"] for entry in small_selected], max_workers=2,
        etag_timeout=30,
    )
    for entry in selected:
        if entry["is_weight"]:
            download_weight_http(config, model_dir, entry)
    for entry in selected:
        print(f"校验：{entry['name']}", flush=True)
        manifest["verified_files"][entry["name"]] = verify_file(model_dir, entry)
    manifest["last_completed_stage"] = args.stage
    manifest["download_transport"] = "huggingface_hub_metadata_and_urllib_range_weights"
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"DOWNLOAD_{args.stage.upper()}: PASS", flush=True)
    print(f"文件清单和校验结果：{manifest_path}", flush=True)


if __name__ == "__main__":
    main()
