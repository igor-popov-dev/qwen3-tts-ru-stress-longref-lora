"""Verify the distributed adapter without loading models or importing ML libraries."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    manifest = json.loads((ROOT / "adapter/manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if name not in {"adapter.json", "adapter.safetensors"}:
            raise ValueError("Unexpected file in adapter manifest")
        if sha256(ROOT / "adapter" / name) != expected:
            raise ValueError(f"Adapter checksum mismatch: {name}")
    config = json.loads((ROOT / "adapter/adapter.json").read_text())
    if config["base"] != "Qwen/Qwen3-TTS-12Hz-1.7B-Base" or config["applied_to"] != "talker.model":
        raise ValueError("Unsupported adapter base or target")
    if config["r"] != 16 or config["alpha"] != 32:
        raise ValueError("Unexpected adapter rank/scale")
    with (ROOT / "adapter/adapter.safetensors").open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < size < 1024 * 1024:
            raise ValueError("Invalid safetensors header")
        header = json.loads(stream.read(size))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if len(tensors) != manifest["tensor_count"]:
        raise ValueError("Unexpected tensor count")
    for name, spec in tensors.items():
        if not name.endswith((".lora_A.weight", ".lora_B.weight")) or len(spec["shape"]) != 2:
            raise ValueError(f"Unexpected tensor: {name}")
    return manifest, config


if __name__ == "__main__":
    manifest, _ = verify()
    print(f"LoRA verified: {manifest['tensor_count']} tensors; SHA-256 matches.")
