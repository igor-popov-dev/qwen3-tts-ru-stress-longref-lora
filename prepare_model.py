"""Merge the released LoRA into the original BF16 base, then quantize to MLX 8-bit.

The operation is W = BF16(FP32(W) + (B @ A) * alpha / rank), followed by
the same per-module quantization as the pinned mlx-community conversion.
Only the adapter is distributed. Base weights are downloaded on the user's Mac.
"""
import json
import os
import shutil
from pathlib import Path

from verify_adapter import ROOT, sha256, verify

# These must be set before importing huggingface_hub/Xet, including its log cache.
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BASE_REPO = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BASE_REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
MLX_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
MLX_REVISION = "e7dd0585652209fa0d7783659aad4e8a324de11c"
MODEL_DIR = ROOT / "models/ru-stress-longref-8bit"
ASSETS = ["config.json", "generation_config.json", "merges.txt", "vocab.json",
          "tokenizer_config.json", "preprocessor_config.json", "speech_tokenizer/*"]


def merge_matrix(weight, pair, scale):
    import mlx.core as mx
    delta = (pair["B"].astype(mx.float32) @ pair["A"].astype(mx.float32)) * scale
    if delta.shape != weight.shape:
        raise ValueError(f"LoRA shape mismatch: {delta.shape} != {weight.shape}")
    return (weight.astype(mx.float32) + delta).astype(weight.dtype)


def prepare():
    manifest, adapter_config = verify()
    stamp = MODEL_DIR / "merge.json"
    identity = {"base_revision": BASE_REVISION, "mlx_revision": MLX_REVISION,
                "adapter_sha256": manifest["sha256"]["adapter.safetensors"],
                "adapter_config_sha256": manifest["sha256"]["adapter.json"]}
    if MODEL_DIR.exists():
        if not stamp.is_file():
            raise RuntimeError(f"Incomplete model directory: {MODEL_DIR}")
        receipt = json.loads(stamp.read_text())
        if receipt.get("identity") != identity:
            raise RuntimeError("The prepared model belongs to another adapter/version")
        weights = MODEL_DIR / "model.safetensors"
        if not weights.is_file() or weights.stat().st_size != receipt["weights_bytes"]:
            raise RuntimeError("Prepared weights are missing or incomplete")
        for name in receipt["assets"]:
            if not (MODEL_DIR / name).is_file():
                raise RuntimeError(f"Missing cached asset: {name}")
        return MODEL_DIR

    import mlx.core as mx
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model

    # Cache stays next to this checkout. No shared production cache is modified.
    cache = ROOT / ".cache/huggingface/hub"
    cache.mkdir(parents=True, exist_ok=True)
    base_cached = try_to_load_from_cache(BASE_REPO, "model.safetensors", revision=BASE_REVISION, cache_dir=cache)
    tokenizer_cached = try_to_load_from_cache(MLX_REPO, "speech_tokenizer/model.safetensors",
                                             revision=MLX_REVISION, cache_dir=cache)
    required_gib = 3 if isinstance(base_cached, str) and isinstance(tokenizer_cached, str) else 8
    if shutil.disk_usage(ROOT).free < required_gib * 1024**3:
        raise RuntimeError(f"At least {required_gib} GiB of free disk space is needed for this merge")
    print("Downloading the pinned base model and tokenizer (first run only)…", flush=True)
    base = Path(snapshot_download(BASE_REPO, revision=BASE_REVISION, cache_dir=cache,
                                  allow_patterns=["model.safetensors"]))
    reference = Path(snapshot_download(MLX_REPO, revision=MLX_REVISION, cache_dir=cache,
                                      allow_patterns=ASSETS + ["model.safetensors.index.json"]))
    config = json.loads((reference / "config.json").read_text())
    quant = config["quantization"]
    index = json.loads((reference / "model.safetensors.index.json").read_text())["weight_map"]
    quantized = {key[:-len(".scales")] for key in index if key.endswith(".scales")}
    adapter = mx.load(str(ROOT / "adapter/adapter.safetensors"))
    pairs = {}
    for key, value in adapter.items():
        stem, side = key.rsplit(".lora_", 1)
        pairs.setdefault(stem, {})[side.split(".")[0]] = value
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Incomplete LoRA A/B pair")
    scale = adapter_config["alpha"] / adapter_config["r"]
    prefix = adapter_config["applied_to"] + "."
    weights = mx.load(str(base / "model.safetensors"))
    result, merged = {}, set()
    print(f"Merging {len(pairs)} LoRA matrices, quantizing to 8-bit…", flush=True)
    for number, key in enumerate(sorted(weights)):
        weight = weights.pop(key)
        if "position_ids" in key:
            continue
        weight = Model.sanitize({key: weight})[key]
        module = key[:-len(".weight")] if key.endswith(".weight") else ""
        stem = module[len(prefix):] if module.startswith(prefix) else ""
        if stem in pairs:
            weight = merge_matrix(weight, pairs[stem], scale)
            merged.add(stem)
        if module in quantized:
            w, scales, biases = mx.quantize(weight, group_size=quant["group_size"],
                                            bits=quant["bits"], mode=quant.get("mode", "affine"))
            mx.eval(w, scales, biases)
            result[module + ".weight"] = w
            result[module + ".scales"] = scales
            result[module + ".biases"] = biases
        else:
            mx.eval(weight)
            result[key] = weight
        del weight
        if number % 200 == 0:
            mx.clear_cache()
    if merged != set(pairs) or set(result) != set(index):
        raise ValueError("Merged model keys do not match the pinned MLX architecture")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = MODEL_DIR.with_name(MODEL_DIR.name + ".building")
    if staging.exists():
        raise RuntimeError(f"Previous interrupted merge remains at {staging}; move it aside before retrying")
    staging.mkdir()
    # Assets are linked to this checkout's cache to avoid duplicating the tokenizer.
    # They are local runtime files, never part of the public repository.
    assets = []
    for path in reference.rglob("*"):
        relative = path.relative_to(reference)
        if path.is_file() and str(relative) not in {"model.safetensors.index.json", "model.safetensors"}:
            dest = staging / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(path.resolve(), dest.parent))
            assets.append(str(relative))
    outfile = staging / "model.safetensors"
    mx.save_safetensors(str(outfile), result, metadata={"format": "mlx"})
    total = sum(value.nbytes for value in result.values())
    (staging / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {key: "model.safetensors" for key in result}}, indent=2))
    (staging / "merge.json").write_text(json.dumps(
        {"identity": identity, "merged_matrices": len(merged), "weights_bytes": outfile.stat().st_size,
         "weights_sha256": sha256(outfile), "assets": sorted(assets)}, indent=2) + "\n")
    staging.rename(MODEL_DIR)
    print(f"Model prepared: {MODEL_DIR}", flush=True)
    return MODEL_DIR


if __name__ == "__main__":
    prepare()
