"""Local Russian speech synthesis with a user-supplied reference recording."""
import argparse
import json
import math
import os
import platform
import re
import subprocess
from pathlib import Path

from verify_adapter import ROOT


def split_text(text, limit=220):
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+|\n+", text.strip()):
        for word in sentence.split():
            if current and len(current) + len(word) + 1 > limit:
                pieces.append(current)
                current = ""
            current = (current + " " + word).strip()
        if current:
            pieces.append(current)
            current = ""
    return pieces


def mark(texts):
    python = ROOT / ".venv-stress/bin/python"
    if not python.is_file():
        raise RuntimeError("Run bash install.sh first, or use --manual-stress with premarked text")
    env = {**os.environ, "HF_HOME": str(ROOT / ".cache/stress-hf"), "HF_HUB_DISABLE_TELEMETRY": "1"}
    print("Расстановка ударений (при первом запуске скачиваются модели RUAccent)…", flush=True)
    result = subprocess.run([str(python), str(ROOT / "mark_stress.py")],
                            input=json.dumps({"texts": texts}, ensure_ascii=False),
                            text=True, stdout=subprocess.PIPE, env=env, check=True)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("STRESS_RESULT="):
            marked = json.loads(line.split("=", 1)[1])
            if len(marked) == len(texts) and all(isinstance(s, str) and s for s in marked):
                return marked
    raise RuntimeError("RUAccent did not return the expected text; synthesis was not started")


def install_eos_guard(model, text):
    import mlx.core as mx
    sample = model._sample_token
    eos = model.config.talker_config.codec_eos_token_id
    minimum = math.ceil(sum(c.isalnum() for c in text) / 20 * 12.5)

    def guarded(logits, *args, suppress_tokens=None, generated_tokens=None, **kwargs):
        if suppress_tokens is not None:
            bias = -math.inf if len(generated_tokens or []) < minimum else -3.0
            logits = mx.where(mx.arange(logits.shape[-1]) == eos, logits + bias, logits)
        return sample(logits, *args, suppress_tokens=suppress_tokens,
                      generated_tokens=generated_tokens, **kwargs)

    model._sample_token = guarded
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Russian text to speak")
    text_group.add_argument("--text-file", type=Path, help="UTF-8 text to speak")
    parser.add_argument("--ref-audio", type=Path, required=True, help="Your own WAV or FLAC recording")
    ref_group = parser.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref-text", help="Exact transcript of the entire reference recording")
    ref_group.add_argument("--ref-text-file", type=Path, help="UTF-8 reference transcript")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--manual-stress", action="store_true", help="Use input stress marks as-is; skip RUAccent")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("This launcher requires an Apple Silicon Mac (M1 or newer), without Rosetta")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    ref_text = args.ref_text if args.ref_text is not None else args.ref_text_file.read_text(encoding="utf-8")
    if not text.strip() or not ref_text.strip():
        parser.error("Text and reference transcript must not be empty")
    if not args.ref_audio.is_file() or args.ref_audio.suffix.lower() not in {".wav", ".flac"}:
        parser.error("Provide an existing WAV or FLAC reference file")
    if args.output.suffix.lower() != ".wav" or args.output.exists():
        parser.error("Output must be a new .wav file; existing files are never overwritten")
    import soundfile as sf
    info = sf.info(args.ref_audio)
    if not 3 <= info.duration <= 60:
        parser.error(f"Reference duration must be 3–60 seconds (received {info.duration:.1f})")
    pieces = split_text(text)
    spoken = pieces if args.manual_stress else mark(pieces)
    # RUAccent exits before MLX model loading, releasing its memory.
    from prepare_model import prepare
    model_dir = prepare()
    import mlx.core as mx
    import numpy as np
    from mlx_audio.tts.utils import load_model
    mx.random.seed(args.seed)
    model = load_model(str(model_dir))
    audio, sample_rate = [], model.sample_rate
    for number, (plain, accented) in enumerate(zip(pieces, spoken, strict=True), 1):
        print(f"[{number}/{len(pieces)}] {accented}", flush=True)
        original = install_eos_guard(model, plain)
        max_tokens = min(1600, math.ceil((sum(c.isalnum() for c in plain) / 5 + 10) * 12.5))
        try:
            for result in model.generate(text=accented, lang_code="russian",
                                         ref_audio=str(args.ref_audio.resolve()), ref_text=ref_text.strip(),
                                         temperature=0.9, max_tokens=max_tokens, stream=False, verbose=False):
                waveform = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if not waveform.size or not np.isfinite(waveform).all():
                    raise RuntimeError("Model returned empty or invalid audio")
                if waveform.size / sample_rate >= (max_tokens - 2) / 12.5:
                    raise RuntimeError("Generation hit its duration limit; shorten the input sentence")
                audio.append(waveform)
        finally:
            model._sample_token = original
        if number < len(pieces):
            audio.append(np.zeros(round(sample_rate * 0.18), dtype=np.float32))
    if not audio:
        raise RuntimeError("No audio generated")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = np.concatenate(audio)
    with args.output.open("xb") as stream:
        sf.write(stream, samples, sample_rate, format="WAV", subtype="PCM_16")
    print(f"Готово: {args.output.resolve()} ({len(samples) / sample_rate:.1f} с, {sample_rate} Гц)")


if __name__ == "__main__":
    main()
