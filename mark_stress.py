"""RUAccent in its own Python environment (Transformers 4.x).

Transfer stress marks back to the original text so punctuation, Latin words,
numbers and words dropped by RUAccent are preserved. Existing manual marks win.
"""
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACUTE = "\u0301"
WORD = re.compile(r"[А-Яа-яЁё][А-Яа-яЁё\u0301]*")
VOWELS = "аеёиоуыэюя"


def normalize(word):
    return word.lower().replace("+", "").replace(ACUTE, "").replace("ё", "е")


def transfer_marks(text, raw):
    source = list(WORD.finditer(text))
    predicted = re.findall(r"[а-яё+]+", raw.lower())
    alignment = difflib.SequenceMatcher(a=[normalize(m[0]) for m in source],
                                       b=[normalize(w) for w in predicted], autojunk=False)
    pairs = {}
    for tag, i, end, j, _ in alignment.get_opcodes():
        if tag == "equal":
            pairs.update((index, j + index - i) for index in range(i, end))
    pieces, cursor = [], 0
    for index, match in enumerate(source):
        word = match[0]
        pieces.append(text[cursor:match.start()])
        cursor = match.end()
        if index in pairs and ACUTE not in word:
            prediction = predicted[pairs[index]]
            plain = prediction.replace("+", "")
            marked = "".join(c.upper() if s.isupper() else c for c, s in zip(plain, word, strict=True))
            if "+" in prediction and "ё" not in plain and sum(c in VOWELS for c in plain) >= 2:
                plus = prediction.rindex("+")
                vowel = plus - prediction[:plus].count("+")
                marked = marked[:vowel + 1] + ACUTE + marked[vowel + 1:]
            word = marked
        pieces.append(word)
    pieces.append(text[cursor:])
    return "".join(pieces)


def main():
    request = json.loads(sys.stdin.read())
    # Some pinned RUAccent ONNX models require zero token_type_ids.
    import numpy as np
    from onnxruntime import InferenceSession
    original = InferenceSession.run

    def run(session, names, feed, *args, **kwargs):
        missing = {item.name for item in session.get_inputs()} - set(feed)
        if "token_type_ids" in missing and "input_ids" in feed:
            feed = {**feed, "token_type_ids": np.zeros_like(feed["input_ids"])}
        return original(session, names, feed, *args, **kwargs)

    InferenceSession.run = run
    from ruaccent import RUAccent
    accent = RUAccent()
    accent.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False,
                workdir=str(ROOT / ".cache/ruaccent"))
    result = [transfer_marks(text, accent.process_all(text.lower().replace(ACUTE, "")))
              for text in request["texts"]]
    # Libraries may print to stdout, so use a dedicated result prefix.
    print("STRESS_RESULT=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
