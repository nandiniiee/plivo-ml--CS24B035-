"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
import json
import os
import re
from collections import Counter

_WORD_RE = re.compile(r'\s+|\S+')
DEFAULT_VOCAB_SIZE = 512  # 256 base bytes + up to 256 learned merges


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(p) for p in (merges or [])]
        self.merge_ranks = {p: i for i, p in enumerate(self.merges)}
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}
        next_id = 256
        for a, b in self.merges:
            self.id_to_bytes[next_id] = self.id_to_bytes[a] + self.id_to_bytes[b]
            next_id += 1
        self.vocab_size = next_id

    def _bpe_word(self, ids):
        ids = list(ids)
        while len(ids) >= 2:
            pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
            ranked = [(self.merge_ranks[p], i) for i, p in enumerate(pairs) if p in self.merge_ranks]
            if not ranked:
                break
            _, i = min(ranked)
            new_id = 256 + self.merge_ranks[(ids[i], ids[i + 1])]
            ids = ids[:i] + [new_id] + ids[i + 2:]
        return ids

    def encode(self, text):
        out = []
        for chunk in _WORD_RE.findall(text):
            out.extend(self._bpe_word(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.id_to_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": self.merges}, f)


def _default_path():
    return os.path.join(os.path.dirname(__file__), "bpe_merges.json")


def load(path=None):
    path = path or _default_path()
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("type") == "bpe":
            return BPETokenizer(merges=data["merges"])
    return BPETokenizer(merges=[])  # falls back to plain byte-level, vocab_size=256


def train_bpe(corpus_path, vocab_size=DEFAULT_VOCAB_SIZE, save_path=None):
    text = open(corpus_path, encoding="utf-8").read()
    word_counts = Counter(_WORD_RE.findall(text))
    word_ids = {w: list(w.encode("utf-8")) for w in word_counts}
    num_merges = vocab_size - 256
    merges = []
    for merge_i in range(num_merges):
        pair_counts = Counter()
        for w, ids in word_ids.items():
            c = word_counts[w]
            for i in range(len(ids) - 1):
                pair_counts[(ids[i], ids[i + 1])] += c
        if not pair_counts:
            break
        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < 2:
            break
        new_id = 256 + merge_i
        merges.append(best_pair)
        for w in list(word_ids.keys()):
            ids = word_ids[w]
            if len(ids) < 2:
                continue
            new_ids, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            word_ids[w] = new_ids
        if merge_i % 20 == 0:
            print(f"merge {merge_i}: {best_pair} -> {new_id} (count={best_count})")
    tok = BPETokenizer(merges=merges)
    save_path = save_path or _default_path()
    tok.save(save_path)
    print(f"saved {len(merges)} merges to {save_path}, vocab_size={tok.vocab_size}")
    return tok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="../data/train_corpus.txt")
    ap.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    args = ap.parse_args()
    train_bpe(args.corpus, vocab_size=args.vocab_size)