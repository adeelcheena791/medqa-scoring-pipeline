"""Build the frozen item pool from MIRAGE benchmark.json.

Outputs data/items.jsonl with one row per question:
  qid, dataset, question, options (shuffled, letters A..), key (letter after shuffle),
  orig_key, perm (original letters in new order), split.

Protocol references: Sections 6-7 (datasets, exclusions), 11 (option shuffle),
23 (split by question ID: fit 30% / calibration 20% / test 50%; MMLU-Med = external test).
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

SPLIT_SEED = 20261004
SHUFFLE_SEED = 20261005
EXPECTED_SHA256 = "6f7f08c64cd2efe02a5d0c247229813c90db345d9dd6e3a451b5d24146d0f8fa"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


VISUAL = re.compile(r"\b(figure|image|photograph|picture|radiograph[^.]{0,40}shown|(shown|seen) (in|on) the (x-ray|ct|mri))\b", re.I)


def references_missing_figure(q):
    """True when the stem depends on a visual that is not in the text.
    'Shown below' counts only when no digits follow it (i.e. the data are absent)."""
    if VISUAL.search(q):
        return True
    for m in re.finditer(r"shown below|table below", q, re.I):
        if not re.search(r"\d", q[m.end():]):
            return True
    return False


def item_seed(qid, base):
    return int(hashlib.sha256(f"{base}:{qid}".encode()).hexdigest()[:12], 16)


def shuffle_options(qid, options, key):
    letters = sorted(options)
    order = letters[:]
    random.Random(item_seed(qid, SHUFFLE_SEED)).shuffle(order)
    new_letters = [chr(ord("A") + i) for i in range(len(order))]
    new_opts = {nl: options[ol] for nl, ol in zip(new_letters, order)}
    new_key = new_letters[order.index(key)]
    return new_opts, new_key, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="data/benchmark.json")
    ap.add_argument("--out", default="data/items.jsonl")
    ap.add_argument("--n_medmcqa", type=int, default=0,
                    help="MedMCQA items to sample (0 = none; decided after pilot)")
    args = ap.parse_args()

    digest = sha256(args.benchmark)
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"benchmark.json hash mismatch: {digest}")
    bench = json.load(open(args.benchmark))

    rows, excluded, seen = [], [], {}
    sources = [("medqa", "internal"), ("mmlu", "external")]
    if args.n_medmcqa:
        sources.insert(1, ("medmcqa", "internal"))

    for ds, role in sources:
        items = list(bench[ds].items())
        if ds == "medmcqa":
            random.Random(SPLIT_SEED).shuffle(items)
            items = items[: args.n_medmcqa]
        for qid, it in items:
            reason = None
            if it.get("answer") not in it.get("options", {}):
                reason = "missing_or_invalid_key"
            elif references_missing_figure(it["question"]):
                reason = "references_missing_figure"
            stem = norm(it["question"]) + "||" + "|".join(sorted(norm(o) for o in it["options"].values()))
            if reason is None and stem in seen:
                reason = f"duplicate_of:{seen[stem]}"
            if reason:
                excluded.append({"qid": qid, "dataset": ds, "reason": reason})
                continue
            seen[stem] = qid
            opts, key, perm = shuffle_options(qid, it["options"], it["answer"])
            rows.append({"qid": qid, "dataset": ds, "role": role,
                         "question": it["question"], "options": opts, "key": key,
                         "orig_key": it["answer"], "perm": perm})

    internal = sorted([r for r in rows if r["role"] == "internal"], key=lambda r: r["qid"])
    random.Random(SPLIT_SEED).shuffle(internal)
    n = len(internal)
    for i, r in enumerate(internal):
        r["split"] = "fit" if i < 0.3 * n else ("calib" if i < 0.5 * n else "test")
    for r in rows:
        if r["role"] == "external":
            r["split"] = "external"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(Path(args.out).with_name("excluded.jsonl"), "w") as f:
        for e in excluded:
            f.write(json.dumps(e) + "\n")

    from collections import Counter
    print("benchmark sha256 ok")
    print("kept:", Counter((r["dataset"], r["split"]) for r in rows))
    print("excluded:", Counter(e["reason"].split(":")[0] for e in excluded))
    print("key letters after shuffle:", Counter(r["key"] for r in rows))


if __name__ == "__main__":
    main()
