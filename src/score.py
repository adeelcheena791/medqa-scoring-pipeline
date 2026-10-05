"""Letter-probability scorer (protocol Sections 11, 14, 24).

For each item and condition, builds the chat prompt, runs ONE forward pass,
and reads the next-token logits for the option letters. Writes one JSON line
per (item, condition) with the full renormalised option distribution, the raw
probability mass on letter tokens (format check), and the greedy answer.

Resumable: rows already present in the output file are skipped, so a Kaggle
session that hits its time limit can simply be restarted.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = ("You are a medical expert answering a multiple-choice exam question. "
          "Reply with the single letter of the correct option only.")

TEMPLATES = {
    # C0: clean question, no context
    "C0": "{question}\n\n{options}\n\nAnswer:",
    # Passage conditions (C+, Cn, C3a, C3b, C3n, C6) share one layout
    "passage": "Reference material:\n{context}\n\n{question}\n\n{options}\n\nAnswer:",
    # C1 / C2: user statement before the question
    "statement": "{context}\n\n{question}\n\n{options}\n\nAnswer:",
}
LAYOUT = {"C0": "C0", "C+": "passage", "Cn": "passage", "C3a": "passage",
          "C3b": "passage", "C3n": "passage", "C6": "passage",
          "C1": "statement", "C2": "statement"}


def format_options(opts):
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))


def build_messages(tok, item, condition, context=None):
    body = TEMPLATES[LAYOUT[condition]].format(
        question=item["question"].strip(), options=format_options(item["options"]),
        context=(context or "").strip())
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": body}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some chat templates (e.g. Gemma) reject a system role: merge it into the user turn
        msgs = [{"role": "user", "content": SYSTEM + "\n\n" + body}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_token_ids(tok, letters):
    """All single-token spellings of each letter ('A', ' A'). Fails loudly otherwise."""
    ids = {}
    for L in letters:
        cands = set()
        for s in (L, " " + L):
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) == 1:
                cands.add(enc[0])
        if not cands:
            raise SystemExit(f"Letter {L!r} is not a single token for this tokenizer")
        ids[L] = sorted(cands)
    return ids


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


@torch.no_grad()
def score_batch(model, tok, prompts, letters, lid):
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[:, -1, :].float()           # left padding -> last position
    logp = torch.log_softmax(logits, dim=-1).cpu()
    out = []
    for row in logp:
        lp = {L: logsumexp([row[i].item() for i in lid[L]]) for L in letters}
        mass = sum(math.exp(v) for v in lp.values())
        z = logsumexp(list(lp.values()))
        dist = {L: math.exp(v - z) for L, v in lp.items()}
        out.append({"probs": dist, "letter_mass": mass,
                    "pred": max(dist, key=dist.get)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--items", default="data/items.jsonl")
    ap.add_argument("--contexts", default=None, help="jsonl: qid, condition, context")
    ap.add_argument("--conditions", default="C0")
    ap.add_argument("--splits", default="fit,calib,test,external")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
        device_map="auto")
    model.eval()

    splits = set(args.splits.split(","))
    items = [json.loads(l) for l in open(args.items)]
    items = [it for it in items if it["split"] in splits]
    if args.limit:
        items = items[: args.limit]

    ctx = {}
    if args.contexts:
        for l in open(args.contexts):
            r = json.loads(l)
            ctx[(r["qid"], r["condition"])] = r["context"]

    done = set()
    if os.path.exists(args.out):
        for l in open(args.out):
            r = json.loads(l)
            done.add((r["qid"], r["condition"]))

    jobs = []
    for cond in args.conditions.split(","):
        for it in items:
            if (it["qid"], cond) in done:
                continue
            if cond != "C0" and (it["qid"], cond) not in ctx:
                continue  # no qualifying context for this item (reported, not an error)
            jobs.append((it, cond))

    meta = {"model": args.model, "revision": args.revision, "dtype": args.dtype,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "template_sha": hashlib.sha256(json.dumps([SYSTEM, TEMPLATES], sort_keys=True).encode()).hexdigest()[:12]}
    print("run meta:", meta, "| jobs:", len(jobs), "| already done:", len(done), flush=True)

    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        for b in range(0, len(jobs), args.batch):
            chunk = jobs[b:b + args.batch]
            letters = sorted(chunk[0][0]["options"])
            groups = {}
            for it, cond in chunk:  # group by option count so letters match
                groups.setdefault(tuple(sorted(it["options"])), []).append((it, cond))
            for letters, grp in groups.items():
                lid = letter_token_ids(tok, letters)
                prompts = [build_messages(tok, it, cond, ctx.get((it["qid"], cond))) for it, cond in grp]
                res = score_batch(model, tok, prompts, list(letters), lid)
                for (it, cond), r in zip(grp, res):
                    f.write(json.dumps({"qid": it["qid"], "dataset": it["dataset"], "split": it["split"],
                                        "condition": cond, "key": it["key"], "pred": r["pred"],
                                        "correct": r["pred"] == it["key"], "probs": r["probs"],
                                        "letter_mass": r["letter_mass"], **{"meta_" + k: v for k, v in meta.items()}}) + "\n")
            f.flush()
            if (b // args.batch) % 20 == 0:
                rate = (b + len(chunk)) / max(time.time() - t0, 1e-6)
                print(f"{b + len(chunk)}/{len(jobs)} done, {rate:.2f} items/s", flush=True)
    print("finished in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
