"""Dense retrieval, fine-tuning, hybrid fusion and reranking on a Kaggle GPU.

Runs every neural system on the same dev / test questions as the CPU BM25 baseline and writes
their top-100 rankings, so that all metrics are computed by the same code locally. Each stage
writes its outputs as soon as it finishes; a failure in a late stage keeps the earlier ones.

Stages
1. BM25 (the project's own implementation) for hard negatives and hybrid fusion.
2. Zero-shot bi-encoders: multilingual-e5-small, multilingual-e5-base, BGE-M3.
   Long articles are split into overlapping passages; an article scores as its best passage
   (MaxP, Dai & Callan 2019).
3. Fine-tuning multilingual-e5-base with an InfoNCE loss: in-batch negatives plus one BM25 hard
   negative per question; the epoch is chosen on dev.
4. Hybrid: reciprocal rank fusion of BM25 and the best bi-encoder.
5. Reranking the hybrid top-50 with a cross-encoder (bge-reranker-v2-m3).
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SMOKE = int(os.environ.get("SMOKE", 0))  # >0: use only this many documents (pipeline test)
OUT = Path("/kaggle/working")
t_start = time.time()


def log(*args):
    print(f"[{time.time() - t_start:7.0f}s]", *args, flush=True)


ROOT = Path(glob.glob("/kaggle/input/**/splits.json", recursive=True)[0]).parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DATA_DIR", str(OUT / "data"))

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

from vilegal import data, metrics  # noqa: E402
from vilegal.bm25 import BM25  # noqa: E402
from vilegal.fusion import reciprocal_rank_fusion  # noqa: E402
from vilegal.text import syllables  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
random.seed(0)
log("device", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

# --- data (raw files shipped in the dataset, same splits as the CPU experiments) ----------------
raw = OUT / "data" / "raw"
raw.mkdir(parents=True, exist_ok=True)
for name in data.FILES:
    src = next(Path(p) for p in glob.glob(str(ROOT / "**" / name), recursive=True))
    (raw / name).write_bytes(src.read_bytes())
ds = data.load(OUT / "data")
splits = json.loads((ROOT / "splits.json").read_text())
for split in ("train", "dev", "test"):
    assert sorted(ds.qrels[split]) == splits[split], f"{split} split differs from the CPU run"
if SMOKE:
    keep = {d for split in ds.qrels.values() for g in split.values() for d in g}
    keep |= set(ds.doc_ids[:SMOKE])
    sel = [i for i, d in enumerate(ds.doc_ids) if d in keep]
    ds.doc_ids = [ds.doc_ids[i] for i in sel]
    ds.doc_texts = [ds.doc_texts[i] for i in sel]
    for split in ds.qrels:
        ds.qrels[split] = dict(list(ds.qrels[split].items())[:100])
log(
    f"{len(ds.doc_ids):,} documents; questions:",
    {k: len(v) for k, v in ds.qrels.items()},
)
doc_index = {d: i for i, d in enumerate(ds.doc_ids)}
runs: dict[str, dict[str, dict[str, list[str]]]] = {}  # system -> split -> qid -> ranked ids
timings: dict[str, float] = {}


def save_run(name: str, split: str, qids, ranked_idx) -> None:
    runs.setdefault(name, {})[split] = {
        q: [ds.doc_ids[i] for i in row] for q, row in zip(qids, ranked_idx, strict=True)
    }
    (OUT / "runs.json").write_text(json.dumps(runs))
    gold = [ds.qrels[split][q] for q in qids]
    m = metrics.evaluate(list(runs[name][split].values()), gold)
    log(
        f"{name:28s} {split:4s} mrr@10 {m['mrr@10'].mean():.4f} r@10 {m['recall@10'].mean():.4f}"
        f" r@100 {m['recall@100'].mean():.4f}"
    )


# --- 1. BM25 ------------------------------------------------------------------------------------
t0 = time.time()
bm25 = BM25(k1=0.9, b=0.75).fit(ds.doc_texts)
timings["bm25_index"] = time.time() - t0
bm25_top = {}
for split in ("train", "dev", "test"):
    qids, qtexts = ds.split_queries(split)
    idx, _ = bm25.search(qtexts, k=100)
    bm25_top[split] = idx
    if split != "train":
        save_run("BM25", split, qids, idx)

# --- passages for MaxP ---------------------------------------------------------------------------
PASSAGE, STRIDE = 220, 180  # syllables


def passages(text: str) -> list[str]:
    title, _, body = text.partition("\n")
    words = body.split()
    if len(words) <= PASSAGE:
        return [text]
    return [
        title + "\n" + " ".join(words[s : s + PASSAGE])
        for s in range(0, max(1, len(words) - PASSAGE + STRIDE), STRIDE)
    ]


passage_text, passage_doc = [], []
for i, t in enumerate(ds.doc_texts):
    for p in passages(t):
        passage_text.append(p)
        passage_doc.append(i)
passage_doc = np.array(passage_doc)
log(f"{len(passage_text):,} passages (MaxP)")

PREFIX = {"intfloat": ("query: ", "passage: ")}


def prefixes(name: str):
    return PREFIX.get(name.split("/")[0], ("", ""))


def _amp():
    return torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda")


@torch.no_grad()
def encode_corpus(model, name) -> torch.Tensor:
    _, p = prefixes(name)
    with _amp():
        emb = model.encode(
            [p + t for t in passage_text],
            batch_size=128,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return emb.half()


@torch.no_grad()
def search(model, name, corpus_emb, qtexts, k=100) -> np.ndarray:
    q, _ = prefixes(name)
    with _amp():
        qe = model.encode(
            [q + t for t in qtexts],
            batch_size=128,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).half()
    pd_t = torch.from_numpy(passage_doc).to(qe.device)
    out = []
    for s in range(0, len(qe), 64):
        sc = (qe[s : s + 64] @ corpus_emb.T).float()  # (q, passages)
        doc_sc = torch.full((sc.shape[0], len(ds.doc_ids)), -1e9, device=sc.device)
        doc_sc.scatter_reduce_(1, pd_t.expand(sc.shape[0], -1), sc, reduce="amax")
        out.append(doc_sc.topk(k, dim=1).indices.cpu().numpy())
    return np.vstack(out)


def evaluate_model(model, name, label) -> float:
    t0 = time.time()
    emb = encode_corpus(model, name)
    timings[f"{label}_encode"] = time.time() - t0
    dev_mrr = 0.0
    for split in ("dev", "test"):
        qids, qtexts = ds.split_queries(split)
        idx = search(model, name, emb, qtexts)
        save_run(label, split, qids, idx)
        if split == "dev":
            gold = [ds.qrels["dev"][q] for q in qids]
            dev_mrr = metrics.evaluate([[ds.doc_ids[i] for i in r] for r in idx], gold)[
                "mrr@10"
            ].mean()
    del emb
    torch.cuda.empty_cache()
    return float(dev_mrr)


# --- 2. zero-shot bi-encoders -------------------------------------------------------------------
ZERO_SHOT = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "BAAI/bge-m3",
]
for name in ZERO_SHOT:
    try:
        model = SentenceTransformer(name, device=device)
        model.max_seq_length = 512
        if device == "cuda":
            model.half()
        evaluate_model(model, name, name.split("/")[-1] + " (zero-shot)")
        del model
        torch.cuda.empty_cache()
    except Exception as exc:  # keep going: one model failing should not lose the others
        log("FAILED", name, repr(exc))

# --- 3. fine-tuning multilingual-e5-base --------------------------------------------------------
BASE = "intfloat/multilingual-e5-base"
qprefix, pprefix = prefixes(BASE)
train_qids, train_q = ds.split_queries("train")
rng = random.Random(0)
examples = []
for qi, (q, text) in enumerate(zip(train_qids, train_q, strict=True)):
    gold = ds.qrels["train"][q]
    negs = [i for i in bm25_top["train"][qi][:30] if ds.doc_ids[i] not in gold]
    for d in gold:
        if d in doc_index:
            examples.append((text, doc_index[d], negs))
log(f"fine-tuning examples: {len(examples)}")


def first_passage(i: int) -> str:
    return passages(ds.doc_texts[i])[0]


model = SentenceTransformer(BASE, device=device)
model.max_seq_length = 256
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
EPOCHS, BATCH, TAU = 3, 32, 0.05
steps_total = EPOCHS * ((len(examples) + BATCH - 1) // BATCH)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * max(0.0, 1 - s / steps_total),
)
scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")


def embed(m, texts):
    # sentence-transformers 5 renamed tokenize() to preprocess() and may return non-tensor
    # entries alongside the tensors; only tensors go to the GPU.
    feats = (m.preprocess if hasattr(m, "preprocess") else m.tokenize)(texts)
    feats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in feats.items()}
    return F.normalize(m(feats)["sentence_embedding"], dim=-1)


best_dev, best_epoch, history = -1.0, 0, []
ft_dir = OUT / "e5-base-finetuned"
for epoch in range(EPOCHS):
    model.train()
    rng.shuffle(examples)
    t0, total = time.time(), 0.0
    for s in range(0, len(examples), BATCH):
        batch = examples[s : s + BATCH]
        qs = [qprefix + q for q, _, _ in batch]
        pos = [pprefix + first_passage(d) for _, d, _ in batch]
        neg = [
            pprefix + first_passage(rng.choice(n)) if n else pprefix + first_passage(d)
            for _, d, n in batch
        ]
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            qe, de = embed(model, qs), embed(model, pos + neg)
            logits = qe @ de.T / TAU  # (B, 2B): own positive at column i, the rest negatives
            loss = F.cross_entropy(logits.float(), torch.arange(len(batch), device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += float(loss.detach()) * len(batch)
    model.eval()
    model.max_seq_length = 512
    dev = evaluate_model(model, BASE, f"e5-base fine-tuned, epoch {epoch + 1}")
    model.max_seq_length = 256
    history.append(
        {
            "epoch": epoch + 1,
            "loss": total / len(examples),
            "dev_mrr@10": dev,
            "seconds": time.time() - t0,
        }
    )
    log(f"epoch {epoch + 1}: loss {total / len(examples):.4f}, dev mrr@10 {dev:.4f}")
    if dev > best_dev:
        best_dev, best_epoch = dev, epoch + 1
        model.save(str(ft_dir))
timings["finetune_history"] = history
log(f"best epoch {best_epoch} (dev mrr@10 {best_dev:.4f})")
best_ft = f"e5-base fine-tuned, epoch {best_epoch}"
runs["e5-base fine-tuned"] = runs[best_ft]
(OUT / "runs.json").write_text(json.dumps(runs))
del model
torch.cuda.empty_cache()

# --- 4. hybrid: reciprocal rank fusion ----------------------------------------------------------
for split in ("dev", "test"):
    qids, _ = ds.split_queries(split)
    fused = []
    for q in qids:
        ranked = reciprocal_rank_fusion(
            [runs["BM25"][split][q], runs["e5-base fine-tuned"][split][q]], k=60
        )[:100]
        fused.append([doc_index[d] for d in ranked])
    save_run("Hybrid (BM25 + fine-tuned e5, RRF)", split, qids, fused)

# --- 5. cross-encoder reranking of the hybrid top-50 --------------------------------------------
try:
    t0 = time.time()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device=device)
    if device == "cuda":
        reranker.model.half()
    for split in ("dev", "test"):
        qids, qtexts = ds.split_queries(split)
        hybrid = runs["Hybrid (BM25 + fine-tuned e5, RRF)"][split]
        reranked = []
        for q, text in zip(qids, qtexts, strict=True):
            cands = hybrid[q][:50]
            pairs = [(text, ds.doc_texts[doc_index[d]][:2000]) for d in cands]
            sc = reranker.predict(pairs, batch_size=50, show_progress_bar=False)
            order = np.argsort(-np.asarray(sc), kind="stable")
            ranked = [cands[i] for i in order] + hybrid[q][50:]
            reranked.append([doc_index[d] for d in ranked])
        save_run("Hybrid + bge-reranker-v2-m3", split, qids, reranked)
    timings["rerank"] = time.time() - t0
except Exception as exc:
    log("FAILED reranker", repr(exc))

(OUT / "timings.json").write_text(json.dumps(timings, indent=2))
log(
    "query-length check:",
    int(np.median([len(syllables(t)) for t in ds.queries.values()])),
)
log("done")
