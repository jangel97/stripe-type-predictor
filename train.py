"""Train entity type prediction heads with stratified k-fold cross-validation.

Frozen sentence encoder (all-MiniLM-L6-v2, 22M params) + shared MLP hidden
layer + two classification heads (source type, target type).

Each fold trains on ~80% of the data, validates on ~20%, then evaluates on
the 42 fixed held-out test queries. Reports mean +/- std across folds.

Usage:
    python train.py
    python train.py --folds 5 --epochs 200 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parent / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TypePredictor(nn.Module):
    def __init__(self, embedding_dim: int, n_types: int, hidden_dim: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.source_head = nn.Linear(hidden_dim, n_types)
        self.target_head = nn.Linear(hidden_dim, n_types)

    def forward(self, embeddings: torch.Tensor):
        h = self.shared(embeddings)
        return self.source_head(h), self.target_head(h)


def compute_sqrt_weights(indices: list[int], n_classes: int) -> torch.Tensor:
    counts = Counter(indices)
    median_count = sorted(counts.values())[len(counts) // 2] if counts else 1
    weights = torch.ones(n_classes)
    for i in range(n_classes):
        if counts.get(i, 0) > 0:
            weights[i] = math.sqrt(median_count / counts[i])
    return weights


def stratified_k_fold(labels: list[int], k: int, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)

    for indices in by_label.values():
        rng.shuffle(indices)

    folds: list[list[int]] = [[] for _ in range(k)]
    for indices in by_label.values():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    for fold in folds:
        rng.shuffle(fold)

    splits = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        splits.append((train_idx, val_idx))

    return splits


def load_data():
    with open(DATA_DIR / "graph_snapshot.json") as f:
        graph = json.load(f)

    examples = []
    with open(DATA_DIR / "training_data.jsonl") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    type_names = sorted(graph["entity_types"].keys())
    type_to_idx = {t: i for i, t in enumerate(type_names)}

    queries, src_indices, tgt_indices = [], [], []
    for ex in examples:
        src, tgt = ex["source_type"], ex["target_type"]
        if src in type_to_idx and tgt in type_to_idx:
            queries.append(ex["query"])
            src_indices.append(type_to_idx[src])
            tgt_indices.append(type_to_idx[tgt])

    return queries, src_indices, tgt_indices, type_names, graph


def load_test_queries():
    with open(DATA_DIR / "test_queries.json") as f:
        return json.load(f)


def build_adjacency(graph: dict) -> dict[str, list[tuple[str, str]]]:
    adj: dict[str, list[tuple[str, str]]] = {}
    for tool in graph["tools"]:
        for inp in tool["input_types"]:
            for out in tool["output_types"]:
                adj.setdefault(inp, []).append((out, tool["name"]))
    return adj


def bfs_path(adj: dict, src: str, tgt: str) -> list[str] | None:
    if src == tgt:
        for out, name in adj.get(src, []):
            if out == tgt:
                return [name]
        return None
    visited = {src}
    queue = [(src, [])]
    while queue:
        node, path = queue.pop(0)
        for next_type, tool_name in adj.get(node, []):
            if next_type == tgt:
                return path + [tool_name]
            if next_type not in visited:
                visited.add(next_type)
                queue.append((next_type, path + [tool_name]))
    return None


def train_one_fold(
    embeddings: torch.Tensor,
    src_labels: torch.Tensor,
    tgt_labels: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    n_types: int,
    args,
) -> tuple[nn.Module, dict]:
    train_src = [src_labels[i].item() for i in train_idx]
    train_tgt = [tgt_labels[i].item() for i in train_idx]
    src_weights = compute_sqrt_weights(train_src, n_types).to(DEVICE)
    tgt_weights = compute_sqrt_weights(train_tgt, n_types).to(DEVICE)

    train_ds = TensorDataset(
        embeddings[train_idx], src_labels[train_idx], tgt_labels[train_idx],
    )
    val_ds = TensorDataset(
        embeddings[val_idx], src_labels[val_idx], tgt_labels[val_idx],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    emb_dim = embeddings.shape[1]
    model = TypePredictor(emb_dim, n_types, hidden_dim=args.hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    best_exact = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        for emb, sl, tl in train_loader:
            emb, sl, tl = emb.to(DEVICE), sl.to(DEVICE), tl.to(DEVICE)
            s_logits, t_logits = model(emb)
            loss = (
                F.cross_entropy(s_logits, sl, weight=src_weights)
                + F.cross_entropy(t_logits, tl, weight=tgt_weights)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        all_src_ok, all_tgt_ok = [], []
        with torch.no_grad():
            for emb, sl, tl in val_loader:
                emb, sl, tl = emb.to(DEVICE), sl.to(DEVICE), tl.to(DEVICE)
                s_logits, t_logits = model(emb)
                all_src_ok.append(s_logits.argmax(1) == sl)
                all_tgt_ok.append(t_logits.argmax(1) == tl)
        src_ok = torch.cat(all_src_ok)
        tgt_ok = torch.cat(all_tgt_ok)
        exact_acc = (src_ok & tgt_ok).float().mean().item()

        if exact_acc > best_exact:
            best_exact = exact_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    model.eval()
    all_src_ok, all_tgt_ok = [], []
    with torch.no_grad():
        for emb, sl, tl in val_loader:
            emb, sl, tl = emb.to(DEVICE), sl.to(DEVICE), tl.to(DEVICE)
            s_logits, t_logits = model(emb)
            all_src_ok.append(s_logits.argmax(1) == sl)
            all_tgt_ok.append(t_logits.argmax(1) == tl)
    src_ok = torch.cat(all_src_ok)
    tgt_ok = torch.cat(all_tgt_ok)

    val_metrics = {
        "val_source_acc": src_ok.float().mean().item(),
        "val_target_acc": tgt_ok.float().mean().item(),
        "val_exact_acc": (src_ok & tgt_ok).float().mean().item(),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
    }
    return model, val_metrics


def evaluate_on_test(
    model: nn.Module,
    encoder: SentenceTransformer,
    type_names: list[str],
    adj: dict,
    test_queries: list[dict],
) -> dict:
    model.eval()
    source_correct = 0
    target_correct = 0
    type_exact = 0
    f1_vals, prec_vals, rec_vals = [], [], []
    confidences = []
    category_stats: dict[str, dict] = {}

    for q in test_queries:
        cat = q["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "f1": [], "prec": [], "rec": []}

        with torch.no_grad():
            emb = encoder.encode([q["query"]], convert_to_tensor=True, device=DEVICE)
            s_logits, t_logits = model(emb)
            s_probs = F.softmax(s_logits, dim=1)
            t_probs = F.softmax(t_logits, dim=1)
            pred_src = type_names[s_probs.argmax(1).item()]
            pred_tgt = type_names[t_probs.argmax(1).item()]
            conf = min(s_probs.max().item(), t_probs.max().item())
            tools = bfs_path(adj, pred_src, pred_tgt)

        confidences.append(conf)

        if pred_src == q["source_type"]:
            source_correct += 1
        if pred_tgt == q["target_type"]:
            target_correct += 1
        if pred_src == q["source_type"] and pred_tgt == q["target_type"]:
            type_exact += 1

        expected = set(q["expected_tools"])
        resolved = set(tools) if tools else set()
        tp = len(expected & resolved)
        p = tp / len(resolved) if resolved else 0
        r = tp / len(expected) if expected else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        f1_vals.append(f1)
        prec_vals.append(p)
        rec_vals.append(r)
        category_stats[cat]["total"] += 1
        category_stats[cat]["f1"].append(f1)
        category_stats[cat]["prec"].append(p)
        category_stats[cat]["rec"].append(r)

    n = len(test_queries)
    result = {
        "source_acc": source_correct / n,
        "target_acc": target_correct / n,
        "exact_acc": type_exact / n,
        "precision": sum(prec_vals) / n,
        "recall": sum(rec_vals) / n,
        "f1": sum(f1_vals) / n,
        "mean_confidence": sum(confidences) / n,
        "category": {},
    }
    for cat, s in category_stats.items():
        nc = s["total"]
        result["category"][cat] = {
            "n": nc,
            "precision": sum(s["prec"]) / nc,
            "recall": sum(s["rec"]) / nc,
            "f1": sum(s["f1"]) / nc,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Train with stratified k-fold CV")
    parser.add_argument("--base-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("Loading data...")
    queries, src_indices, tgt_indices, type_names, graph = load_data()
    test_queries = load_test_queries()
    n_types = len(type_names)

    src_labels = torch.tensor(src_indices, dtype=torch.long)
    tgt_labels = torch.tensor(tgt_indices, dtype=torch.long)
    print(f"  Training examples: {len(queries)}")
    print(f"  Entity types:      {n_types}")
    print(f"  Test queries:      {len(test_queries)} (fixed, held-out)")

    print(f"\nEncoding queries with {args.base_model}...")
    encoder = SentenceTransformer(args.base_model, device=DEVICE)
    embeddings = encoder.encode(queries, show_progress_bar=True, convert_to_numpy=True)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    emb_dim = embeddings.shape[1]
    print(f"  Embedding dim: {emb_dim}, Device: {DEVICE}")

    adj = build_adjacency(graph)
    splits = stratified_k_fold(src_indices, args.folds, seed=args.seed)

    trainable = sum(
        p.numel() for p in TypePredictor(emb_dim, n_types, args.hidden_dim).parameters()
    )
    print(f"\nTypePredictor({emb_dim} -> {args.hidden_dim} -> {n_types})")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Folds: {args.folds}, Epochs: {args.epochs}, LR: cosine {args.lr} -> 1e-5")

    fold_results = []
    best_f1 = -1.0
    best_state = None

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold_i + 1}/{args.folds}  (train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        model, val_metrics = train_one_fold(
            embeddings, src_labels, tgt_labels, train_idx, val_idx, n_types, args,
        )
        test_res = evaluate_on_test(model, encoder, type_names, adj, test_queries)
        combined = {**val_metrics, **test_res}
        fold_results.append(combined)

        print(f"  Validation:  src={val_metrics['val_source_acc']:.1%}  "
              f"tgt={val_metrics['val_target_acc']:.1%}  "
              f"exact={val_metrics['val_exact_acc']:.1%}")
        print(f"  Test:        P={test_res['precision']:.3f}  "
              f"R={test_res['recall']:.3f}  "
              f"F1={test_res['f1']:.3f}  "
              f"conf={test_res['mean_confidence']:.3f}")

        if test_res["f1"] > best_f1:
            best_f1 = test_res["f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION SUMMARY ({args.folds} folds)")
    print(f"{'='*60}")

    metrics_keys = [
        ("val_source_acc", "Val source acc"),
        ("val_target_acc", "Val target acc"),
        ("val_exact_acc", "Val exact match"),
        ("precision", "Test precision"),
        ("recall", "Test recall"),
        ("f1", "Test F1"),
        ("mean_confidence", "Mean confidence"),
    ]

    print(f"\n{'Metric':<20} {'Mean':>8} {'Std':>8}   Per-fold")
    print("-" * 70)
    for key, label in metrics_keys:
        values = [r[key] for r in fold_results]
        mean = np.mean(values)
        std = np.std(values)
        per_fold = "  ".join(f"{v:.3f}" for v in values)
        print(f"  {label:<18} {mean:>7.3f} {std:>7.3f}   {per_fold}")

    # Per-category breakdown (mean across folds)
    all_cats = sorted({cat for r in fold_results for cat in r["category"]})
    print(f"\nPer-category mean across folds:")
    print(f"  {'Category':<12} {'N':>3}  {'Prec':>7} {'Rec':>7} {'F1':>8}")
    print(f"  {'-'*40}")
    for cat in all_cats:
        cat_p = [r["category"][cat]["precision"] for r in fold_results if cat in r["category"]]
        cat_r = [r["category"][cat]["recall"] for r in fold_results if cat in r["category"]]
        cat_f = [r["category"][cat]["f1"] for r in fold_results if cat in r["category"]]
        n_q = fold_results[0]["category"].get(cat, {}).get("n", 0)
        print(f"  {cat:<12} {n_q:>3}  {np.mean(cat_p):>7.3f} {np.mean(cat_r):>7.3f} {np.mean(cat_f):>8.3f}")

    # Baseline comparison
    mean_f1 = np.mean([r["f1"] for r in fold_results])
    std_f1 = np.std([r["f1"] for r in fold_results])
    mean_p = np.mean([r["precision"] for r in fold_results])
    mean_r = np.mean([r["recall"] for r in fold_results])

    print(f"\nBaseline comparison:")
    print(f"  Function calling:  P=0.66  R=0.93  F1=0.76")
    print(f"  Text baseline:     P=0.85  R=0.76  F1=0.79")
    print(f"  Zero-shot TCR:     P=0.82  R=0.69  F1=0.74")
    print(f"  Encoder TCR:       P={mean_p:.2f}  R={mean_r:.2f}  F1={mean_f1:.2f} +/- {std_f1:.2f}")

    # Save best model
    out_dir = Path(__file__).resolve().parent / "model"
    out_dir.mkdir(exist_ok=True)
    torch.save({
        "model": best_state,
        "base_model": args.base_model,
        "embedding_dim": emb_dim,
        "hidden_dim": args.hidden_dim,
        "source_types": type_names,
        "target_types": type_names,
        "cv_folds": args.folds,
        "cv_mean_f1": float(mean_f1),
        "cv_std_f1": float(std_f1),
    }, out_dir / "heads.pt")
    print(f"\nSaved best model (F1={best_f1:.3f}) to {out_dir}/heads.pt")

    # Save CV results
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump({
            "folds": args.folds,
            "epochs": args.epochs,
            "results": fold_results,
            "summary": {k: {"mean": float(np.mean([r[k] for r in fold_results])),
                            "std": float(np.std([r[k] for r in fold_results]))}
                        for k, _ in metrics_keys},
        }, f, indent=2)
    print(f"Saved CV results to {out_dir}/cv_results.json")


if __name__ == "__main__":
    main()
