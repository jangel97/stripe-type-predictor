"""Evaluate a trained type predictor on held-out test queries.

Reports precision, recall, and F1 overall and per category.
Supports confidence threshold analysis to study the coverage vs quality tradeoff.

Usage:
    python evaluate.py
    python evaluate.py --model-dir model
    python evaluate.py --threshold-sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

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


def load_model(model_dir: Path):
    ckpt = torch.load(model_dir / "heads.pt", weights_only=True)
    encoder = SentenceTransformer(ckpt["base_model"], device=DEVICE)
    hidden_dim = ckpt.get("hidden_dim", 256)
    predictor = TypePredictor(ckpt["embedding_dim"], len(ckpt["source_types"]), hidden_dim)
    predictor.load_state_dict(ckpt["model"])
    predictor.eval().to(DEVICE)
    return encoder, predictor, ckpt["source_types"], ckpt


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


def predict_with_confidence(
    predictor: nn.Module,
    encoder: SentenceTransformer,
    query: str,
    type_names: list[str],
) -> tuple[str, str, float, float]:
    with torch.no_grad():
        emb = encoder.encode([query], convert_to_tensor=True, device=DEVICE)
        s_logits, t_logits = predictor(emb)
        s_probs = F.softmax(s_logits, dim=1)
        t_probs = F.softmax(t_logits, dim=1)
        src_conf = s_probs.max().item()
        tgt_conf = t_probs.max().item()
        pred_src = type_names[s_probs.argmax(1).item()]
        pred_tgt = type_names[t_probs.argmax(1).item()]
    return pred_src, pred_tgt, src_conf, tgt_conf


def evaluate(model_dir: str, threshold_sweep: bool = False):
    model_path = Path(__file__).resolve().parent / model_dir
    encoder, predictor, type_names, ckpt = load_model(model_path)

    with open(DATA_DIR / "graph_snapshot.json") as f:
        graph = json.load(f)
    with open(DATA_DIR / "test_queries.json") as f:
        test_queries = json.load(f)

    adj = build_adjacency(graph)

    print(f"Model: {model_path}")
    print(f"Types: {len(type_names)}, Test queries: {len(test_queries)}")
    if "cv_folds" in ckpt:
        print(f"CV: {ckpt['cv_folds']}-fold, mean F1={ckpt.get('cv_mean_f1', 0):.3f} "
              f"+/- {ckpt.get('cv_std_f1', 0):.3f}")
    print(f"Device: {DEVICE}")
    print()

    # ── Collect predictions with confidence ─────────────────────────
    results = []
    for q in test_queries:
        pred_src, pred_tgt, src_conf, tgt_conf = predict_with_confidence(
            predictor, encoder, q["query"], type_names,
        )
        tools = bfs_path(adj, pred_src, pred_tgt)

        expected = set(q["expected_tools"])
        resolved = set(tools) if tools else set()
        tp = len(expected & resolved)
        p = tp / len(resolved) if resolved else 0
        r = tp / len(expected) if expected else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        results.append({
            "id": q["id"],
            "category": q["category"],
            "expected_src": q["source_type"],
            "expected_tgt": q["target_type"],
            "pred_src": pred_src,
            "pred_tgt": pred_tgt,
            "src_conf": src_conf,
            "tgt_conf": tgt_conf,
            "confidence": min(src_conf, tgt_conf),
            "src_ok": pred_src == q["source_type"],
            "tgt_ok": pred_tgt == q["target_type"],
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    # ── Per-query results ───────────────────────────────────────────
    header = (f"{'ID':<35} {'Cat':<10} {'Expected':<40} {'Predicted':<40} "
              f"{'Conf':>5} {'P':>5} {'R':>5} {'F1':>5}")
    print(header)
    print("-" * len(header))

    for res in results:
        expected_st = f"{res['expected_src']}->{res['expected_tgt']}"
        predicted_st = f"{res['pred_src']}->{res['pred_tgt']}"
        mark = "ok" if (res["src_ok"] and res["tgt_ok"]) else "MISS"
        print(f"{res['id']:<35} {res['category']:<10} {expected_st:<40} {predicted_st:<40} "
              f"{res['confidence']:>5.2f} {res['precision']:>5.2f} {res['recall']:>5.2f} "
              f"{res['f1']:>5.2f}  {mark}")

    n = len(results)
    print("-" * len(header))

    source_correct = sum(1 for r in results if r["src_ok"])
    target_correct = sum(1 for r in results if r["tgt_ok"])
    type_exact = sum(1 for r in results if r["src_ok"] and r["tgt_ok"])

    print(f"\nType Prediction ({n} queries):")
    print(f"  Source accuracy:  {source_correct}/{n} ({source_correct/n:.0%})")
    print(f"  Target accuracy:  {target_correct}/{n} ({target_correct/n:.0%})")
    print(f"  Exact match:      {type_exact}/{n} ({type_exact/n:.0%})")

    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    print(f"\nTool Resolution:")
    print(f"  Precision:  {avg_p:.3f}")
    print(f"  Recall:     {avg_r:.3f}")
    print(f"  F1:         {avg_f1:.3f}")

    # ── Per-category ────────────────────────────────────────────────
    category_stats: dict[str, list[dict]] = {}
    for res in results:
        category_stats.setdefault(res["category"], []).append(res)

    print(f"\nPer-category:")
    print(f"  {'Category':<12} {'N':>3}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Conf':>6}")
    print(f"  {'-'*45}")
    for cat in sorted(category_stats.keys()):
        items = category_stats[cat]
        nc = len(items)
        print(f"  {cat:<12} {nc:>3}  {sum(r['precision'] for r in items)/nc:>6.3f}  "
              f"{sum(r['recall'] for r in items)/nc:>6.3f}  "
              f"{sum(r['f1'] for r in items)/nc:>6.3f}  "
              f"{sum(r['confidence'] for r in items)/nc:>6.3f}")

    # ── Baseline comparison ─────────────────────────────────────────
    print(f"\nBaseline comparison:")
    print(f"  Function calling:  P=0.66  R=0.93  F1=0.76")
    print(f"  Text baseline:     P=0.85  R=0.76  F1=0.79")
    print(f"  Zero-shot TCR:     P=0.82  R=0.69  F1=0.74")
    beat = "BEATS ALL" if avg_f1 > 0.79 else "beats FC+zero-shot" if avg_f1 > 0.76 else ""
    print(f"  Encoder TCR:       P={avg_p:.2f}  R={avg_r:.2f}  F1={avg_f1:.2f}  {beat}")

    # ── Confidence threshold sweep ──────────────────────────────────
    if threshold_sweep:
        print(f"\n{'='*60}")
        print("CONFIDENCE THRESHOLD ANALYSIS")
        print(f"{'='*60}")
        print(f"  Confidence = min(src_softmax_max, tgt_softmax_max)")
        print()
        print(f"  {'Threshold':>10} {'Coverage':>10} {'Answered':>10} "
              f"{'P':>7} {'R':>7} {'F1':>7}  {'ExactAcc':>10}")
        print(f"  {'-'*70}")

        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        for thresh in thresholds:
            above = [r for r in results if r["confidence"] >= thresh]
            coverage = len(above) / n
            if above:
                p = sum(r["precision"] for r in above) / len(above)
                r_ = sum(r["recall"] for r in above) / len(above)
                f1 = sum(r["f1"] for r in above) / len(above)
                exact = sum(1 for r in above if r["src_ok"] and r["tgt_ok"]) / len(above)
            else:
                p = r_ = f1 = exact = 0.0
            print(f"  {thresh:>10.2f} {coverage:>9.0%} {len(above):>8}/{n}  "
                  f"{p:>7.3f} {r_:>7.3f} {f1:>7.3f}  {exact:>9.0%}")

        # Confidence distribution
        print(f"\n  Confidence distribution:")
        confs = sorted(r["confidence"] for r in results)
        print(f"    Min:    {confs[0]:.3f}")
        print(f"    25th:   {confs[n//4]:.3f}")
        print(f"    Median: {confs[n//2]:.3f}")
        print(f"    75th:   {confs[3*n//4]:.3f}")
        print(f"    Max:    {confs[-1]:.3f}")

        # Show low-confidence predictions (potential abstentions)
        low_conf = [r for r in results if r["confidence"] < 0.5]
        if low_conf:
            print(f"\n  Low-confidence predictions (conf < 0.50):")
            for r in sorted(low_conf, key=lambda x: x["confidence"]):
                mark = "ok" if (r["src_ok"] and r["tgt_ok"]) else "MISS"
                print(f"    {r['confidence']:.3f}  {r['id']:<30} {r['category']:<10} {mark}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate type predictor")
    parser.add_argument("--model-dir", default="model", help="Directory with heads.pt")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Run confidence threshold analysis")
    args = parser.parse_args()
    evaluate(args.model_dir, threshold_sweep=args.threshold_sweep)


if __name__ == "__main__":
    main()
