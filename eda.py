"""Exploratory data analysis for the Stripe type predictor dataset.

Prints summary statistics and generates 6 figures in figures/.

Usage:
    python eda.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = sns.color_palette("Set2")


def load_data():
    with open(DATA_DIR / "graph_snapshot.json") as f:
        graph = json.load(f)
    with open(DATA_DIR / "test_queries.json") as f:
        test = json.load(f)
    examples = []
    with open(DATA_DIR / "training_data.jsonl") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return graph, examples, test


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def print_stats(graph, train, test):
    n_types = len(graph["entity_types"])
    n_tools = len(graph["tools"])

    print("=" * 70)
    print("STRIPE DATASET OVERVIEW")
    print("=" * 70)
    print(f"Training examples:  {len(train)}")
    print(f"Test queries:       {len(test)}")
    print(f"Entity types:       {n_types}")
    print(f"Tools:              {n_tools}")

    train_pairs = Counter((ex["source_type"], ex["target_type"]) for ex in train)
    vals = sorted(train_pairs.values())
    print(f"\nTraining pairs (unique): {len(train_pairs)}")
    print(f"  Min examples/pair:  {vals[0]}")
    print(f"  Max examples/pair:  {vals[-1]}")
    print(f"  Median:             {vals[len(vals) // 2]}")
    print(f"  Mean:               {sum(vals) / len(vals):.1f}")

    src_counts = Counter(ex["source_type"] for ex in train)
    tgt_counts = Counter(ex["target_type"] for ex in train)

    print(f"\nSource type distribution (top 10):")
    for s, c in src_counts.most_common(10):
        print(f"  {s:<35} {c:>4} ({c / len(train) * 100:.1f}%)")

    print(f"\nTarget type distribution (top 10):")
    for t, c in tgt_counts.most_common(10):
        print(f"  {t:<35} {c:>4} ({c / len(train) * 100:.1f}%)")

    train_lens = [len(ex["query"].split()) for ex in train]
    test_lens = [len(q["query"].split()) for q in test]
    print(f"\nQuery length (words):")
    print(f"  Training: min={min(train_lens)}, max={max(train_lens)}, "
          f"mean={sum(train_lens) / len(train_lens):.1f}, "
          f"median={sorted(train_lens)[len(train_lens) // 2]}")
    print(f"  Test:     min={min(test_lens)}, max={max(test_lens)}, "
          f"mean={sum(test_lens) / len(test_lens):.1f}, "
          f"median={sorted(test_lens)[len(test_lens) // 2]}")

    test_cats = Counter(q["category"] for q in test)
    print(f"\nTest category breakdown:")
    for c in sorted(test_cats):
        print(f"  {c:<12} {test_cats[c]:>3}")

    test_pairs = set((q["source_type"], q["target_type"]) for q in test)
    novel = test_pairs - set(train_pairs.keys())
    print(f"\nTest pair coverage:")
    print(f"  Seen in training: {len(test_pairs) - len(novel)} ({(len(test_pairs) - len(novel)) / len(test_pairs) * 100:.0f}%)")
    print(f"  Novel (unseen):   {len(novel)} ({len(novel) / len(test_pairs) * 100:.0f}%)")
    for s, t in sorted(novel):
        print(f"    {s} -> {t}")

    train_src = set(src_counts.keys())
    train_tgt = set(tgt_counts.keys())
    test_src = set(q["source_type"] for q in test)
    test_tgt = set(q["target_type"] for q in test)
    all_types = set(graph["entity_types"].keys())
    print(f"\nEntity type coverage:")
    print(f"  Training src: {len(train_src)}/{len(all_types)}")
    print(f"  Training tgt: {len(train_tgt)}/{len(all_types)}")
    print(f"  Test src:     {len(test_src)}/{len(all_types)}")
    print(f"  Test tgt:     {len(test_tgt)}/{len(all_types)}")

    src_vals = sorted(src_counts.values(), reverse=True)
    print(f"\nClass imbalance (source types):")
    print(f"  Most frequent:  {src_vals[0]}")
    print(f"  Least frequent: {src_vals[-1]}")
    print(f"  Ratio:          {src_vals[0] / src_vals[-1]:.1f}x")

    train_tokens = [tok for ex in train for tok in tokenize(ex["query"])]
    test_tokens = [tok for q in test for tok in tokenize(q["query"])]
    train_vocab = set(train_tokens)
    test_vocab = set(test_tokens)
    oov = test_vocab - train_vocab
    print(f"\nLexical stats:")
    print(f"  Training vocab: {len(train_vocab)}")
    print(f"  Test vocab:     {len(test_vocab)}")
    print(f"  Test OOV:       {len(oov)} ({len(oov) / len(test_vocab) * 100:.1f}%)")

    actual_edges = len(set(
        (inp, out) for t in graph["tools"]
        for inp in t["input_types"] for out in t["output_types"]
    ))
    possible = n_types * n_types
    print(f"\nGraph density:")
    print(f"  Direct edges:    {actual_edges}")
    print(f"  Possible pairs:  {possible}")
    print(f"  Density:         {actual_edges / possible * 100:.2f}%")


def plot_training_pair_heatmap(train):
    train_pairs = Counter((ex["source_type"], ex["target_type"]) for ex in train)
    src_types = sorted(set(s for s, _ in train_pairs))
    tgt_types = sorted(set(t for _, t in train_pairs))

    matrix = np.zeros((len(src_types), len(tgt_types)))
    src_idx = {s: i for i, s in enumerate(src_types)}
    tgt_idx = {t: i for i, t in enumerate(tgt_types)}
    for (s, t), count in train_pairs.items():
        matrix[src_idx[s], tgt_idx[t]] = count

    fig, ax = plt.subplots(figsize=(16, 10))
    masked = np.ma.masked_where(matrix == 0, matrix)
    im = ax.pcolormesh(
        masked, cmap="YlOrRd",
        norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=matrix.max()),
    )
    ax.set_xticks(np.arange(len(tgt_types)) + 0.5)
    ax.set_xticklabels(tgt_types, rotation=90, fontsize=4)
    ax.set_yticks(np.arange(len(src_types)) + 0.5)
    ax.set_yticklabels(src_types, fontsize=4)
    ax.set_xlabel("Target type", fontsize=12)
    ax.set_ylabel("Source type", fontsize=12)
    fig.colorbar(im, ax=ax, label="Training examples (log scale)")
    fig.savefig(FIG_DIR / "training_pair_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_source_distribution(train):
    src_counts = Counter(ex["source_type"] for ex in train)
    top = src_counts.most_common(20)
    names, counts = zip(*reversed(top))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), counts, color=PALETTE[0])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Training examples", fontsize=12)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    fig.savefig(FIG_DIR / "source_type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(train):
    tgt_counts = Counter(ex["target_type"] for ex in train)
    top = tgt_counts.most_common(20)
    names, counts = zip(*reversed(top))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), counts, color=PALETTE[1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Training examples", fontsize=12)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8)
    fig.savefig(FIG_DIR / "target_type_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_query_length(train, test):
    train_lens = [len(ex["query"].split()) for ex in train]
    test_lens = [len(q["query"].split()) for q in test]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, max(max(train_lens), max(test_lens)) + 2)
    ax.hist(train_lens, bins=bins, alpha=0.6, label=f"Training (n={len(train)})",
            color=PALETTE[0], edgecolor="white")
    ax.hist(test_lens, bins=bins, alpha=0.7, label=f"Test (n={len(test)})",
            color=PALETTE[3], edgecolor="white")
    ax.set_xlabel("Query length (words)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=11)
    fig.savefig(FIG_DIR / "query_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_categories(test):
    cats = Counter(q["category"] for q in test)
    order = ["clean", "multihop", "synonym", "ambiguous", "noisy", "multipath"]
    names = [c for c in order if c in cats]
    counts = [cats[c] for c in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, counts, color=[PALETTE[i] for i in range(len(names))],
                  edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Test category", fontsize=12)
    ax.set_ylabel("Number of queries", fontsize=12)
    fig.savefig(FIG_DIR / "test_category_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_pair_support(train, test):
    train_pairs = Counter((ex["source_type"], ex["target_type"]) for ex in train)
    records = []
    for q in test:
        support = train_pairs.get((q["source_type"], q["target_type"]), 0)
        records.append({"category": q["category"], "support": support})

    order = ["clean", "multihop", "synonym", "ambiguous", "noisy", "multipath"]
    present = [c for c in order if any(r["category"] == c for r in records)]
    categories = [r["category"] for r in records]
    supports = [r["support"] for r in records]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.stripplot(x=categories, y=supports, order=present, hue=categories,
                  jitter=0.25, alpha=0.7, size=6, palette="Set2",
                  legend=False, ax=ax)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Test category", fontsize=12)
    ax.set_ylabel("Training examples for (src, tgt) pair", fontsize=12)
    fig.savefig(FIG_DIR / "test_pair_support.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    graph, train, test = load_data()

    print_stats(graph, train, test)

    print(f"\nGenerating figures...")
    plot_training_pair_heatmap(train)
    print(f"  saved training_pair_heatmap.png")
    plot_source_distribution(train)
    print(f"  saved source_type_distribution.png")
    plot_target_distribution(train)
    print(f"  saved target_type_distribution.png")
    plot_query_length(train, test)
    print(f"  saved query_length_distribution.png")
    plot_test_categories(test)
    print(f"  saved test_category_breakdown.png")
    plot_test_pair_support(train, test)
    print(f"  saved test_pair_support.png")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
