"""Benchmark Gemini on Stripe: function calling vs zero-shot TCR vs encoder TCR.

Three strategies:
  1. function_calling — pass 536 tools, let model select directly
  2. text_baseline    — list 536 tools as text, ask for JSON array
  3. zero_shot_tcr    — list 163 entity types, ask model to predict source/target,
                        then resolve via BFS graph search

Requires: GEMINI_API_KEY environment variable (free tier from aistudio.google.com)

Usage:
    python benchmark_gemini.py
    python benchmark_gemini.py --model gemini-2.5-flash
    python benchmark_gemini.py --strategy zero_shot_tcr
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_RETRIES = 6


def load_data():
    with open(DATA_DIR / "graph_snapshot.json") as f:
        graph = json.load(f)
    with open(DATA_DIR / "test_queries.json") as f:
        test_queries = json.load(f)
    return graph, test_queries


def build_tool_schemas(tools: list[dict]) -> list[dict]:
    schemas = []
    for tool in tools:
        inp = ", ".join(tool["input_types"])
        out = ", ".join(tool["output_types"])
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": f"{tool['description']}. Input: {inp}. Output: {out}.",
                "parameters": {"type": "object", "properties": {}},
            },
        })
    return schemas


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
        for next_type, tool_name in sorted(adj.get(node, []), key=lambda x: x[1]):
            if next_type == tgt:
                return path + [tool_name]
            if next_type not in visited:
                visited.add(next_type)
                queue.append((next_type, path + [tool_name]))
    return None


# ── Strategy implementations ────────────────────────────────────────


def run_function_calling(client, model: str, query: str, tool_schemas: list[dict]) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a tool router. Select the tool(s) needed to answer the user's query. Call the appropriate tools."},
            {"role": "user", "content": query},
        ],
        tools=tool_schemas,
        tool_choice="auto",
        temperature=0,
    )
    tool_calls = response.choices[0].message.tool_calls or []
    return [tc.function.name for tc in tool_calls]


def run_text_baseline(client, model: str, query: str, tools: list[dict]) -> list[str]:
    tool_list = "\n".join(
        f"- {t['name']}: ({', '.join(t['input_types'])}) -> ({', '.join(t['output_types'])})"
        for t in tools
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": f"Available tools:\n{tool_list}\n\nSelect the tool(s) needed to answer the user's query. Respond ONLY with a JSON array of tool names, e.g. [\"ToolA\", \"ToolB\"]. No explanation."},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(t) for t in result]
    except json.JSONDecodeError:
        pass
    return []


def run_zero_shot_tcr(
    client, model: str, query: str,
    type_names: list[str], adj: dict,
) -> tuple[list[str], str, str]:
    type_list = ", ".join(type_names)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                f"You are an entity type predictor for a typed API graph. "
                f"The available entity types are:\n{type_list}\n\n"
                f"Given a natural language query, predict the SOURCE entity type "
                f"(where the query starts) and the TARGET entity type (what the query wants to retrieve). "
                f"Respond ONLY with JSON: {{\"source_type\": \"...\", \"target_type\": \"...\"}}\n"
                f"No explanation."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    text = response.choices[0].message.content.strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(text)
        pred_src = result.get("source_type", "")
        pred_tgt = result.get("target_type", "")
    except json.JSONDecodeError:
        return [], "", ""

    if pred_src not in type_names or pred_tgt not in type_names:
        return [], pred_src, pred_tgt

    path = bfs_path(adj, pred_src, pred_tgt)
    return path if path else [], pred_src, pred_tgt


# ── Evaluation ──────────────────────────────────────────────────────


def evaluate(predicted: list[str], expected: list[str]) -> tuple[float, float, float]:
    pred_set = set(predicted)
    exp_set = set(expected)
    tp = len(pred_set & exp_set)
    p = tp / len(pred_set) if pred_set else 0
    r = tp / len(exp_set) if exp_set else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


def run_strategy(
    strategy_name: str,
    run_fn,
    test_queries: list[dict],
    valid_tool_names: set[str],
    model: str,
    is_tcr: bool = False,
):
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")

    if is_tcr:
        header = f"{'ID':<35} {'Cat':<10} {'Exp Types':<40} {'Pred Types':<40} {'P':>5} {'R':>5} {'F1':>5}"
    else:
        header = f"{'ID':<35} {'Cat':<10} {'Expected':<45} {'Predicted':<45} {'P':>5} {'R':>5} {'F1':>5}"
    print(header)
    print("-" * len(header))

    results = []
    category_stats: dict[str, list[dict]] = {}
    type_correct = 0

    for q in test_queries:
        for attempt in range(MAX_RETRIES):
            try:
                if is_tcr:
                    predicted, pred_src, pred_tgt = run_fn(q["query"])
                else:
                    predicted = run_fn(q["query"])
                    pred_src, pred_tgt = "", ""
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES - 1:
                    wait = min(2 ** (attempt + 1), 60)
                    print(f"  Rate limited on {q['id']}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  Error on {q['id']}: {e}")
                    predicted = []
                    pred_src, pred_tgt = "", ""
                    break

        hallucinated = [t for t in predicted if t not in valid_tool_names]
        p, r, f1 = evaluate(predicted, q["expected_tools"])

        src_ok = pred_src == q["source_type"] if is_tcr else None
        tgt_ok = pred_tgt == q["target_type"] if is_tcr else None
        if is_tcr and src_ok and tgt_ok:
            type_correct += 1

        res = {
            "id": q["id"],
            "category": q["category"],
            "expected": q["expected_tools"],
            "predicted": predicted,
            "hallucinated": hallucinated,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
        if is_tcr:
            res.update({
                "pred_src": pred_src, "pred_tgt": pred_tgt,
                "exp_src": q["source_type"], "exp_tgt": q["target_type"],
                "src_ok": src_ok, "tgt_ok": tgt_ok,
            })
        results.append(res)
        category_stats.setdefault(q["category"], []).append(res)

        if is_tcr:
            exp_str = f"{q['source_type']}->{q['target_type']}"
            pred_str = f"{pred_src}->{pred_tgt}"
            mark = "ok" if (src_ok and tgt_ok) else "MISS"
            print(f"{q['id']:<35} {q['category']:<10} {exp_str:<40} {pred_str:<40} {p:>5.2f} {r:>5.2f} {f1:>5.2f}  {mark}")
        else:
            exp_str = ",".join(q["expected_tools"])[:44]
            pred_str = ",".join(predicted)[:44]
            mark = "ok" if f1 == 1.0 else ("HALL" if hallucinated else ("MISS" if f1 == 0 else "part"))
            print(f"{q['id']:<35} {q['category']:<10} {exp_str:<45} {pred_str:<45} {p:>5.2f} {r:>5.2f} {f1:>5.2f}  {mark}")

        time.sleep(0.5)

    n = len(results)
    print("-" * len(header))

    avg_p = sum(r["precision"] for r in results) / n
    avg_r = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    total_hall = sum(len(r["hallucinated"]) for r in results)

    print(f"\nOverall ({n} queries):")
    print(f"  Precision:      {avg_p:.3f}")
    print(f"  Recall:         {avg_r:.3f}")
    print(f"  F1:             {avg_f1:.3f}")
    if is_tcr:
        print(f"  Type exact:     {type_correct}/{n} ({type_correct/n:.0%})")
    else:
        print(f"  Hallucinated:   {total_hall} tool calls across {sum(1 for r in results if r['hallucinated'])} queries")

    print(f"\nPer-category:")
    print(f"  {'Category':<12} {'N':>3}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*38}")
    for cat in sorted(category_stats.keys()):
        items = category_stats[cat]
        nc = len(items)
        print(f"  {cat:<12} {nc:>3}  {sum(r['precision'] for r in items)/nc:>6.3f}  "
              f"{sum(r['recall'] for r in items)/nc:>6.3f}  "
              f"{sum(r['f1'] for r in items)/nc:>6.3f}")

    print(f"\nComparison:")
    print(f"  Function calling (Haiku):  P=0.66  R=0.93  F1=0.76")
    print(f"  Text baseline (Haiku):     P=0.85  R=0.76  F1=0.79")
    print(f"  Zero-shot TCR (Haiku):     P=0.82  R=0.69  F1=0.74")
    print(f"  Encoder TCR (5-fold CV):   P=0.91  R=0.91  F1=0.91")
    beat_encoder = "BEATS ENCODER" if avg_f1 > 0.91 else ""
    print(f"  {strategy_name} ({model}): P={avg_p:.2f}  R={avg_r:.2f}  F1={avg_f1:.2f}  {beat_encoder}")

    out_path = Path(__file__).resolve().parent / f"results_{strategy_name}_{model.replace('/', '_')}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model,
            "strategy": strategy_name,
            "n_tools": len(valid_tool_names),
            "n_queries": n,
            "precision": avg_p,
            "recall": avg_r,
            "f1": avg_f1,
            "hallucinated_total": total_hall,
            "per_query": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Gemini on Stripe tool selection")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--strategy",
                        choices=["function_calling", "text", "zero_shot_tcr", "all"],
                        default="all")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    graph, test_queries = load_data()
    tools = graph["tools"]
    tool_schemas = build_tool_schemas(tools)
    valid_tool_names = {t["name"] for t in tools}
    type_names = sorted(graph["entity_types"].keys())
    adj = build_adjacency(graph)

    print(f"Model: {args.model}")
    print(f"Tools: {len(tools)}, Entity types: {len(type_names)}, Test queries: {len(test_queries)}")

    if args.strategy in ("function_calling", "all"):
        run_strategy(
            "function_calling",
            lambda q: run_function_calling(client, args.model, q, tool_schemas),
            test_queries, valid_tool_names, args.model,
        )

    if args.strategy in ("text", "all"):
        run_strategy(
            "text_baseline",
            lambda q: run_text_baseline(client, args.model, q, tools),
            test_queries, valid_tool_names, args.model,
        )

    if args.strategy in ("zero_shot_tcr", "all"):
        run_strategy(
            "zero_shot_tcr",
            lambda q: run_zero_shot_tcr(client, args.model, q, type_names, adj),
            test_queries, valid_tool_names, args.model,
            is_tcr=True,
        )


if __name__ == "__main__":
    main()
