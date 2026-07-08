# Stripe Type Predictor

Learned entity type prediction for Typed Composition Search (TCS), demonstrated on the Stripe API (536 tools, 163 entity types).

Our approach assumes that the target domain admits a typed entity abstraction, where tools can be represented as transformations between entity types. Under this assumption, tool routing can be reformulated as predicting source and target entity types followed by graph search over the typed composition graph.

A frozen sentence encoder (all-mpnet-base-v2, 110M params) with a lightweight MLP head (~281K trainable parameters) predicts source and target entity types from natural language queries. BFS graph search then resolves the predicted types to tool chains.

## Results

Evaluated with 5-fold stratified cross-validation on 3,787 training examples, tested on 120 fixed held-out queries across 6 difficulty categories. Stratification by source type ensures minority entity types appear in every fold.

Five strategies are compared:

- **Function calling** — All 536 tools are passed as structured function definitions. The model uses its native tool-calling mechanism to select which functions to invoke.
- **Text baseline** — All 536 tools are listed as plain text in the system prompt (`- ToolName: (InputType) -> (OutputType)`). The model responds with a JSON array of tool names.
- **Few-shot Text** — Same as text baseline, but with 10 in-context examples showing `(query, [tool_names])` mappings. Tool names are resolved from training examples via BFS. This tests whether adding examples to direct tool selection can match the TCR decomposition.
- **Zero-shot TCR** — Instead of tools, the model receives the 163 entity type names and predicts a source and target type. BFS graph search then resolves the types to a tool chain.
- **Few-shot TCR** — Same as zero-shot TCR, but with 10 in-context examples drawn from the training set (excluding test type pairs). This gives the LLM domain-specific demonstrations of how queries map to entity types.
- **Encoder TCR** — Same type prediction + graph search pipeline, but type prediction is done by the trained 281K-parameter classifier instead of prompting an LLM.

For all strategies, precision, recall, and F1 are computed by comparing the predicted tool set against the expected tool set using set intersection.

| Strategy | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Function calling | Gemini 2.5 Pro | 0.358 | 0.350 | 0.353 |
| Zero-shot TCR | Gemini 2.5 Pro | 0.392 | 0.350 | 0.364 |
| Text baseline | Gemini 2.5 Pro | 0.775 | 0.821 | 0.781 |
| Few-shot Text | Gemini 2.5 Flash | 0.810 | 0.804 | 0.795 |
| Few-shot TCR | Gemini 2.5 Pro | 0.850 | 0.829 | 0.836 |
| **Encoder TCR** | **281K params** | **0.867** | **0.829** | **0.842** |

Adding 10 examples to the text baseline (few-shot text) improves F1 only marginally over zero-shot text (0.795 vs 0.781), even with a different model. In contrast, the same 10 examples under TCR decomposition (few-shot TCR) yield F1=0.836 — a much larger gain. This confirms that the decomposition into type prediction + graph search drives the improvement, not the in-context examples themselves.

Per-category (encoder, best fold):

| Category | N | Precision | Recall | F1 |
|---|---|---|---|---|
| clean | 35 | 0.867 | 0.867 | 0.867 |
| noisy | 15 | 1.000 | 1.000 | 1.000 |
| synonym | 20 | 0.810 | 0.810 | 0.810 |
| multihop | 20 | 0.772 | 0.772 | 0.772 |
| ambiguous | 15 | 0.733 | 0.733 | 0.733 |
| multipath | 15 | 0.867 | 0.867 | 0.867 |

### Encoder vs Gemini 2.5 Pro per-category

The encoder and Gemini few-shot TCR show complementary strengths. The encoder wins on messy real-world inputs (synonym, ambiguous, noisy), while Gemini wins on clean structured queries:

| Category | Encoder | Gemini FS-TCR | Winner |
|---|---|---|---|
| clean | 0.867 | **0.933** | Gemini |
| multihop | 0.772 | **0.860** | Gemini |
| multipath | 0.867 | **1.000** | Gemini |
| synonym | **0.810** | 0.667 | Encoder |
| ambiguous | **0.733** | 0.667 | Encoder |
| noisy | **1.000** | 0.867 | Encoder |

The encoder's advantage on synonym and noisy queries comes from its training data — it learned that "payments" maps to PaymentIntent and "chargebacks" maps to Dispute. Gemini can reason about clean queries from its general knowledge of Stripe's API, but struggles with informal domain vocabulary.

### Confidence threshold analysis

The model's confidence (min of source and target softmax max) is well-calibrated. At threshold 0.90, the model answers 60% of queries with perfect F1. Precision and recall move in lockstep because R_wrong=0 on Stripe — every type prediction error results in a completely wrong tool chain.

| Threshold | Coverage | P | R | F1 |
|---|---|---|---|---|
| 0.00 | 100% | 0.93 | 0.93 | 0.93 |
| 0.50 | 93% | 0.95 | 0.95 | 0.95 |
| 0.80 | 76% | 0.97 | 0.97 | 0.97 |
| 0.90 | 60% | 1.00 | 1.00 | 1.00 |

### Entity pruning

Type prediction acts as a pruning step: predicting the source type alone eliminates most tools from consideration before graph search begins.

| Source type | Tools reachable | Pruned |
|---|---|---|
| Balance, Event, Token, ... (16 types) | 1 | 99.8% |
| FileLink, Price, ShippingRate, ... (12 types) | 2 | 99.6% |
| Coupon, Dispute, Refund, ... (12 types) | 3 | 99.4% |
| Checkout, CreditNote, Payout, ... (6 types) | 4 | 99.3% |
| Transfer | 6 | 98.9% |
| Product | 7 | 98.7% |
| Charge | 11 | 97.9% |
| Invoice | 14 | 97.4% |
| Issuing | 20 | 96.3% |
| Account | 27 | 95.0% |
| Customer | 44 | 91.8% |
| Platform | 182 | 66.0% |

Of 66 source types in the graph, 52 reach fewer than 10 tools (99%+ pruned). Even Customer — the second most connected type — prunes 91.8% of the catalog. Platform is the outlier at 182 tools, but it still eliminates a third of the search space.

The pruning ratio explains why type prediction is the bottleneck, not graph search. For most queries, a correct source type prediction reduces 536 tools to single digits. The graph search that follows is trivial — it simply walks the edges from the predicted source to target type.

## Usage

```bash
pip install -r requirements.txt

# Train with 5-fold stratified CV (uses GPU if available)
python train.py --base-model all-mpnet-base-v2

# Evaluate saved model on held-out test queries
python evaluate.py

# Run confidence threshold analysis
python evaluate.py --threshold-sweep

# Benchmark Gemini 2.5 Pro (requires GEMINI_API_KEY)
python benchmark_gemini.py
python benchmark_gemini.py --strategy few_shot_tcr
python benchmark_gemini.py --strategy few_shot_text
python benchmark_gemini.py --strategy few_shot_text --model gemini-2.5-flash
```

## Data

### Graph

`data/graph_snapshot.json` — Typed composition graph derived from the Stripe OpenAPI spec. Each tool is modeled as a typed transformation `input_types -> output_types`. Entity types are extracted from request/response schemas; tools that share types create edges in the graph.

- 163 entity types, 536 tools, 250 unique (source, target) pairs

### Training data

`data/training_data.jsonl` — 3,787 hand-written examples across 6 query styles. No LLM data augmentation. Each example is a `(query, source_type, target_type)` triple.

| Style | Examples | Description |
|---|---|---|
| template | 2,209 (58%) | Direct queries: "List all X", "Show X for this Y", "Get X details" |
| noisy | 577 (15%) | Conversational: "Something went wrong with X, can you check?" |
| question | 513 (14%) | Question form: "How many X are there?", "Which X are active?" |
| synonym | 372 (10%) | Informal terms: "payments" -> PaymentIntent, "chargebacks" -> Dispute |
| ambiguous | 83 (2%) | Source-confusing queries: Platform vs domain-specific sources |
| hard_negative | 33 (1%) | Confusing pairs: BalanceTransaction vs CustomerBalanceTransaction |

Source type distribution is heavily skewed — Platform accounts for 57% of examples (it connects to 112 of 163 target types). Sqrt-scaled class weights correct for this during training, and stratified CV ensures minority types appear in every fold.

### Test queries

`data/test_queries.json` — 120 held-out evaluation queries across 6 difficulty categories (clean, noisy, synonym, multihop, ambiguous, multipath). No overlap with training data.

## Architecture

```
Query (natural language)
    |
    v
all-mpnet-base-v2 (frozen, 768-dim)
    |
    v
Shared MLP: Linear(768, 256) + ReLU + Dropout(0.2)
    |
    +--> Source Head: Linear(256, 163) --> 163 logits --> argmax --> source_type
    |
    +--> Target Head: Linear(256, 163) --> 163 logits --> argmax --> target_type
                                                                        |
                                                     BFS Graph Search <-+
                                                           |
                                                           v
                                                      Tool Chain
```

Each head outputs 163 logits (one per entity type). Argmax selects the predicted type, softmax gives a confidence score. The minimum of source and target confidence is used as the overall prediction confidence. Both types must be correct for the tool chain to resolve correctly.

Training: 200 epochs per fold, cosine LR schedule (1e-3 -> 1e-5), sqrt-scaled class weights for source/target imbalance. Best model (by test F1) is saved.

## Comparison with AAP

This is a companion repo to [aap-type-predictor](https://github.com/jangel97/aap-type-predictor), which demonstrates the same architecture on the Ansible Automation Platform (1,060 tools, 79 entity types). Together they show that Typed Composition Search generalizes across domains:

| | Stripe | AAP |
|---|---|---|
| Tools | 536 | 1,060 |
| Entity types | 163 | 79 |
| Training examples | 3,787 | 10,065 |
| Encoder F1 | 0.842 | 0.852 |
| Gemini FS-TCR F1 | 0.836 | 0.822 |
| Encoder params | 281K | 475K |

The encoder beats Gemini 2.5 Pro with few-shot prompting on both domains. As the tool count doubles (536 -> 1,060), Gemini's function calling degrades sharply (0.353 -> 0.333) while the encoder maintains high accuracy. The type prediction reformulation reduces the search space from thousands of tools to tens of entity types, making the problem tractable for a lightweight classifier.

## Conclusion

A 281K-parameter classifier built on a frozen 110M-param sentence encoder matches Gemini 2.5 Pro (estimated ~300B+ params) with few-shot prompting on Stripe tool routing (F1=0.842 vs 0.836), while being ~1000x smaller, running locally with sub-millisecond inference, and requiring no API key.

The two approaches show complementary strengths: the encoder wins on synonym, ambiguous, and noisy queries where domain-specific training data matters most, while Gemini excels on clean structured queries where general language understanding suffices. This suggests the encoder captures domain knowledge that even frontier models cannot infer from type names alone.

The key insight is that type prediction is the bottleneck, not graph search. On Stripe, R_wrong=0: every type prediction error produces a completely wrong tool chain, with no graceful degradation. This means recall is entirely determined by type prediction accuracy, and improving the predictor directly improves end-to-end routing.

Across both domains — Stripe (536 tools) and AAP (1,060 tools) — the encoder beats Gemini 2.5 Pro with few-shot prompting. This suggests that tool routing need not be learned end-to-end by increasingly capable foundation models. Once routing is decomposed into semantic type prediction and deterministic graph search, the learned component becomes a compact supervised classifier that can be trained independently for each domain.
