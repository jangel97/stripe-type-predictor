# Data

Training and evaluation data for entity type prediction on the Stripe API.

## Graph

`graph_snapshot.json` — Typed composition graph derived from the Stripe OpenAPI spec. Each tool is modeled as a typed transformation `input_types -> output_types`. Entity types are extracted from request/response schemas; tools that share types create edges in the graph.

- 163 entity types, 536 tools, 250 unique (source, target) pairs

## Training data

`training_data.jsonl` — 3,787 hand-written examples across 6 query styles. No LLM data augmentation. Each example is a `(query, source_type, target_type, style)` tuple.

### Style distribution

| Style | Examples | Description |
|---|---|---|
| template | 2,209 (58%) | Direct queries: "List all {X}", "Show {X} for this {Y}", "Get {X} details" |
| noisy | 577 (15%) | Conversational: "Something went wrong with {X}, can you check?" |
| question | 513 (14%) | Question form: "How many {X} are there?", "Which {X} are active?" |
| synonym | 372 (10%) | Informal terms: "payments" -> PaymentIntent, "chargebacks" -> Dispute |
| ambiguous | 83 (2%) | Source-confusing queries: Platform vs domain-specific sources (Terminal, Checkout, Billing, Radar) |
| hard_negative | 33 (1%) | Confusing pairs: BalanceTransaction vs CustomerBalanceTransaction |

### Source type distribution

Platform accounts for 58% of source types (2,209 examples), reflecting that it connects to 112 of 163 target types.

| Source type | Examples | % |
|---|---|---|
| Platform | 2,266 | 60% |
| Customer | 227 | 6% |
| Treasury | 132 | 3% |
| Issuing | 108 | 3% |
| Account | 73 | 2% |
| Billing | 60 | 2% |
| Other (61 types) | 921 | 24% |

### Data construction

The training data was built without LLM augmentation — all examples are hand-written or template-generated:

1. **Deterministic templates** (2,209) — Template-based generation from the graph snapshot. Covers all direct (source, target) pairs with structured query patterns.
2. **Noisy/conversational** (577) — Realistic scenarios with situational context: failed payments, disputed charges, expired subscriptions.
3. **Question form** (513) — Same coverage as templates but phrased as questions.
4. **Synonyms** (372) — Informal terms mapped to Stripe entity types. Critical for handling queries like "show me the chargebacks" (Dispute) or "check the payments" (PaymentIntent).
5. **Ambiguous** (83) — Source disambiguation examples for types where Platform competes with domain sources (e.g., Terminal->TerminalReader vs Platform->TerminalReader).
6. **Hard negatives** (33) — Confusing entity pairs: BalanceTransaction vs CustomerBalanceTransaction, SetupIntent vs PaymentIntent.

### Key design decisions

- **No LLM augmentation**: All training data is deterministic or hand-written. This keeps the data pipeline reproducible and avoids contamination from model biases.
- **Sqrt-scaled class weights** during training correct for the heavy Platform skew.
- **Zero test/train overlap**: All 120 test queries are filtered out during data construction.
- **Iterative disambiguation**: Ambiguous examples were added after analyzing source type confusion in the initial model — particularly for Terminal, Checkout, Billing, and Radar sources that compete with Platform.

## Test queries

`test_queries.json` — 120 held-out evaluation queries across 6 difficulty categories. Written independently of the training data.

| Category | N | Description |
|---|---|---|
| clean | 45 | Well-structured, unambiguous API requests |
| synonym | 21 | Use informal terminology for Stripe concepts |
| multihop | 19 | Require traversing 2+ edges in the composition graph |
| ambiguous | 15 | Multiple valid interpretations or source type confusion |
| noisy | 15 | Real-world conversational language with extra context |
| multipath | 5 | Multiple valid solution paths exist |
