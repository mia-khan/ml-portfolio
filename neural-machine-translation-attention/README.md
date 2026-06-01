# Neural Machine Translation with Attention

Implementation of a full sequence-to-sequence neural machine translation system in PyTorch,
built from scratch for Spanish-to-English translation. Every attention component —
cross-attention, multi-head attention, encoder self-attention, decoder self-attention —
is implemented at the weight level without relying on pretrained frameworks.

## What This Project Does

Translation is one of the canonical problems in NLP. The goal here is not to beat a
production model, but to deeply understand how attention-based architectures work by
implementing them from the ground up. Starting from a simple RNN baseline and building
up to a full Transformer decoder allows each design decision to be isolated and studied.

## Architecture Overview

```
Source sentence (Spanish)
        │
        ▼
  Token Embeddings + Positional Encoding
        │
        ▼
  Transformer Encoder
  ┌──────────────────────────────────┐
  │  Multi-Head Self-Attention       │  ← attends across source tokens
  │  Add & Norm                      │
  │  Position-wise Feed-Forward      │
  │  Add & Norm                      │
  └──────────────────────────────────┘  × N blocks
        │
        ▼  (encoder outputs = contextualized source representations)
        │
  Transformer Decoder
  ┌──────────────────────────────────┐
  │  Masked Multi-Head Self-Attention│  ← attends to previous target tokens only
  │  Add & Norm                      │
  │  Cross-Attention                 │  ← queries from decoder, keys/values from encoder
  │  Add & Norm                      │
  │  Position-wise Feed-Forward      │
  │  Add & Norm                      │
  └──────────────────────────────────┘  × N blocks
        │
        ▼
  Linear + Softmax → Target vocabulary distribution
        │
        ▼
  Target token (English)
```

## Attention Mechanisms Implemented

### Scaled Dot-Product Attention
The core operation: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`

Scaling by `sqrt(d_k)` prevents the dot products from growing large in magnitude and
pushing the softmax into regions with very small gradients. A masked softmax ensures
padded tokens contribute zero probability mass.

### Single-Head Attention
Learned linear projections map queries, keys, and values to a common hidden dimension
before computing scaled dot-product attention. Equivalent to one head of multi-head attention.

### Multi-Head Attention
Runs `h` attention heads in parallel, each with its own Q/K/V projection matrices.
Each head can learn to attend to different positions or relationship types simultaneously.
Outputs are concatenated and projected back to the model dimension:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
where head_i = Attention(Q * W_Q_i, K * W_K_i, V * W_V_i)
```

### Encoder Self-Attention
Every source token attends to every other source token (no masking). This builds a
contextualized representation where each token's embedding reflects the full sentence.

### Decoder Self-Attention (Masked)
Target tokens can only attend to previous positions — future tokens are masked out.
This preserves the autoregressive property: at inference time, the decoder generates
one token at a time and cannot "look ahead."

### Cross-Attention
Queries come from the decoder; keys and values come from the encoder outputs.
This is the mechanism by which the decoder "reads" the source sentence when
deciding what to generate next. It is the core of sequence-to-sequence translation.

## Baseline Comparison: RNN and GRU Encoders

The project also trains RNN/GRU baselines to contrast with the Transformer. Key differences:
- RNNs process tokens sequentially and compress the entire source into a fixed-size hidden state
- Transformers process all tokens in parallel and maintain separate representations per position
- The attention mechanism allows the decoder to selectively focus on any part of the source,
  bypassing the bottleneck of a single context vector

## Training and Evaluation

**Dataset:** Spanish-English parallel corpus (`sentence_pairs.tsv`)

**Evaluation Metrics:**
- **BLEU Score** — n-gram overlap between generated and reference translations.
  A BLEU of 0.3+ is generally considered acceptable for a from-scratch implementation
  on a mid-size corpus.
- **BERTScore** — semantic similarity using contextual embeddings. Captures meaning-level
  accuracy that BLEU misses (e.g. synonym substitutions).

**Beam Search Decoding:**
Greedy decoding always picks the single most likely next token. Beam search maintains
the top `k` candidate sequences at each step, then applies a length penalty to prevent
shorter sequences from being unfairly preferred:

```
score(sequence) = log_prob / ((5 + length) / 6)^alpha
```

## Key Implementation Details

- `MaskedSoftmax`: custom softmax that zeros out padded positions before computing
  the probability distribution over keys
- `PositionalEncoding`: sinusoidal fixed encodings added to token embeddings so the
  model can distinguish position without recurrence
- `AddNorm`: residual connection followed by layer normalization — critical for
  training stability in deep Transformer stacks
- State caching in beam search: each beam hypothesis carries its own copy of the
  decoder key/value cache so beam paths don't interfere with each other

## Setup

```bash
pip install torch d2l sacrebleu bert_score nltk
python attention_models.py
```

## Files

| File | Description |
|------|-------------|
| `attention_models.py` | All attention and Transformer implementations |
| `data_utils.py` | Data loading, tokenization, vocabulary building |
| `sentence_pairs.tsv` | Spanish-English parallel corpus |

## Technologies

Python · PyTorch · BLEU · BERTScore · Beam Search
