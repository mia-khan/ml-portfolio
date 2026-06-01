# Transformer Architectures for Financial Language Modeling

A comparative study of language model architectures trained on Warren Buffett's
shareholder letters — a specialized financial corpus with distinctive vocabulary,
long-range dependencies, and highly consistent stylistic patterns. Three architectures
are implemented and evaluated: a dictionary-based n-gram model, a PyTorch embedding
model, and a full Transformer, with analysis of how each handles memorization,
generalization, and the inductive biases of language modeling in a narrow domain.

## Why This Corpus?

Most language modeling benchmarks use general-purpose corpora. Buffett's letters offer
a small, dense, domain-specific dataset written by one author over decades — ideal for
studying memorization vs. generalization and how inductive biases play out when data is
scarce and the vocabulary is specialized.

## Architectures Compared

### N-gram (Dictionary) Model
Builds frequency tables over word sequences of length `n`. Predicts the next word by
looking up observed continuations given the last `n-1` words.

- **Inductive bias:** Markov assumption — only the previous `n-1` words matter
- **Strength:** Lowest perplexity on in-distribution text; memorizes exact Buffett phrases
- **Weakness:** Fails completely on unseen n-gram contexts; brittle to novel phrasing

### Embedding Model (PyTorch)
Learns dense vector representations for each word. Context window is embedded,
concatenated, and passed through a hidden layer to predict the next token.

- **Inductive bias:** Words with similar distributions get similar embeddings
- **Strength:** Generalizes to unseen word combinations by composing known embeddings
- **Weakness:** Less exact than memorized tables; occasionally grammatically inconsistent

### Transformer Language Model
Multi-head self-attention over a context window with positional encoding and a
position-wise feed-forward network. Attends to any previous token regardless of distance.

- **Inductive bias:** Uniform attention over all positions; no locality constraint
- **Strength:** Best generalization; captures long-range dependencies and paragraph coherence
- **Weakness:** Requires most compute; needs more data to outperform simpler models

## Evaluation

### Perplexity
```
Perplexity = exp(average negative log-likelihood per token)
```
Lower is better. Measures how "surprised" the model is by held-out text.

| Model | Behavior |
|-------|----------|
| N-gram | Lowest train perplexity, high test perplexity on novel inputs |
| Embedding | Balanced — worse than n-gram on training, better on novel phrasing |
| Transformer | Best generalization across all held-out splits |

### Memorization vs. Generalization Analysis
Key experiment: compare model performance on (1) verbatim training sentences,
(2) same vocabulary but novel structure, (3) completely unseen constructions.
The n-gram model excels at (1) but fails at (3). The Transformer generalizes most
evenly — evidence that attention captures structural rather than surface-level patterns.

## Key Findings

- Dictionary models excel at reproducing training text but are brittle to novel inputs
- Embedding models balance creativity and coherence — useful when vocabulary is stable
  but new combinations are expected at test time
- Transformers generalize best due to attention's ability to capture long-range structure
- All three models quickly pick up financial register (passive constructions, hedged
  language, technical terminology) faster than they would on a general corpus

## Setup

```bash
pip install torch numpy matplotlib jupyter
jupyter notebook assignment_3.ipynb
```

## Files

| File | Description |
|------|-------------|
| `assignment_3.ipynb` | All three models, training loops, perplexity evaluation |
| `WarrenBuffet.txt` | Training corpus: Warren Buffett shareholder letters |

## Technologies

Python · PyTorch · NumPy · Matplotlib · Jupyter
