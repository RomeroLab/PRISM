# PRISM

**P**artitioning **R**esidue **I**dentity in **S**omatic **M**aturation

This is the official repository for the paper:

> **Explicit representation of germline and non-germline residues improves antibody language modeling**

PRISM is a PyTorch Lightning-based framework for supervised fine-tuning of ESM2 protein language models on antibody sequences. It features a multi-head architecture that jointly learns amino acid identity prediction and germline/non-germline (GL/NGL) position classification.

---

# Part 1: User Guide

Everything you need to run inference with PRISM on your own antibody data.

## Installation

```bash
pip install prism-antibody
```

Or install from source:

```bash
git clone https://github.com/RomeroLab-Duke/prism-antibody.git
cd prism-antibody
pip install -e .
```

### Verify Installation

```python
import prism
print(prism.__version__)
```

## Quick Start

```python
import prism

model     = prism.pretrained("RomeroLab-Duke/prism-antibody")
tokenizer = model.get_tokenizer()

# Tokenize → model (standard HuggingFace-style pipeline)
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")

# Forward pass — logits, embeddings, origin, alpha
result = model.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

# Predict germline — revert somatic mutations
germline = model.predict_germline(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
# germline["heavy"]["predicted_germline"]  → germline-reverted heavy chain
# germline["heavy"]["ngl_positions"]       → which positions were mutated
```

## Tokenizer

`PrismTokenizer` wraps the ESM2 tokenizer with PRISM's 53-token vocabulary (33 ESM2 base + 20 lowercase NGL tokens).

```python
tokenizer = prism.PrismTokenizer()          # standalone (no model needed)
tokenizer = model.get_tokenizer()           # or from a loaded model

# Paired heavy + light chain
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
# inputs["input_ids"]       -> [1, L_H+L_L+4]  (CLS + VH + CLS + CLS + VL + EOS)
# inputs["attention_mask"]  -> [1, L_H+L_L+4]

# Batch
inputs = tokenizer(
    ["EVQLVESGGGLVQ", "QVQLVQSGAEVKK"],
    light_chain=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
    return_tensors="pt",
)

# Unpaired (single chain)
inputs = tokenizer("EVQLVESGGGLVQ", return_tensors="pt")

# Encode / decode (paired)
ids = tokenizer.encode_paired("EVQLV", "DIQMT")
heavy, light = tokenizer.decode_paired(ids)    # ("EVQLV", "DIQMT")

# Encode / decode (unpaired)
ids = tokenizer.encode("EVQLV")               # [CLS, E, V, Q, L, V, EOS]
seq = tokenizer.decode(ids)                    # "EVQLV"
```

### NGL-Aware Tokenization

By default, all amino acids are tokenized as uppercase (GL) tokens — this is the standard mode and matches the training format. Use `preserve_case=True` **only** when you need `exact` mode in `pseudo_log_likelihood()`, which scores each position using its actual GL or NGL log-probability.

```python
# For exact PLL: lowercase = NGL (somatic mutation) positions
inputs = tokenizer("EvQLvESGGglvq", preserve_case=True, return_tensors="pt")
# 'v', 'g', 'l', 'v' → NGL token IDs;  'E', 'Q', 'L', ... → GL token IDs

result = model.pseudo_log_likelihood(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
# result["exact"] now uses NGL log-prob at lowercase positions
```

### GL/NGL Token Mappings

```python
tokenizer.gl_token_ids     # {"A": 5, "C": 23, ...}  — 20 uppercase (germline)
tokenizer.ngl_token_ids    # {"a": 33, "c": 34, ...}  — 20 lowercase (non-germline)
tokenizer.gl_to_ngl        # {5: 33, 23: 34, ...}     — GL→NGL token ID mapping
tokenizer.vocab_size       # 53
```

## API Overview

PRISM has **5 core methods**. All accept pre-tokenized `input_ids` (recommended) or raw strings.

| Method | Cost | Returns |
|--------|------|---------|
| `forward()` | 1 forward pass | logits, embeddings, origin, alpha |
| `pseudo_log_likelihood()` | ceil(L / batch_size) forward passes | PLL, perplexity, per-position log-probs (4 modes) |
| `score_mutations()` | 2 × ceil(M / batch_size) forward passes | masked marginal mutation scores (4 modes) |
| `predict_germline()` | 1 forward pass | predicted germline sequences, NGL positions/probs |
| `generate()` | L + N forward passes | PLL-guided antibody variants |

## `forward()` --- Logits, Embeddings, Everything

Single forward pass through the model. Returns all outputs as numpy arrays.

```python
import prism
import numpy as np

model     = prism.pretrained("RomeroLab-Duke/prism-antibody")
tokenizer = model.get_tokenizer()

# Standard: tokenize → forward (paired)
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
result = model.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
# result["final_logits"]  -> [L, 53]  alpha-gated combined logits
# result["aa_logits"]     -> [L, 33]  AA head logits (pre-gating)
# result["origin_logits"] -> [L]      GL/NGL classification logits
# result["alpha"]         -> [L]      gating values
# result["embedding"]     -> [L, H]   per-residue hidden states

# GL/NGL log-probabilities (slice from 53-vocab)
gl_logits  = result["final_logits"][:, model.GL_INDICES]   # [L, 20]
ngl_logits = result["final_logits"][:, model.NGL_INDICES]  # [L, 20]
```

### Batch, Unpaired (string convenience)

```python
# Batch (returns list of {"heavy": {...}, "light": {...}})
results = model.forward(
    heavy_chains=["EVQLVESGGGLVQ", "QVQLVQSGAEVKK"],
    light_chains=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
)

# String convenience (paired)
result = model.forward(heavy_chains="EVQLVESGGGLVQ", light_chains="DIQMTQSPSSLSA")

# Unpaired (single chain)
result = model.forward("EVQLVESGGGLVQPGGSLRL")
```

## `pseudo_log_likelihood()` --- PLL and Perplexity

Masks each position one at a time, accumulates log P(true token). Returns **4 scoring modes** in one pass.

```python
# Standard: tokenize → PLL
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
result = model.pseudo_log_likelihood(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
# {
#   "marginalized": {"pll": -45.3, "perplexity": 2.34, "per_position": [L]},
#   "gl":           {"pll": -50.1, "perplexity": 2.71, "per_position": [L]},
#   "ngl":          {"pll": -48.2, "perplexity": 2.56, "per_position": [L]},
#   "exact":        {"pll": -50.1, "perplexity": 2.71, "per_position": [L]},
# }

ppl = result["marginalized"]["perplexity"]
```

### NGL-Aware Scoring with `exact` Mode

When the input contains NGL tokens (lowercase via `preserve_case=True` --- see [NGL-Aware Tokenization](#ngl-aware-tokenization)), the `exact` mode scores each position using its actual token: uppercase log-prob for GL positions, lowercase log-prob for NGL positions.

### Scoring Modes

All modes are computed from the 53-vocab alpha-gated logits. `gl`, `ngl`, and `marginalized` extract the GL/NGL slots and combine them back into 20-AA probabilities.

| Mode | What it scores | Use case |
|------|---------------|----------|
| `marginalized` | `logsumexp(GL, NGL)` per AA | General-purpose scoring |
| `gl` | Uppercase (GL) token log-prob | Germline likeness |
| `ngl` | Lowercase (NGL) token log-prob | Somatic mutation preference |
| `exact` | Actual input token log-prob | NGL-aware scoring (with `preserve_case=True`) |

### Batch Processing

`batch_size` controls how many masked positions are processed in a single forward pass. Higher values use more GPU memory but run faster.

```python
# Fast: 64 positions per forward pass (needs ~2x memory vs default)
result = model.pseudo_log_likelihood(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    batch_size=64,
)
```

For multiple sequences, pass a list --- they are scored sequentially, each with the same `batch_size` parallelism:

```python
# Multiple sequences (processed one at a time, results in order)
results = model.pseudo_log_likelihood(
    heavy_chains=["EVQLVESGGGLVQ", "QVQLVQSGAEVKK"],
    light_chains=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
)
# results[0] → first pair, results[1] → second pair
```

### String Convenience

```python
# Paired
result = model.pseudo_log_likelihood(
    heavy_chains="EVQLVESGGGLVQ",
    light_chains="DIQMTQSPSSLSA",
)

# Unpaired
result = model.pseudo_log_likelihood("EVQLVESGGGLVQPGGSLRL")
```

## `score_mutations()` --- Mutation Effect Prediction

Masked marginal scoring at mutation positions. For each mutated position, masks that position in both WT and mutant, runs a forward pass, and computes the log-likelihood difference. Returns all 4 scoring modes.

```python
# Standard: tokenize → score (paired)
wt_inputs  = tokenizer("EVQLVESGGGLVQPGGSLRL", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
mut_inputs = tokenizer("EVQLVASGGGLVQPGGSLRL", light_chain="DIQMTQSPSSLSA", return_tensors="pt")  # V6A
result = model.score_mutations(
    wt_input_ids=wt_inputs["input_ids"],
    mut_input_ids=mut_inputs["input_ids"],
)
# {
#   "positions": [5],  # 0-indexed mutation positions (detected from token diff)
#   "marginalized": {"score": 0.42, "per_position": [1]},
#   "gl":           {"score": 0.31, "per_position": [1]},
#   "ngl":          {"score": 0.55, "per_position": [1]},
#   "exact":        {"score": 0.31, "per_position": [1]},
# }
# score > 0 = mutant preferred over WT
```

### Batch Processing

`batch_size` controls how many mutation positions are masked per forward pass. For sequences with many mutations, higher values are faster.

```python
result = model.score_mutations(
    wt_input_ids=wt_inputs["input_ids"],
    mut_input_ids=mut_inputs["input_ids"],
    batch_size=64,
)
```

For multiple WT/mutant pairs, pass lists --- they are scored sequentially:

```python
results = model.score_mutations(
    wt=["EVQLVESGGGLVQ", "QVQLVQSGAEVKK"],
    mutant=["EVQLVASGGGLVQ", "QVQLVQSGAEVAK"],
    wt_light_chains=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
    mut_light_chains=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
)
# results[0] → first pair, results[1] → second pair
```

### String Convenience

```python
# Paired
result = model.score_mutations(
    wt="EVQLVESGGGLVQPGGSLRL",
    mutant="EVQLVASGGGLVQPGGSLRL",
    wt_light_chains="DIQMTQSPSSLSA",
    mut_light_chains="DIQMTQSPSSLSA",
)

# Unpaired
result = model.score_mutations(
    wt="EVQLVESGGGLVQPGGSLRL",
    mutant="EVQLVASGGGLVQPGGSLRL",
)
```

## `predict_germline()` --- Germline Sequence Prediction

Predicts the unmutated germline sequence from a somatically hypermutated antibody. Uses the origin head to identify non-germline (NGL) positions and reverts them to the top-scoring germline amino acid --- all in a single forward pass.

```python
import prism

model     = prism.pretrained("RomeroLab-Duke/prism-antibody")
tokenizer = model.get_tokenizer()

# Standard: tokenize → predict germline (paired)
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
result = model.predict_germline(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
# {
#   "heavy": {
#     "sequence": "EVQLVESGGGLVQ",          # original
#     "predicted_germline": "EVQLVESGGGLVQ", # germline-reverted
#     "ngl_positions": [5, 8],               # 0-indexed NGL positions
#     "ngl_count": 2,
#     "ngl_probs": array([0.02, ..., 0.91, ..., 0.87, ...]),  # [L_H] P(NGL)
#   },
#   "light": { ... same structure ... },
# }
```

### How It Works

1. **Origin head** classifies each position as GL or NGL via `sigmoid(origin_logits)`.
2. Positions with `P(NGL) > ngl_threshold` are identified as somatically mutated.
3. At those positions, the residue is replaced with `argmax` over the 20 GL amino acid logits from the final head.
4. GL positions are left unchanged.

### Paired Auto-Detection with `input_ids`

When using pre-tokenized `input_ids`, paired sequences are **automatically detected** by finding the `<cls><cls>` separator in the token IDs --- no additional parameters needed:

```python
# Paired: auto-detected from <cls><cls> in input_ids
inputs = tokenizer("EVQLVESGGGLVQ", light_chain="DIQMTQSPSSLSA", return_tensors="pt")
result = model.predict_germline(input_ids=inputs["input_ids"])
# → result["heavy"], result["light"]

# Unpaired: no <cls><cls> → flat output
inputs = tokenizer("EVQLVESGGGLVQ", return_tensors="pt")
result = model.predict_germline(input_ids=inputs["input_ids"])
# → result["sequence"], result["predicted_germline"], ...
```

### Controlling the Threshold

```python
# Aggressive: revert more positions (lower threshold)
result = model.predict_germline(
    input_ids=inputs["input_ids"],
    ngl_threshold=0.3,
)

# Conservative: only revert high-confidence NGL positions
result = model.predict_germline(
    input_ids=inputs["input_ids"],
    ngl_threshold=0.8,
)
```

### Batch Processing

```python
inputs = tokenizer(
    ["EVQLVESGGGLVQ", "QVQLVQSGAEVKK"],
    light_chain=["DIQMTQSPSSLSA", "EIVLTQSPGTLSL"],
    return_tensors="pt",
)
results = model.predict_germline(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
)
# results[0] → first pair, results[1] → second pair
```

### String Convenience

```python
# Paired
result = model.predict_germline(
    heavy_chains="EVQLVESGGGLVQ",
    light_chains="DIQMTQSPSSLSA",
)

# Unpaired
result = model.predict_germline(heavy_chains="EVQLVESGGGLVQ")
```

## `generate()` --- PLL-Guided Variant Generation

Generates antibody variants using pseudo-log-likelihood guided sampling:

1. **Collect** --- mask each position one at a time, collect pre-gating logits (L forward passes, cached and reusable)
2. **Select positions** --- rank by WT log-probability, sample via Gumbel-Top-k with controllable temperature
3. **Sample amino acids** --- draw from GL, NGL, marginalized, or region-specific logits with temperature, top-k, and nucleus sampling

```python
# Standard: tokenize → generate
inputs = tokenizer(
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMS",
    light_chain="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN",
    return_tensors="pt",
)

variants = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100,                # number of variants to generate
    n_mutations=5,                # mutations per variant
    mode="full",                  # gl | ngl | full | region_specific
    seed=42,
)
# List of 100 dicts:
# [
#   {"sequence": "EVQLVE...", "mutations": "S7A,G10D,...", "positions": [6, 9, ...],
#    "mode": "full", "n_mut": 5},
#   ...
# ]
```

### Sampling Modes

| Mode | Position scoring | AA sampling | Use case |
|------|-----------------|-------------|----------|
| `"full"` | Marginalized log P(wt) | `logsumexp(GL, NGL)` logits | General-purpose diversification |
| `"gl"` | GL log P(wt) | GL (germline) logits only | Germline reversion / humanization |
| `"ngl"` | NGL log P(wt) | NGL (non-germline) logits only | Affinity maturation mimicry |
| `"region_specific"` | FR: GL, CDR: NGL | FR: GL logits, CDR: NGL logits | Targeted: conserve FRs, diversify CDRs |

### Controlling Generation

```python
variants = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100,
    n_mutations=5,

    # --- Position selection ---
    pool_size=30,                 # candidate pool (top-30 worst positions)
    position_temperature=0.5,     # lower = more deterministic position choice
    exclude_positions=np.array([0, 1, 2]),  # never mutate these (0-indexed)

    # --- Amino acid sampling ---
    temperature=0.8,              # lower = more conservative AA choices
    top_k=10,                     # only consider top-10 AAs per position
    top_p=0.9,                    # nucleus sampling threshold

    # --- Variation ---
    randomize_n_mutations=True,   # n_mut ~ Beta(2,1) in [1, n_mutations]
    seed=42,                      # reproducibility
)
```

### Region-Specific Mode

The `"region_specific"` mode uses framework region (FR) and complementarity-determining region (CDR) annotations to apply different sampling strategies: GL logits for FR positions (conserve structure) and NGL logits for CDR positions (diversify binding).

Regions are auto-detected using [ANARCI](https://github.com/oxpig/ANARCI) (IMGT numbering). Pass `heavy_chain_length` so VH and VL are numbered separately:

```python
variants = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100,
    n_mutations=5,
    mode="region_specific",
    heavy_chain_length=len("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMS"),
)
```

Or provide region labels manually (0 = FR, 1 = CDR):

```python
import numpy as np
L = len(vh_seq) + len(vl_seq)
region_labels = np.zeros(L, dtype=np.int32)
region_labels[26:34] = 1   # CDR1
region_labels[51:57] = 1   # CDR2
region_labels[93:102] = 1  # CDR3
# ... repeat for VL

variants = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100,
    n_mutations=5,
    mode="region_specific",
    region_labels=region_labels,
)
```

### Caching Masked Logits Across Modes

The most expensive step (L forward passes) can be computed once and reused across different modes:

```python
# First call: collect masked logits + generate
variants_full, cache = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100, n_mutations=5, mode="full", seed=42,
    return_masked_data=True,
)

# Subsequent calls: skip L forward passes (instant)
variants_gl = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100, n_mutations=5, mode="gl", seed=42,
    masked_data=cache,
)

variants_ngl = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    n_samples=100, n_mutations=5, mode="ngl", seed=42,
    masked_data=cache,
)
```

### String Convenience

```python
variants = model.generate(
    heavy_chains="EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMS",
    light_chains="DIQMTQSPSSLSASVGDRVTITCRASQSISSYLN",
    n_samples=100, n_mutations=5, mode="full", seed=42,
)
```

## Reference

### `forward()` Return Dict

| Key | Shape | Description |
|-----|-------|-------------|
| `final_logits` | `[L, 53]` | Alpha-gated combined logits (53-vocab) |
| `aa_logits` | `[L, 33]` | AA head logits, before gating |
| `origin_logits` | `[L]` | GL/NGL binary classification logits |
| `alpha` | `[L]` | Per-position gating values |
| `embedding` | `[L, H]` | Per-residue hidden states from backbone |

When paired (string API or auto-detected `<cls><cls>`), returns `{"heavy": {dict}, "light": {dict}}`.

### `predict_germline()` Return Dict

| Key | Type | Description |
|-----|------|-------------|
| `sequence` | `str` | Original amino acid sequence |
| `predicted_germline` | `str` | Germline-reverted sequence (NGL positions replaced) |
| `ngl_positions` | `list[int]` | 0-indexed positions classified as NGL |
| `ngl_count` | `int` | Number of NGL positions |
| `ngl_probs` | `[L]` numpy array | Per-position P(NGL) from origin head |

When paired, returns `{"heavy": {dict}, "light": {dict}}` with per-chain values.

### Index Constants

- `model.GL_INDICES` --- 20 uppercase AA token indices in the 53-vocab
- `model.NGL_INDICES` --- 20 lowercase AA token indices in the 53-vocab
- `model.AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"` --- column order for the 20 AA indices

---

# Part 2: Developer Guide

For researchers and developers who want to train from scratch, run analysis pipelines, or extend the codebase.

## Development Installation

```bash
git clone https://github.com/RomeroLab-Duke/prism-antibody.git
cd prism-antibody
pip install -e ".[dev,analysis]"
```

## Project Structure

```
prism/
├── src/prism/                    # Core Python package
│   ├── api.py                    # High-level inference API
│   ├── tokenizer.py              # PrismTokenizer (53-vocab, paired support)
│   ├── model.py                  # SFT_ESM2 PyTorch Lightning module
│   ├── io_utils.py               # Dataset & DataModule classes
│   ├── multimodal_io.py          # Gene vocabulary & antibody dataset
│   └── utils.py                  # Utility functions
│
├── configs/                      # YAMLs to reproduce paper results
│   ├── v34_pretrain.yaml             # canonical pretrain
│   ├── v34_1b_finetune.yaml          # canonical paper model
│   ├── v34_1b_noise{2,4}_finetune.yaml   # noise-robustness ablation
│   ├── v_baseline_{pre,fine}tune.yaml    # PRISM-less baseline
│   ├── ablation_{alpha_*,no_pretrain,simple_*}.yaml  # architectural ablations
│   └── pll_guided_sampler.yaml       # generation sampler
│
├── script/
│   ├── train_esm.py                  # PRISM trainer
│   ├── train_pure_esm.py             # vanilla-ESM2 baseline trainer
│   ├── inference_esm.py              # forward + embeddings
│   ├── inference_pure_esm_with_logprobs.py   # vanilla-ESM2 baseline scorer
│   ├── data/                         # OAS preprocessing pipeline (1-8)
│   └── analyze/                      # paper-section ordered analyses
│       ├── 1.disentanglement/        # Fig. 2A-C: linear probing + UMAP
│       ├── 2.pseudo_perplexity/      # Fig. 2D-G + SI 2: PPL stratified
│       ├── 3.controllable_generation/ # Fig. 3 + SI 4-5: Rosetta/MLP/CamSol
│       ├── 4.zero_shot_binding/      # Fig. 4: DMS + FLAb2 binding
│       ├── 5.zero_shot_developability/ # Fig. 5: developability assays
│       ├── 6.ablation/               # Fig. 6 + SI 8-12: arch + alpha + noise
│       └── 7.thera_sabdab/           # SI 17: therapeutic generalization
│
├── pyproject.toml                # Package configuration
├── LICENSE                       # MIT
└── CITATION.cff                  # Paper citation
```

## Training from Scratch

### Two-Stage Training Protocol

**Stage 1 --- Pretraining** on large unpaired OAS dataset (~60M+ sequences):

```bash
python script/train_esm.py --config configs/v34_pretrain.yaml
```

**Stage 2 --- Finetuning** on paired antibody sequences (~764K):

```bash
python script/train_esm.py --config configs/v34_1b_finetune.yaml
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python script/train_esm.py --config configs/v34_pretrain.yaml
```

Edit `data_path`, `gene_vocab_path`, etc. in the configs from the `/path/to/prism/...` placeholder to the location where you downloaded the OAS data.

## Reproducing Paper Figures

Each paper figure maps to a numbered subdirectory under `script/analyze/`:

| Paper section | Figure | Script directory |
|---|---|---|
| Disentanglement | Fig. 2A-C, SI 10 | `script/analyze/1.disentanglement/` |
| Pseudo-perplexity | Fig. 2D-G, SI 2 | `script/analyze/2.pseudo_perplexity/` |
| Controllable generation | Fig. 3, SI 4-5, 13, 16 | `script/analyze/3.controllable_generation/` |
| Zero-shot binding | Fig. 4 | `script/analyze/4.zero_shot_binding/` |
| Zero-shot developability | Fig. 5, SI 3, 6 | `script/analyze/5.zero_shot_developability/` |
| Ablation + α-gating + noise | Fig. 6, SI 8-12 | `script/analyze/6.ablation/` |
| Therapeutic generalization | SI 17 | `script/analyze/7.thera_sabdab/` |

### Baselines

PRISM's figures compare against five baseline language models. Scoring scripts live alongside the PRISM-side scripts in each `analyze/N.*/` directory:

- **IgLM, AntiBerty**: `4.zero_shot_binding/iglm/`, `5.zero_shot_developability/iglm/`, `7.thera_sabdab/score_therasabdab_w_iglm.py` (shared utilities at `script/analyze/utils/`).
- **ESM2-35M, ESM2-650M, AbLang2, Sapiens**: scored via the corresponding `evaluate_*_baselines.py` / `benchmark_*_baselines.py` files in `4.zero_shot_binding/` and `5.zero_shot_developability/`.

### DMS surrogate (Fig. 3 MLP axis)

The "MLP" scoring axis of Fig. 3 is a Ridge regressor trained on each antibody's DMS labels (paper line 913). Trainer at `3.controllable_generation/train_dms_surrogate.py`; pre-trained weights for the three Fig. 3 antibodies at `3.controllable_generation/dms_surrogates/{cr9114,g631,trastuzumab}_ridge.joblib`.

### CamSol

CamSol scores (Fig. 3) are produced by the [CamSol web server](https://www-cohsoftware.ch.cam.ac.uk/index.php/login) at pH 7.0; the local pipeline reads back the server outputs from `data/`.

---

## License

MIT License --- see [LICENSE](LICENSE) for details.
