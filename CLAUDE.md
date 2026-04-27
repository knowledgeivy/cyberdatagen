# CLAUDE.md — cyberdata-journal

This file is read automatically by Claude Code at the start of every session.
It provides full project context so no re-explanation is needed.

---

## What This Project Is

**Journal post-publication extension** of a conference paper published at ICISSP 2026:

> "Can Synthetic Spam Beat Real-World Detectors? Evaluating LLMs' Dual Role in Spam Generation and Detection"

The conference version is archived in `paper_conference/` and in the separate GitHub
repo `cyberdata` (tagged `v1.0-icissp2026`). This repo (`cyberdata-journal`) contains
the extended journal version.

- **Journal submission deadline**: 2026-07-08
- **Acceptance confirmation deadline**: 2026-06-05
- **Requirement**: ≥30% new material; new title, abstract, conclusion; no verbatim copying

---

## What the Conference Paper Did

- **Dataset**: CEAS-08 spam email corpus (2008 data), sampled at **1:9 spam/ham imbalance**
- **LLM Generators**: GPT-4.1-mini (OpenAI), Claude-3.5-Haiku (Anthropic)
- **Feature extraction**: TF-IDF (max 50k features, 1-2 ngrams, sublinear_tf)
- **Classifiers**: SVM, Random Forest (grid search, 5-fold CV, F1 metric)
- **Experimental design**:
  - R = 20 independent groups, N = 1,100 emails/group (100 spam + 1,000 ham)
  - **Track 1**: Synthetic ratio sweep θ = 0%, 10%, ..., 100%
    - Train: (1−θ)×100 real spam + θ×100 synthetic + 1,000 ham
    - Test: 100 real spam + 1,000 ham (fixed, never changes)
  - **Track 2a**: Zero-shot real→synthetic detection (train on real only, test on synthetic)
  - **Track 2b**: Cross-model augmentation (train on LLM-A synthetic, test on LLM-B synthetic)
  - 3 prompt strategies: **Original** (faithful), **Strong** (amplified), **Weak** (subtle)
  - Cross-group mixing strategy (synthetic from group j≠i used with real from group i)
- **Key findings**:
  1. Synthetic spam can partially substitute real training data; quality varies dramatically by model
  2. Zero-shot detection of AI spam achieves moderate baseline; classifier architecture matters
  3. Cross-model augmentation significantly improves detection; diversity > quality

---

## What This Extension Adds

### 1. New Dataset: Enron Spam
- Source: HuggingFace `SetFit/enron_spam` (download to `data/raw/enron/`)
- ~33,716 emails (1999–2002), native ratio ~51/49
- **Experimental ratio: same 1:9 imbalance** as conference (stratified sampling)
- Do NOT change to balanced ratio — the imbalance setting is core to the paper

### 2. New LLM Generators (3 models replacing 2)

| Model | API Backend | Model ID | Cost |
|-------|------------|----------|------|
| GPT-5.4-mini | OpenAI direct | `gpt-5.4-mini` | $0.75/$4.50 per 1M |
| Claude Haiku 4.5 | Anthropic direct | `claude-haiku-4-5-20251001` | $1.00/$5.00 per 1M |
| Gemma 4 26B A4B | **OpenRouter** | `google/gemma-4-26b-a4b-it` | $0.06/$0.33 per 1M |

All models use: `temperature=0.7`, `max_tokens=1000`, `top_p=1.0`

### 3. New Feature Extraction: Sentence-BERT
- Library: `sentence-transformers` (built on HuggingFace + PyTorch, no custom training needed)
- Model: `all-mpnet-base-v2` (768-dim embeddings)
- Device: `mps` (Mac M3 Max) — fallback to `cpu` if MPS issues
- Strategy: **feature extraction only** (no fine-tuning) — same SVM/RF classifiers on SBERT embeddings
- Purpose: **compare TF-IDF vs SBERT** — both are run for RQ-N3

### 4. Execution Strategy
- **Run Track 1 first** (synthetic ratio sweep 0–100%)
- Evaluate Track 1 results before deciding Track 2 scope and naming
- All experiments use Enron only (no CEAS-08 in new experiments)

---

## New Research Questions

- **RQ-N1**: How do newer LLM generations (GPT-5.4-mini, Claude Haiku 4.5) compare to predecessors in spam generation quality?
- **RQ-N2**: Can open-source LLMs (Gemma 4 26B A4B) match proprietary models in generation quality, and what are the cost implications?
- **RQ-N3**: Does Sentence-BERT improve detection of AI-generated spam vs TF-IDF?

---

## Repository Architecture

```
cyberdata-journal/
├── CLAUDE.md                        ← this file (read every session)
├── EXTENSION_PLAN.md                ← full planning document with all decisions
├── pyproject.toml                   ← uv-managed dependencies (NOT poetry)
├── .env                             ← API keys (not committed)
├── .env.public                      ← API key template (committed)
│
├── src/cyberdata/
│   ├── utils/
│   │   ├── llm_invoke.py            ← KEY: supports openai / anthropic / openrouter
│   │   ├── prompt_loader.py
│   │   └── logger_config.py
│   ├── feature_extraction/          ← NEW module (added for journal version)
│   │   ├── __init__.py              ← build_extractor(config) dispatcher
│   │   ├── base_extractor.py        ← abstract base class
│   │   ├── tfidf_extractor.py       ← refactored from step4
│   │   └── sbert_extractor.py       ← NEW: Sentence-BERT wrapper
│   ├── data_processing/
│   ├── llm_generation/
│   ├── classification/
│   └── analysis/
│
├── scripts/
│   ├── step1_data_preprocessing.py  ← supports dataset: enron (and ceas08)
│   ├── step2_llm_generation.py      ← supports openrouter backend
│   ├── step3_dataset_construction.py
│   ├── step4_classification.py      ← feature_type: tfidf | sbert dispatch
│   ├── step5_statistical_analysis.py
│   ├── step6_visualization.py
│   └── run_*.sh                     ← pipeline runner scripts
│
├── config/
│   ├── templates/                   ← original CEAS-08 configs (reference only)
│   ├── enron_gpt54mini_tfidf_v1.yaml
│   ├── enron_claude_haiku45_tfidf_v1.yaml
│   ├── enron_gemma4_26b_tfidf_v1.yaml
│   ├── enron_gpt54mini_sbert_v1.yaml
│   ├── enron_claude_haiku45_sbert_v1.yaml
│   └── enron_gemma4_26b_sbert_v1.yaml
│
├── data/
│   └── raw/enron/                   ← download with: load_dataset('SetFit/enron_spam')
│
├── output/                          ← generated results (not committed)
│
├── paper_conference/                ← ICISSP 2026 paper archive (READ ONLY reference)
│   ├── main.tex
│   ├── sections/
│   └── pic/
│
└── paper/                           ← new journal paper (active writing)
    ├── main.tex
    ├── references.bib               ← must include citation to conference paper
    └── sections/
        ├── abstract.tex             ← MUST differ from conference version
        ├── introduction.tex
        ├── related_work.tex
        ├── methodology.tex          ← add SBERT + Enron subsections
        ├── experiment.tex
        ├── evaluation.tex
        ├── conclusion.tex           ← MUST differ from conference version
        └── differences.tex          ← NEW: required 2-paragraph statement
```

---

## Key Implementation Details

### OpenRouter Backend (for Gemma 4)

```python
# In src/cyberdata/utils/llm_invoke.py
# OpenRouter is OpenAI-compatible; just change base_url and api_key

from openai import OpenAI

LLM_BACKENDS = {
    "gpt-5.4-mini":              "openai",
    "claude-haiku-4-5-20251001": "anthropic",
    "google/gemma-4-26b-a4b-it": "openrouter",
}

def _call_openrouter(model: str, messages: list, **kwargs) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    return response.choices[0].message.content
```

### SBERT Feature Extractor

```python
# In src/cyberdata/feature_extraction/sbert_extractor.py
from sentence_transformers import SentenceTransformer
import numpy as np

class SBERTExtractor:
    def __init__(self, model_name="all-mpnet-base-v2", device="mps"):
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts):
        return self  # pretrained; no fitting needed

    def transform(self, texts):
        return self.model.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )

    def fit_transform(self, texts):
        return self.transform(texts)
```

### Feature Dispatch (step4_classification.py)

```python
# Controlled by config YAML field: feature_extraction.type
def build_extractor(config):
    ftype = config.get("feature_extraction", {}).get("type", "tfidf")
    if ftype == "sbert":
        from src.cyberdata.feature_extraction.sbert_extractor import SBERTExtractor
        return SBERTExtractor(device=config["feature_extraction"].get("device", "mps"))
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
```

### Enron Dataset Loader (step1_data_preprocessing.py)

```python
def load_enron_dataset(path: str) -> pd.DataFrame:
    # HuggingFace disk format
    from datasets import load_from_disk
    ds = load_from_disk(path)
    df = ds["train"].to_pandas()
    # Fields: subject, message, label (0=ham, 1=spam)
    df["text"] = df["subject"].fillna("") + " " + df["message"].fillna("")
    return df[["text", "label"]]
```

---

## Environment

- **Python**: 3.12
- **Package manager**: `uv` (NOT poetry)
  - `uv sync` to install, `uv add <pkg>` to add dependencies
  - `uv run python script.py` or activate `.venv`
- **Hardware**: Mac M3 Max (use `device="mps"` for SBERT)
- **API keys** (in `.env`):
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `OPENROUTER_API_KEY`

---

## ICISSP Journal Requirements (Quick Reference)

| Requirement | Status |
|-------------|--------|
| Cite conference paper | `\cite{icissp2026_original}` in intro + references.bib |
| ≥30% new material | 3 new models + SBERT + Track 2 (TBD) ≈ 45% |
| Reused figures need source citation | Add "(adapted from [icissp2026_original])" in captions |
| No verbatim copying | Use `paper_conference/` as structural reference only |
| New title, abstract, conclusion | All three must be completely rewritten |
| 2-paragraph differences statement | Write last, after all experiments complete |
