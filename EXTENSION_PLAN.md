# Extension Plan: ICISSP 2026 → Journal Post-Publication

**Original paper**: "Can Synthetic Spam Beat Real-World Detectors? Evaluating LLMs' Dual Role in Spam Generation and Detection"
**Target**: Extended journal version (≥30% new material)
**Acceptance deadline**: 2026-06-05 | **Submission deadline**: 2026-07-08
**Execution strategy**: Track 1 first → evaluate results → decide Track 2 scope

---

## 1. What Changes (Summary)

| Dimension | Conference Version | Extended Version |
|-----------|-------------------|-----------------|
| Dataset | CEAS-08 | **Enron Spam** (replace entirely) |
| LLM Generators | GPT-4.1-mini, Claude-3.5-Haiku | **GPT-5.4-mini, Claude Haiku 4.5, Gemma 4 26B A4B** |
| Feature Extraction | TF-IDF only | TF-IDF **+ Sentence-BERT** (comparison) |
| Classifiers | SVM, Random Forest | SVM, Random Forest (same, on both feature sets) |
| Class ratio | 1:9 imbalance | **1:9 imbalance (unchanged)** |
| Tracks | Track 1, 2a, 2b | **Track 1 first**; Track 2 decided after results |
| Package manager | Poetry | **uv** |
| Repository | `cyberdata` (current) | **New repo** `cyberdata-journal` |

**New Research Questions:**
- **RQ-N1**: How do newer LLM generations (GPT-5.4-mini, Claude Haiku 4.5) compare to their predecessors in spam generation quality on a new corpus?
- **RQ-N2**: Can open-source LLMs (Gemma 4 26B A4B) achieve comparable generation quality to proprietary models, and what are the cost-performance implications?
- **RQ-N3**: Does semantic feature representation (Sentence-BERT) improve detection of AI-generated spam compared to lexical features (TF-IDF)?

---

## 2. Repository Setup

### 2.1 Decision: New Repo

**Use a new repo `cyberdata-journal`**, not a branch in the current repo.

| | New Branch | New Repo ✓ |
|--|-----------|------------|
| Git history | Mixed with conference work | Clean slate |
| Published paper reference | Complicated | Original repo stays intact as artifact |
| LaTeX management | Confusing | Separate archive vs. active paper |
| Sharing / citation | Must specify branch | Independent URL |

### 2.2 Tag Current Repo Before Migrating

```bash
# In current cyberdata repo
cd /Users/tianyu/Notebooks/cyberdata
git checkout dev
git tag v1.0-icissp2026
git push origin v1.0-icissp2026
```

### 2.3 Create New Repo

```bash
# On GitHub: create new repo "cyberdata-journal" (private), then:
mkdir ~/Notebooks/cyberdata-journal
cd ~/Notebooks/cyberdata-journal
git init
git remote add origin git@github.com:<username>/cyberdata-journal.git
```

---

## 3. Migration: What to Copy

### 3.1 Files to Copy (manually or via script)

```
FROM: ~/Notebooks/cyberdata/
TO:   ~/Notebooks/cyberdata-journal/

COPY:
  src/                          → src/                    (core library, will be modified)
  scripts/                      → scripts/                (pipeline scripts, will be modified)
  config/                       → config/templates/       (as reference templates only)
  paper/                        → paper_conference/       (archive of conference version)
  .gitignore                    → .gitignore
  .env.public                   → .env.public             (API key template)
  README.md                     → README.md               (rewrite for journal version)

DO NOT COPY:
  data/                         (too large; re-download Enron fresh)
  output/                       (too large; regenerate from new experiments)
  .venv/                        (recreate with uv)
  pyproject.toml                (recreate with uv)
  poetry.lock                   (not needed; using uv)
  raw/                          (re-download Enron)
  .git/                         (new repo has its own)
```

### 3.2 Migration Shell Script

Save as `migrate_to_journal.sh` in current repo, run once:

```bash
#!/bin/bash
SRC=~/Notebooks/cyberdata
DST=~/Notebooks/cyberdata-journal

# Core code
cp -r $SRC/src $DST/src
cp -r $SRC/scripts $DST/scripts

# Config as templates (not active configs)
mkdir -p $DST/config/templates
cp $SRC/config/*.yaml $DST/config/templates/

# Conference paper as archive reference
cp -r $SRC/paper $DST/paper_conference

# Dotfiles
cp $SRC/.gitignore $DST/.gitignore
cp $SRC/.env.public $DST/.env.public

echo "Migration complete. Do NOT copy data/, output/, .venv/"
```

### 3.3 LaTeX: Why Copy the Conference Paper

The conference LaTeX goes into `paper_conference/` for three reasons:

1. **ICISSP requirement**: The journal paper must cite the conference version — you need the original text as reference for what counts as "verbatim" vs. rewritten
2. **Framework reuse**: Methodology, Related Work, and Experiment setup sections serve as structural templates that you rewrite rather than start from scratch
3. **Figure/table source**: ICISSP requires adding "(adapted from [conf_citation])" to all reused figures — you need the originals to know which ones to cite

New `paper/` directory in the journal repo starts fresh:
```
paper_conference/       ← archived original (READ ONLY reference)
paper/                  ← new journal version (active writing)
  main.tex
  references.bib        ← includes citation to conference paper
  sections/
    abstract.tex        ← must be different from conference
    introduction.tex    ← new RQs added
    related_work.tex    ← can reuse structure, must rewrite prose
    methodology.tex     ← add SBERT + Enron subsections
    experiment.tex      ← new model/dataset configs
    evaluation.tex      ← new results sections
    conclusion.tex      ← must be different from conference
    differences.tex     ← new: required 2-paragraph statement
```

---

## 4. Environment: uv (replacing Poetry)

### 4.1 Why uv

| | Poetry | uv |
|--|--------|-----|
| Dependency resolution | Minutes | Seconds (10-100× faster) |
| Status (2025-2026) | Stable but declining | New Python standard |
| pyproject.toml compatible | Yes | Yes |
| M3 Mac support | Good | Good + automatic MPS detection |

### 4.2 Setup in New Repo

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/Notebooks/cyberdata-journal

# Initialize project
uv init --no-readme
uv python pin 3.12

# Add all dependencies
uv add numpy pandas scikit-learn openai anthropic \
       sentence-transformers datasets huggingface-hub \
       pyyaml tqdm scipy matplotlib seaborn \
       plotly jinja2 python-dotenv

# Dev dependencies
uv add --dev pytest ruff
```

### 4.3 Key Command Differences

```bash
# Poetry → uv
poetry install          →  uv sync
poetry add package      →  uv add package
poetry shell            →  source .venv/bin/activate
poetry run python X     →  uv run python X
```

### 4.4 New `.env` File

```bash
# .env (copy from .env.public and fill in)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
# GOOGLE_API_KEY not needed; Gemma 4 accessed via OpenRouter
```

---

## 5. AI Session Continuity: CLAUDE.md

### 5.1 The Problem

When you open `cyberdata-journal` as a new Claude Code workspace, a new session starts with **no memory of this planning conversation**. Claude Code will see a fresh codebase with no context about the ICISSP paper, the extension goals, or the decisions made.

### 5.2 The Solution: CLAUDE.md

Claude Code **automatically reads `CLAUDE.md` at the start of every session**. This file acts as persistent project context — equivalent to a permanent system prompt for all future sessions in this workspace.

Create `~/Notebooks/cyberdata-journal/CLAUDE.md` with the content below. Any new Claude Code session opening this repo will immediately understand:
- What this project is and where it comes from
- What the conference paper did
- What the extension adds and why
- Key architectural decisions
- Current status

### 5.3 CLAUDE.md Template (copy to new repo)

```markdown
# CLAUDE.md — cyberdata-journal

## Project Context

This is a **journal post-publication extension** of a conference paper published at
ICISSP 2026: "Can Synthetic Spam Beat Real-World Detectors? Evaluating LLMs' Dual
Role in Spam Generation and Detection".

The conference version is archived in `paper_conference/` and in the GitHub repo
`cyberdata` (tag: v1.0-icissp2026). This repo contains the extended journal version.

**Journal submission deadline**: 2026-07-08

---

## What the Conference Paper Did

- **Dataset**: CEAS-08 spam email corpus (2008), 1:9 spam/ham imbalance
- **LLM Generators**: GPT-4.1-mini, Claude-3.5-Haiku
- **Features**: TF-IDF (max 50k features, 1-2 ngrams)
- **Classifiers**: SVM, Random Forest
- **Experimental design**:
  - 20 independent groups, 1100 emails/group (100 spam + 1000 ham)
  - Track 1: Synthetic ratio sweep (0%–100%, 11 points)
  - Track 2a: Zero-shot real→synthetic detection
  - Track 2b: Cross-model augmentation (A trains on B's synthetic, tests on A's)
  - 3 prompt strategies: Original, Strong, Weak
  - Cross-group mixing strategy
- **Key findings**:
  1. Synthetic spam can partially substitute real training data; quality varies by model
  2. Zero-shot detection of AI spam achieves moderate baseline with classifier-dependent variation
  3. Cross-model augmentation significantly improves detection; diversity > quality

---

## What This Extension Adds

### New Dataset
- **Enron Spam** (1999–2002, ~33,716 emails)
- Same 1:9 experimental imbalance as conference version (stratified sampling)
- Source: HuggingFace `SetFit/enron_spam` or GitHub CSV
- Replaces CEAS-08 entirely in new experiments

### New LLM Generators
| Model | API | Notes |
|-------|-----|-------|
| GPT-5.4-mini | OpenAI direct | Replaces GPT-4.1-mini |
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Anthropic direct | Replaces Claude-3.5-Haiku |
| Gemma 4 26B A4B (`google/gemma-4-26b-a4b-it`) | OpenRouter | New: open-source MoE model |

### New Feature Extraction
- **Sentence-BERT** (`all-mpnet-base-v2`) via `sentence-transformers` library
- Used as **drop-in replacement** for TF-IDF (same SVM/RF classifiers)
- Device: MPS (Mac M3 Max) — set `device="mps"` in SBERTExtractor
- Comparison: TF-IDF vs SBERT on same experimental conditions

### Execution Strategy
- **Track 1 first** (synthetic ratio sweep), then decide Track 2 scope based on results
- All new experiments use Enron only

---

## Architecture

```
cyberdata-journal/
├── CLAUDE.md                    ← this file (always read first)
├── EXTENSION_PLAN.md            ← full planning document
├── src/cyberdata/
│   ├── utils/
│   │   ├── llm_invoke.py        ← supports openai / anthropic / openrouter backends
│   │   ├── prompt_loader.py
│   │   └── logger_config.py
│   ├── feature_extraction/      ← NEW module
│   │   ├── base_extractor.py
│   │   ├── tfidf_extractor.py
│   │   └── sbert_extractor.py
│   ├── data_processing/
│   ├── llm_generation/
│   ├── classification/
│   └── analysis/
├── scripts/
│   ├── step1_data_preprocessing.py   ← supports "enron" dataset
│   ├── step2_llm_generation.py       ← supports openrouter backend
│   ├── step3_dataset_construction.py
│   ├── step4_classification.py       ← feature_type: tfidf | sbert
│   ├── step5_statistical_analysis.py
│   └── step6_visualization.py
├── config/
│   ├── templates/               ← original CEAS-08 configs (reference only)
│   └── *.yaml                   ← active Enron experiment configs
├── data/
│   └── raw/enron/               ← download fresh
├── output/
├── paper_conference/            ← archived ICISSP 2026 paper (READ ONLY)
└── paper/                       ← new journal paper (active)
```

---

## Key Implementation Notes

### OpenRouter (Gemma 4)
```python
# llm_invoke.py — OpenRouter uses OpenAI-compatible API
from openai import OpenAI
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
# Model ID: "google/gemma-4-26b-a4b-it"
```

### SBERT Feature Extraction
```python
# src/cyberdata/feature_extraction/sbert_extractor.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2", device="mps")
embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True)
# Returns (n, 768) numpy array — feed directly to SVM/RF
```

### Feature Dispatch (step4)
Controlled by config YAML:
```yaml
feature_extraction:
  type: "tfidf"   # or "sbert"
  model: "all-mpnet-base-v2"   # only used when type=sbert
  device: "mps"
```

### Class Imbalance
Always use 1:9 (spam:ham) — same as conference paper. Do NOT change to balanced ratio.

---

## ICISSP Journal Requirements

| Requirement | Status |
|-------------|--------|
| Cite conference paper | Add `\cite{icissp2026_original}` to references.bib |
| ≥30% new material | New dataset + 3 new models + SBERT = ~40-50% new |
| Figures from conference need source citation | Add "(adapted from [icissp2026_original])" |
| No verbatim copying | Rewrite all prose; keep structure only |
| New title, abstract, conclusion | All three must differ from conference version |
| 2-paragraph differences statement | Write after experiments complete |
```

---

## 6. Dataset: Enron Spam

### 6.1 Overview

| Attribute | Value |
|-----------|-------|
| Source | HuggingFace: `SetFit/enron_spam` |
| Native ratio | ~51% spam / 49% ham |
| Experimental ratio | **1:9** (100 spam : 900 ham per group, same as conference) |
| Groups | 20 independent groups |
| Group size | 1,100 emails |
| Total emails needed | ~33,000 (dataset has ~33,716 — sufficient) |

### 6.2 Download

```bash
# In new repo, after uv sync:
python -c "
from datasets import load_dataset
ds = load_dataset('SetFit/enron_spam')
ds.save_to_disk('data/raw/enron')
print('Done:', ds)
"
# Or download CSV directly:
# https://github.com/MWiechmann/enron_spam_data → enron_spam_data.csv
```

### 6.3 Column Mapping

```
HuggingFace fields:
  "subject"    → subject
  "message"    → body
  "label"      → 1=spam, 0=ham (already numeric)

CSV fields (GitHub version):
  "Subject"    → subject
  "Message"    → body
  "Spam/Ham"   → "spam"→1, "ham"→0
```

---

## 7. New LLM Models

### 7.1 Model Selection

| Model | API Backend | Model ID | Input/1M | Output/1M |
|-------|------------|----------|----------|----------|
| GPT-5.4-mini | OpenAI | `gpt-5.4-mini` | $0.75 | $4.50 |
| Claude Haiku 4.5 | Anthropic | `claude-haiku-4-5-20251001` | $1.00 | $5.00 |
| Gemma 4 26B A4B | OpenRouter | `google/gemma-4-26b-a4b-it` | $0.06 | $0.33 |

All models: `temperature=0.7`, `max_tokens=1000`, `top_p=1.0`

### 7.2 Cost Estimate

Per model per dataset: 20 groups × 100 emails × 3 prompts × ~500 tokens = 3M tokens

| Model | Cost per run | 3 models × 1 dataset |
|-------|-------------|---------------------|
| GPT-5.4-mini | ~$15–20 | |
| Claude Haiku 4.5 | ~$18–25 | |
| Gemma 4 26B A4B | ~$1–2 | |
| **Total** | | **~$35–50** |

### 7.3 OpenRouter Integration

```python
# src/cyberdata/utils/llm_invoke.py

LLM_BACKENDS = {
    "gpt-5.4-mini":               "openai",
    "claude-haiku-4-5-20251001":  "anthropic",
    "google/gemma-4-26b-a4b-it":  "openrouter",
}

def _call_openrouter(model: str, messages: list, **kwargs) -> str:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    return response.choices[0].message.content
```

---

## 8. Sentence-BERT Integration

### 8.1 Why TF-IDF vs SBERT Comparison (not just SBERT alone)

- SBERT without TF-IDF baseline has no reference point → weak contribution
- The comparison directly answers RQ-N3 and adds a clean new section to the paper
- TF-IDF results from new experiments serve as baseline; SBERT is the new treatment
- Both use same SVM/RF classifiers → fair comparison

### 8.2 Feature Extractor Module

```python
# src/cyberdata/feature_extraction/base_extractor.py
from abc import ABC, abstractmethod
import numpy as np

class BaseExtractor(ABC):
    @abstractmethod
    def fit(self, texts: list[str]): ...
    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray: ...
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
```

```python
# src/cyberdata/feature_extraction/sbert_extractor.py
from sentence_transformers import SentenceTransformer
import numpy as np
from .base_extractor import BaseExtractor

class SBERTExtractor(BaseExtractor):
    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "mps"):
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts: list[str]):
        return self  # pretrained; no fitting needed

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
```

```python
# src/cyberdata/feature_extraction/__init__.py
def build_extractor(config: dict):
    cfg = config.get("feature_extraction", {})
    if cfg.get("type", "tfidf") == "sbert":
        from .sbert_extractor import SBERTExtractor
        return SBERTExtractor(
            model_name=cfg.get("model", "all-mpnet-base-v2"),
            device=cfg.get("device", "mps"),
        )
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(
            max_features=cfg.get("max_features", 50000),
            ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
            sublinear_tf=True,
        )
```

### 8.3 SBERT Config

```yaml
feature_extraction:
  type: "sbert"
  model: "all-mpnet-base-v2"
  device: "mps"          # Mac M3 Max; fallback to "cpu" if issues
```

---

## 9. Experimental Design

### 9.1 Strategy: Track 1 First

Run Track 1 completely → analyze results → decide Track 2 scope and framing.

**Track 1**: Synthetic ratio sweep (θ = 0%, 10%, ..., 100%)
- Training: (1−θ)×100 real spam + θ×100 synthetic spam + 900 ham
- Testing: 100 real spam + 900 ham (fixed)

Track 2 decision points (after Track 1):
- If cross-model transfer is interesting → run Track 2b (3-model version with 6 pairs)
- If zero-shot detection varies interestingly → run Track 2a
- If results are decisive → possibly skip Track 2 entirely

### 9.2 Experiment Blocks

| Block | Dataset | LLM | Features | Track |
|-------|---------|-----|----------|-------|
| **B1** | Enron | GPT-5.4-mini | TF-IDF | Track 1 |
| **B2** | Enron | Claude Haiku 4.5 | TF-IDF | Track 1 |
| **B3** | Enron | Gemma 4 26B A4B | TF-IDF | Track 1 |
| **B4** | Enron | GPT-5.4-mini | SBERT | Track 1 |
| **B5** | Enron | Claude Haiku 4.5 | SBERT | Track 1 |
| **B6** | Enron | Gemma 4 26B A4B | SBERT | Track 1 |
| **B7–B9** | Enron | All 3 models | TF-IDF | Track 2 (TBD) |

### 9.3 Config Files to Create

```
config/
  enron_gpt54mini_tfidf_v1.yaml
  enron_claude_haiku45_tfidf_v1.yaml
  enron_gemma4_26b_tfidf_v1.yaml
  enron_gpt54mini_sbert_v1.yaml
  enron_claude_haiku45_sbert_v1.yaml
  enron_gemma4_26b_sbert_v1.yaml
```

Each config controls: dataset, llm, feature_extraction, classification, experiment, output.

---

## 10. Execution Timeline

### Phase 0: Migration & Setup (May 13–17)
- [ ] Tag current repo: `git tag v1.0-icissp2026 && git push origin v1.0-icissp2026`
- [ ] Create new GitHub repo `cyberdata-journal`
- [ ] Run `migrate_to_journal.sh` (or manually copy)
- [ ] `uv init` + `uv add` all dependencies
- [ ] Copy CLAUDE.md template (Section 5.3) into new repo root
- [ ] Create `.env` with all 3 API keys
- [ ] Download Enron dataset
- [ ] Create all 6 config files

### Phase 1: Code Updates (May 17–20)
- [ ] Add OpenRouter backend to `llm_invoke.py`
- [ ] Add Enron loader to `step1_data_preprocessing.py`
- [ ] Create `src/cyberdata/feature_extraction/` module
- [ ] Update feature dispatch in `step4_classification.py`
- [ ] Smoke test each component (10 emails, 2 groups)

### Phase 2: Data & Generation (May 20–27)
- [ ] `step1` Enron: generate 20 groups (~1 hour)
- [ ] `step2` Gemma 4 generation — run first (cheapest, validates pipeline)
- [ ] `step2` GPT-5.4-mini + Claude Haiku 4.5 (parallel if budget allows)

### Phase 3: Classification — Track 1 (May 27 – Jun 5)
- [ ] B1–B3: TF-IDF classification (all 3 models, parallel)
- [ ] B4–B6: SBERT classification (all 3 models, parallel)
- [ ] Merge results, run `step5` statistical analysis
- [ ] Quick review: are results interesting? Decide Track 2 scope.

### Phase 4: Track 2 (if needed, Jun 5–12)
- [ ] Track 2 experiments based on Track 1 findings
- [ ] Statistical analysis + visualization

### Phase 5: Full Analysis & Visualization (Jun 12–19)
- [ ] Complete `step5` + `step6` for all blocks
- [ ] New cross-model comparison plots (TF-IDF vs SBERT, 3 LLMs)
- [ ] Cost analysis table (proprietary vs open-source)

### Phase 6: Paper Writing (Jun 19 – Jul 5)
- [ ] New abstract, title, conclusion (all required to differ)
- [ ] Rewrite methodology (add SBERT + Enron subsections)
- [ ] New results sections for each RQ
- [ ] "Differences from Conference Version" 2-paragraph statement
- [ ] Add `\cite{icissp2026_original}` to all reused figures/tables
- [ ] Final proofread + submit

---

## 11. ICISSP Journal Requirements Checklist

| Requirement | Action |
|-------------|--------|
| Cite conference paper | `\cite{icissp2026_original}` in intro + references.bib |
| ≥30% new material | B1–B6 (new models + SBERT) + Track 2 = ~45% new content |
| Figures from conference need source citation | Add caption note: "(adapted from [icissp2026_original])" |
| No large verbatim blocks | Rewrite all prose; use `paper_conference/` as structural reference only |
| New title | e.g., "Generative AI in Spam Detection: Multi-Model, Multi-Feature Evaluation with Open-Source LLMs" |
| New abstract | Completely new; cover Enron, Gemma 4, SBERT results |
| New conclusion | Completely new; synthesize extended findings |
| 2-paragraph differences statement | Draft after experiments complete (template in Section 12) |

---

## 12. "Differences from Conference Version" Draft Template

> The extended version expands the ICISSP 2026 conference paper across three main dimensions. First, the evaluation adopts the Enron corpus as the primary benchmark dataset, replacing the CEAS-08 dataset used in the conference version. This shift enables assessment of whether findings generalize beyond a single corpus and provides a widely-cited reference benchmark for comparison with prior work. Second, we expand the LLM generator set from two proprietary models (GPT-4.1-mini, Claude-3.5-Haiku) to three models that include an open-source alternative (Gemma 4 26B A4B via OpenRouter), allowing direct cost-performance comparison between proprietary and open-source approaches to synthetic spam generation. Third, we introduce Sentence-BERT as a semantic feature representation alongside the original TF-IDF features, enabling systematic comparison of lexical versus contextual embeddings for detecting AI-generated spam.
>
> These extensions collectively address the limitations explicitly acknowledged in the conference paper: single-corpus evaluation, absence of open-source LLM comparison, and exclusive reliance on lexical features. The extended version introduces three new research questions, six new experiment blocks, and updated statistical analyses including cross-model and cross-feature comparisons. The methodology, results, and discussion sections are substantially rewritten to incorporate the expanded scope, and the title, abstract, and conclusions are entirely new. All figures and tables carried over from the conference version are clearly attributed to the original publication.

---

## 13. Key Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Gemma 4 generation quality poor | Medium | Run 50-email pilot before full generation |
| SBERT not better than TF-IDF | Medium | Valid finding; honest negative results are publishable |
| OpenRouter rate limits | Medium | Use paid tier; add exponential backoff |
| MPS backend issues with SBERT | Low | Set `device="cpu"` as fallback |
| Enron emails too short (subject only) | Low | Concatenate subject + body; check avg token count |
