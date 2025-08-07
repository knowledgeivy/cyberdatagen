# CyberData: Synthetic Cybersecurity Dataset Generator

CyberData extracts patterns from real cybersecurity datasets and generates synthetic data that preserves technical accuracy and schema consistency. The system uses a three-step pipeline to ensure quality and production readiness.

## Key Features

- Real-world data pattern extraction
- Schema-preserving synthetic generation
- Multi-dimensional quality validation
- Large-scale production generation
- Domain-specific intelligence integration

## Research Experiment: Batch 6

The `batch6` experiment is a comprehensive research pipeline designed to generate high-quality synthetic data and evaluate its impact on machine learning model performance. The experiment is structured into seven distinct phases, from seed sample preparation to model training and performance visualization.

### Research Goal

The primary goal of the `batch6` experiment is to investigate whether high-quality synthetic data can be used to train robust machine learning models for malicious content detection, achieving performance comparable to or better than models trained on real data.

### Workflow: The 7 Phases of Batch 6

The `batch6` experiment follows a structured 7-phase workflow:

1.  **Phase 1: Enhanced Seed Preparation:** Selects a diverse set of 2,000 high-quality "seed" samples from a larger dataset, using a 2-layer stratification method (core and edge) to ensure representation of the full spectrum of malicious data.

2.  **Phase 2: High-Quality Synthetic Data Generation:** Generates 6,000 synthetic samples from the 2,000 seeds using a sophisticated "real_data_rewriter.py" methodology with three different prompt strategies (original, strong, and weak).

3.  **Phase 3: Synthetic Data ID Annotation:** Assigns unique, traceable IDs to all synthetic samples, linking them back to their original seed, layer, and prompt strategy.

4.  **Phase 4: Unified Embedding Space Construction:** Creates a unified embedding space for all data (seeds, synthetic, background, etc.) using the `all-MiniLM-L6-v2` model to represent text data in a way that captures its meaning.

5.  **Phase 5 & 5a/5b: Enhanced Visualization and Analysis:** Performs a comprehensive visualization of the embedding space to analyze the quality and distribution of the synthetic data, the effects of different prompt strategies, and the quality of data clusters.

6.  **Phase 6: Pure Dataset Construction:** Constructs 16 "pure" datasets with varying proportions of malicious data (5%, 10%, 15%, 20%) and different types of malicious data (real only, or purely synthetic from each of the three prompt strategies).

7.  **Phase 7a & 7b: Model Training and Performance Evaluation:** Trains three different machine learning models (RandomForest, SVM, and a Deep Learning model) on each of the 16 pure datasets and evaluates their performance on a fixed, independent test set.

## Installation

### Requirements
- Python 3.12+
- OpenAI API key (GPT-4.1-mini)

### Setup
```bash
# Clone repository
git clone <repository_url>
cd cyberdata

# Install with Poetry
poetry install

# Or install with pip
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OpenAI API key to .env
```

## Usage

To run the `batch6` experiment, you can execute the scripts in the `src/cyberdata/scripts/` directory in the order of the phases. For example:

```bash
# Run Phase 1: Enhanced Seed Preparation
python src/cyberdata/scripts/batch6_phase1_seed_preparation.py

# Run Phase 2: High-Quality Synthetic Data Generation
python src/cyberdata/scripts/batch6_phase2_synthetic_generation.py

# ... and so on for the remaining phases.
```

## File Structure

The `batch6` experiment generates a comprehensive set of artifacts in the `data/batch6/` directory:

```
cyberdata/
├── src/cyberdata/
│   ├── scripts/
│   │   ├── batch6_phase1_seed_preparation.py
│   │   ├── batch6_phase2_synthetic_generation.py
│   │   ├── batch6_phase3_synthetic_id_annotation.py
│   │   ├── batch6_phase4_unified_embedding.py
│   │   ├── batch6_phase5a_data_processing.py
│   │   ├── batch6_phase5b_visualization.py
│   │   ├── batch6_phase6_pure_dataset_construction.py
│   │   ├── batch6_phase7a_model_training.py
│   │   └── batch6_phase7b_visualization.py
│   └── utils/
├── config/
├── data/
│   └── batch6/
│       ├── phase1_analysis/
│       ├── phase2_analysis/
│       ├── phase3_analysis/
│       ├── phase4_analysis/
│       ├── phase5a_processed_data/
│       ├── phase5b_analysis/
│       ├── phase7a_training/
│       ├── phase7b_analysis/
│       ├── pure_datasets/
│       ├── seed_preparation/
│       ├── synthetic_generation/
│       ├── synthetic_with_ids/
│       └── unified_embeddings/
├── raw/
└── logs/
```

## Output

The `batch6` experiment generates a wide range of outputs, including:

-   **High-quality synthetic data:** `data/batch6/synthetic_with_ids/`
-   **Unified embeddings:** `data/batch6/unified_embeddings/`
-   **Pure datasets for model training:** `data/batch6/pure_datasets/`
-   **Trained machine learning models:** `data/batch6/phase7a_training/models/`
-   **Comprehensive performance results and visualizations:** `data/batch6/phase7b_analysis/`

## Version

Current version: 2.1.6

## License

MIT License - see LICENSE file for details.

## Contact

knowledgeivy01@gmail.com
