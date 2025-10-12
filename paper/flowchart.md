# Methodology Framework Flowchart

This mermaid flowchart visualizes the complete methodology framework comparing traditional feature-space augmentation (SMOTE) and neural text-space generation (LLMs) with systematic replication (R=20).

```mermaid
flowchart TB
    %% Data Source
    A["<b>CEAS-08 Dataset</b><br/>39,154 emails"]

    %% Data Preparation
    subgraph Prep["<b>Data Preparation</b>"]
        B1["Stratified Sampling<br/>(20 groups, 10% spam)"]
        B2["Train-Test Split<br/>(Fixed test set)"]
    end

    %% Two Parallel Paths
    subgraph LLM["<b>LLM Text-Space Generation</b>"]
        C1["Prompt Strategies<br/>(Original/Strong/Weak)"]
        C2["Neural Generation<br/>(GPT-4.1-mini / Claude-3.5-Haiku)"]
        C3["<i>Synthetic Text</i>"]
        C4["TF-IDF Vectorization"]
    end

    subgraph SMOTE["<b>SMOTE Feature-Space Interpolation</b>"]
        D1["TF-IDF Vectorization<br/>(Real samples)"]
        D2["K-NN Interpolation<br/>(k=5 neighbors)"]
        D3["<i>Synthetic Vectors</i>"]
    end

    %% Mixing and Evaluation
    subgraph Mix["<b>Mixing & Training</b>"]
        E1["Mixing Strategies<br/>(Within-Group / Cross-Group)"]
        E2["Synthetic Ratios<br/>(0%, 10%, ..., 100%)"]
    end

    subgraph Eval["<b>Evaluation</b>"]
        F1["Classification<br/>(SVM / Random Forest)"]
        F2["Performance Metrics<br/>(F1, Accuracy, AUC-ROC, ...)"]
        F3["Statistical Analysis<br/>(t-tests, FDR, Cohen's d, Regression)"]
    end

    G["<b>Results & Insights</b><br/>Architecture sensitivity<br/>Paradigm comparison<br/>Deployment guidance"]

    %% Flow connections
    A --> Prep
    Prep --> B1 --> B2

    B2 --> LLM
    B2 --> SMOTE

    C1 --> C2 --> C3 --> C4 --> Mix
    D1 --> D2 --> D3 --> Mix

    Mix --> E1 --> E2 --> Eval
    Eval --> F1 --> F2 --> F3 --> G

    %% Styling
    classDef dataStyle fill:#E3F2FD,stroke:#1976D2,stroke-width:2px,color:#000
    classDef prepStyle fill:#FFF3E0,stroke:#F57C00,stroke-width:2px,color:#000
    classDef llmStyle fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#000
    classDef smoteStyle fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000
    classDef mixStyle fill:#FFF9C4,stroke:#F9A825,stroke-width:2px,color:#000
    classDef evalStyle fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
    classDef resultStyle fill:#ECEFF1,stroke:#455A64,stroke-width:3px,color:#000

    class A dataStyle
    class Prep,B1,B2 prepStyle
    class LLM,C1,C2,C3,C4 llmStyle
    class SMOTE,D1,D2,D3 smoteStyle
    class Mix,E1,E2 mixStyle
    class Eval,F1,F2,F3 evalStyle
    class G resultStyle
```

## Key Components

### Data Preparation
- **Stratified Sampling**: 20 independent groups, each with 1,000 samples (10% spam)
- **Fixed Test Set**: Consistent evaluation across all experiments

### Paradigm Comparison

**LLM Text-Space Path:**
1. Prompt engineering (3 strategies)
2. Neural generation (2 models: GPT-4.1-mini, Claude-3.5-Haiku)
3. Produces interpretable synthetic text
4. Vectorization applied to synthetic text

**SMOTE Feature-Space Path:**
1. Vectorization applied to real samples first
2. K-nearest neighbors interpolation (k=5)
3. Produces abstract synthetic feature vectors (no text)

### Experimental Design
- **Mixing Strategies**: Within-group vs Cross-group
- **Synthetic Ratios**: 11 levels (0%, 10%, 20%, ..., 100%)
- **Total Configurations**: 1,760 LLM + 440 SMOTE = 2,200 experiments

### Evaluation Framework
- **Classifiers**: SVM (margin-based) vs Random Forest (ensemble-based)
- **Metrics**: 7 classification metrics (F1 as primary)
- **Statistical Analysis**: Hypothesis testing, effect sizes, sensitivity regression
