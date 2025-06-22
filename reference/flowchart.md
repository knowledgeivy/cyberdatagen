┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    INPUT DATA                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Raw CSV Files (e.g., email_phishing.csv.gz, nsl_kdd_rare.csv.gz)                 │
│  └─ Malicious/Benign samples with labels                                           │
│  Data Schema Definitions (data_info.yaml)                                          │
│  └─ Column types, label encoding, domain metadata                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PHASE 1: REAL-WORLD ANALYSIS                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
         ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
         │   STEP 1        │  │   STEP 2        │  │   STEP 3        │
         │ Domain          │  │ Contextual      │  │ Schema-Aware    │
         │ Discovery       │  │ Enrichment      │  │ Seeds           │
         └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
        │• Attack       │      │• Problem        │      │• High-quality   │
        │  Patterns     │      │  Context        │      │  Examples       │
        │• Normal       │ ────▶│• Threat         │ ────▶│• Schema         │
        │  Baselines    │      │  Intelligence   │      │  Compliant      │
        │• Technical    │      │• Risk           │      │• Domain         │
        │  Indicators   │      │  Assessment     │      │  Accurate       │
        └───────────────┘      └─────────────────┘      └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
        │ OUTPUT:       │      │ OUTPUT:         │      │ OUTPUT:         │
        │ domain_       │      │ contextual_     │      │ seeds-raw/      │
        │ discovery.json│      │ problems.json   │      │ area/nature_    │
        │               │      │ attack_         │      │ examples.json   │
        │               │      │ taxonomy.json   │      │                 │
        └───────────────┘      └─────────────────┘      └─────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            VALIDATION & FILTERING                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Parallel Multi-Dimensional Quality Assessment                                     │
│  └─ Technical Accuracy, Schema Consistency, Realism, Uniqueness, Domain Alignment │
│                                                                                     │
│  OUTPUT: seeds-validated/ (high-quality) + seeds-filtered/ (low-quality)          │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                             PHASE 2: SCALE GENERATION                              │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
         ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
         │ Context         │  │ Parallel        │  │ Quality         │
         │ Integration     │  │ Generation      │  │ Control         │
         └─────────────────┘  └─────────────────┘  └─────────────────┘
                │                       │                       │
                ▼                       ▼                       ▼
        ┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
        │• Domain       │      │• Configurable   │      │• Duplicate      │
        │  Discovery    │      │  Scale Count    │      │  Detection      │
        │• Contextual   │ ────▶│• Malicious      │ ────▶│• Schema         │
        │  Problems     │      │  Ratio Control  │      │  Validation     │
        │• Validated    │      │• Batch          │      │• Quality        │
        │  Seeds        │      │  Processing     │      │  Filtering      │
        └───────────────┘      └─────────────────┘      └─────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │ OUTPUT:         │
                              │ scaled-raw/     │
                              │ area/nature_    │
                              │ scale.json      │
                              │ (1000s samples) │
                              └─────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           FINAL VALIDATION & DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Production-Ready Quality Assessment                                               │
│  └─ Full/Statistical Validation, Performance Metrics, Deployment Readiness        │
│                                                                                     │
│  OUTPUT: scaled-validated/ (production-ready) + validation reports                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL OUTPUTS                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  Production Datasets: Large-scale, high-quality synthetic cybersecurity data      │
│  Quality Reports: Comprehensive validation metrics and deployment guidance        │
│  Mixed Datasets: Optional real + synthetic combinations                            │
│  Configuration: Reproducible settings for future generation                        │
└─────────────────────────────────────────────────────────────────────────────────────┘