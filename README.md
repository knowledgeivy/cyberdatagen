# CyberData: Advanced Synthetic Cybersecurity Dataset Generator

A comprehensive Python toolkit for generating high-quality synthetic cybersecurity datasets with domain-specific intelligence, schema preservation, and configurable attack scenarios. CyberData bridges the gap between limited real-world security data and the extensive datasets needed for robust machine learning model development and evaluation.

## 🎯 Project Overview

CyberData provides an intelligent, multi-modal approach to synthetic cybersecurity data generation that combines manual threat taxonomy definition with real-world data enhancement. The system leverages large language models (LLMs) with extensive cybersecurity domain knowledge to create realistic, technically accurate datasets for training and evaluating security-focused ML models.

### Key Innovations

- **Domain-Enhanced Generation**: Integrates comprehensive cybersecurity knowledge from `data_info.yaml` for authentic threat modeling
- **Schema Preservation**: Maintains exact data structure compatibility with existing ML pipelines
- **Intelligent Data Mixing**: Automatically combines real-world and synthetic data with schema alignment
- **Configurable Attack Ratios**: Supports realistic class imbalance scenarios (0% to 50+ attack prevalence)
- **Parallel Processing**: High-performance generation with configurable workers and batch processing
- **Quality Assurance**: Multi-layered validation including LLM-based technical accuracy assessment

### Supported Workflows

#### Workflow A: Expert-Driven Threat Modeling
1. **Define cybersecurity problems** using expert knowledge and threat frameworks
2. **Generate seed datasets** with LLM-enhanced technical examples
3. **Scale to large datasets** using parallel generation with quality controls
4. **Validate outputs** with domain-specific metrics and technical accuracy checks

#### Workflow B: Real-World Data Enhancement
1. **Analyze existing datasets** with comprehensive domain context from `data_info.yaml`
2. **Generate schema-preserving seeds** that maintain ML pipeline compatibility
3. **Scale intelligently** while preserving original data characteristics and distributions
4. **Mix with real data** using intelligent schema alignment for hybrid datasets

## 📂 Project Architecture

```
cyberdata/
├── src/cyberdata/
│   ├── process/                     # Core pipeline modules
│   │   ├── problem_generator.py     # Expert threat taxonomy generation
│   │   ├── problem_enhancer.py      # Threat landscape evaluation & extension
│   │   ├── seed_generator.py        # Technical seed example creation
│   │   ├── realworld_processor.py   # Real-world data analysis & enhancement
│   │   ├── synthetic_generator.py   # Parallel synthetic dataset generation
│   │   ├── seed_validator.py        # Technical accuracy validation
│   │   ├── dataset_validator.py     # Large-scale quality assessment
│   │   └── data_mixer.py            # Intelligent data mixing & alignment
│   └── utils/                       # Core utilities
│       ├── llm_invoke.py            # LLM interface with error handling
│       ├── logger_config.py         # Comprehensive logging system
│       ├── prompt_loader.py         # Dynamic YAML prompt management
│       └── config_manager.py        # Centralized configuration management
├── config/                          # Configuration & domain knowledge
│   ├── problems_init.json           # Initial threat taxonomy
│   ├── problems.json                # LLM-enhanced threat definitions
│   ├── problems_updated.json        # Extended threat landscape
│   ├── data_info.yaml              # Domain-specific dataset intelligence
│   ├── ratio_config.json           # Attack prevalence configurations
│   └── prompts/                     # Specialized prompt templates
│       ├── generation_config.yaml   # Unified generation parameters
│       ├── realworld_analysis_prompts.yaml  # Real-world data processing
│       ├── seed_generation_prompts.yaml     # Technical seed creation
│       └── validation_prompts.yaml          # Quality assessment prompts
├── data/                            # Generated datasets
│   ├── seeds/                       # Technical seed examples by threat type
│   ├── large_samples/               # Scaled synthetic datasets
│   ├── mixed/                       # Hybrid real-world + synthetic datasets
│   ├── validation_reports/          # Technical accuracy assessments
│   └── quality_reports/             # Comprehensive quality analysis
├── raw/                             # Real-world input datasets
│   └── *.csv.gz                     # Compressed datasets with labels
└── logs/                            # Detailed execution logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12.2+
- OpenAI API key with GPT-4.1-mini access
- Poetry for dependency management

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd cyberdata

# Install dependencies
poetry install

# Configure environment
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Workflow A: Expert-Driven Generation

```bash
cd src/cyberdata/process

# 1. Generate comprehensive threat taxonomy (optional - included)
python problem_generator.py

# 2. Enhance with emerging threats and frameworks (optional - included)
python problem_enhancer.py

# 3. Create technical seed examples
python seed_generator.py

# 4. Validate seed technical accuracy
python seed_validator.py

# 5. Generate large-scale datasets with configurable attack ratios
python synthetic_generator.py --count 500 --malicious-ratio 0.3 --max-workers 6

# 6. Comprehensive quality assessment
python dataset_validator.py
```

### Workflow B: Real-World Data Enhancement

```bash
cd src/cyberdata/process

# 1. Enhance real-world data with domain intelligence
python realworld_processor.py --csv-file your_dataset.csv.gz --samples-per-class 10

# 2. Generate schema-preserving synthetic data
python synthetic_generator.py --count 500 --malicious-ratio 0.25

# 3. Create hybrid datasets with intelligent mixing
python data_mixer.py your_dataset.csv.gz --n-raw-samples 300 --n-synthetic-samples 200

# 4. Validate hybrid dataset quality
python dataset_validator.py
```

### Advanced Usage Examples

```bash
# High-performance generation for specific threats
python synthetic_generator.py --count 1000 --problems phishing spear_phishing \
    --malicious-ratio 0.15 --max-workers 8 --batch-size 4

# Real-world data processing with custom parameters
python realworld_processor.py --csv-file network_intrusion.csv \
    --data-info-name nsl_kdd_rare --samples-per-class 15

# Intelligent data mixing with automatic schema detection
python data_mixer.py five_email_phishing.csv.gz --n-raw-samples 500 \
    --n-synthetic-samples 500 --output-format csv

# Performance optimization for large-scale generation
python synthetic_generator.py --count 2000 --max-workers 12 \
    --batch-size 6 --disable-duplicates
```

## 🔧 Configuration

### Domain Knowledge Integration

The `config/data_info.yaml` file provides comprehensive domain-specific intelligence:

```yaml
datasets:
  five_email_phishing:
    data_description: |
      Curated datasets for phishing email detection with machine learning.
      Combines diverse examples from five sources for comprehensive coverage.
    data_schema: |
      subject: Email subject line text
      body: Email body content text
      label: Binary classification (1=malicious, 0=benign)
    domain: email_security
    attack_types:
      - phishing
      - social_engineering
```

### Attack Ratio Configuration

Configure realistic attack prevalence in `config/ratio_config.json`:

```json
{
  "ratios": {
    "global": 0.05,                    // 5% global attack rate
    "Enterprise": 0.08,                // 8% for enterprise scenarios
    "phishing": 0.15,                  // 15% for phishing datasets
    "credential_harvesting": 0.25      // 25% for credential attacks
  }
}
```

### Generation Parameters

Unified configuration in `config/prompts/generation_config.yaml`:

```yaml
generation:
  default_count: 500
  default_malicious_ratio: 0.05
  max_workers: 8
  batch_size: 3
  retry_attempts: 3
  temperature:
    malicious_generation: 0.7
    benign_generation: 0.7
    validation: 0.0
```

### Intelligent Data Mixing

The system automatically detects schema compatibility and creates hybrid datasets:

```bash
# Automatic schema detection and mixing
python data_mixer.py email_dataset.csv.gz
# → Finds compatible synthetic email data
# → Creates mixed dataset with 'synthetic' marker column
# → Maintains original ML pipeline compatibility
```

## 🔍 Quality Assurance Framework

### Multi-Layered Validation

1. **Schema Compliance**: Ensures exact structural compatibility
2. **Domain Accuracy**: LLM-based technical correctness assessment
3. **Statistical Analysis**: Distribution and uniqueness metrics
4. **Duplicate Detection**: Content-based deduplication with configurable fields
5. **Real-World Alignment**: Validates synthetic data against original patterns

### Quality Metrics

- **Technical Accuracy**: Domain expert-level technical detail validation
- **Schema Fidelity**: Exact column structure and type preservation
- **Attack Realism**: Authentic attack pattern representation
- **Benign Authenticity**: Realistic legitimate activity modeling
- **Balance Adherence**: Configurable malicious/benign ratio maintenance



### Class Imbalance Studies

```python
# Extreme imbalance scenarios
python synthetic_generator.py --count 10000 --malicious-ratio 0.001  # 0.1% attacks
python synthetic_generator.py --count 1000 --malicious-ratio 0.50    # Balanced
```

### Cross-Domain Evaluation

```python
# Generate data for different security domains
python synthetic_generator.py --problems phishing malware ddos \
    --count 500 --malicious-ratio 0.20
```

### Benchmark Creation

```python
# Standardized evaluation datasets
python synthetic_generator.py --count 5000 --malicious-ratio 0.05 \
    --problems enterprise_attacks --max-workers 8
```

## To-do
- Inject more real-world data into the generation to produce more various data.
- Test with more real-world datasets.
- M/L pipelines and benchmarks

## Version
0.3.0

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact
knowledgeivy01@gmail.com