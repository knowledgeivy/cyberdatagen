# Cyber Synthetic Data Generation

A modular Python toolkit for generating and validating synthetic cybersecurity datasets from both manual problem definitions and real-world data sources. This project provides a complete pipeline supporting multiple data input methods with comprehensive validation and quality assurance.

## 🎯 Project Overview

CyberData provides two distinct workflows for synthetic cybersecurity data generation:

### Workflow A: Manual Problem Definition
1. **Define cybersecurity problems** by operational environment and threat type
2. **Create seed datasets** with LLM-generated technical examples
3. **Generate large synthetic datasets** using LLM-based expansion
4. **Validate datasets** with statistical and quality metrics

### Workflow B: Real-World Data Processing
1. **Process real-world data** with stratified sampling and LLM analysis
2. **Generate seed datasets** from actual cybersecurity incidents
3. **Generate large synthetic datasets** using real-world patterns
4. **Validate datasets** with enhanced real-world alignment checks

Both workflows converge at the synthetic data generation step, enabling flexible data sourcing while maintaining consistent output quality.

## 📂 Project Structure

```
cyberdata/
├── src/cyberdata/
│   ├── process/                     # Core pipeline scripts
│   │   ├── problem_generator.py     # Generate cybersecurity problems
│   │   ├── problem_enhancer.py      # Evaluate & enhance taxonomy
│   │   ├── seed_generator.py        # Create seeds from problem definitions
│   │   ├── realworld_processor.py   # Create seeds from real-world data
│   │   ├── synthetic_generator.py   # Generate large synthetic datasets
│   │   ├── seed_validator.py        # Validate seed examples
│   │   └── dataset_validator.py     # Quality assessment of large datasets
│   └── utils/                       # Utility modules
│       ├── llm_invoke.py            # LLM API interface
│       ├── logger_config.py         # Logging configuration
│       ├── prompt_loader.py         # YAML prompt management
│       └── config_manager.py        # Centralized configuration management
├── config/                          # Configuration files
│   ├── problems_init.json           # Initial problem definitions
│   ├── problems.json                # LLM-generated problems
│   ├── problems_updated.json        # Enhanced with taxonomy & new threats
│   ├── problems_evaluation_report.json  # Evaluation analysis
│   └── prompts/                     # YAML prompt templates
│       ├── problems_prompts.yaml
│       ├── extension_prompts.yaml
│       ├── seed_generation_prompts.yaml
│       ├── generation_config.yaml
│       ├── validation_prompts.yaml
│       └── realworld_analysis_prompts.yaml
├── data/                            # Generated datasets
│   ├── seeds/                       # Seed examples by area
│   ├── large_samples/               # Large synthetic datasets
│   ├── validation_reports/          # Validation results
│   └── quality_reports/             # Quality assessment reports
├── raw/                             # Raw real-world datasets
│   └── *.csv.gz                     # Compressed CSV files with labels
├── logs/                            # Execution logs
└── pyproject.toml                   # Project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12.2+
- OpenAI API key
- Poetry for dependency management

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cyberdata
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Set up environment:**
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

### Workflow A: Manual Problem Definition

```bash
# Navigate to process directory
cd src/cyberdata/process

# 1. Generate cybersecurity problems (optional - problems.json included)
python problem_generator.py

# 2. Evaluate and enhance taxonomy (optional - problems_updated.json included)
python problem_enhancer.py

# 3. Create seed examples from problem definitions
python seed_generator.py

# 4. Validate seed examples
python seed_validator.py

# 5. Generate large synthetic datasets
python synthetic_generator.py --count 50 --malicious-ratio 0.6

# 6. Validate large datasets
python dataset_validator.py
```

### Workflow B: Real-World Data Processing

```bash
# Navigate to process directory
cd src/cyberdata/process

# 1. Process real-world data to create seeds
python realworld_processor.py --csv-file your_dataset.csv.gz --label-column label --samples-per-class 5

# 2. Generate large synthetic datasets (same as Workflow A)
python synthetic_generator.py --count 50 --malicious-ratio 0.6

# 3. Validate large datasets
python dataset_validator.py
```

### Advanced Usage Examples

```bash
# Process different datasets with custom parameters
python realworld_processor.py --csv-file network_logs.csv --label-column is_malicious --samples-per-class 10

# Generate more samples with different configurations
python synthetic_generator.py --count 100 --malicious-ratio 0.3 --max-workers 6 --batch-size 3

# Process specific problems only
python synthetic_generator.py --count 30 --problems phishing credential_harvesting

# Performance tuning for large datasets
python synthetic_generator.py --count 200 --max-workers 8 --batch-size 4 --disable-duplicates
```

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
```

### Real-World Data Requirements

Place your raw datasets in the `raw/` directory. Supported formats:
- **CSV files** (`.csv` or `.csv.gz`)
- **Required columns**: A label column indicating malicious (1) vs benign (0) samples
- **Optional**: Any additional feature columns for context

Example structure:
```
raw/
├── email_phishing.csv.gz          # Email security data
├── network_intrusion.csv          # Network traffic data
└── malware_samples.csv.gz         # Malware analysis data
```

### Model Configuration

- **Default Model**: GPT-4.1-mini
- **Temperature**: 0.7 (generation), 0.0 (validation)
- **Max Tokens**: 16,384

## 📊 Data Schema

### Problem Definition Schema
```json
{
  "area": "High-level category (e.g., Social Engineering, Network Attacks)",
  "nature": "Specific threat type (e.g., phishing, ddos, ransomware)",
  "description": "Detailed explanation of the threat scenario",
  "attack_vector": "Primary attack method",
  "asset_targeted": "Target resource type",
  "impact_level": "Severity assessment (Low, Medium, High, Critical)",
  "risk_reduction": ["List of mitigation strategies"]
}
```

### Seed Example Schema
```json
{
  "scenario": "Brief description of the attack instance",
  "technical_data": "Detailed technical information",
  "indicators": ["Specific technical indicators of compromise"],
  "detection_method": "How this would be detected in practice",
  "relevant_mitre_techniques": ["MITRE ATT&CK techniques"],
  "sample_type": "malicious or benign",
  "real_world_alignment": "How this relates to actual data patterns"
}
```

## 🔍 Quality Assurance

The toolkit implements multi-layered quality assurance:

1. **Schema Validation**: Ensures consistent data structure across both workflows
2. **LLM-based Quality Assessment**: Evaluates realism and technical accuracy
3. **Statistical Analysis**: Measures uniqueness, distribution, and balance
4. **Real-World Alignment**: Validates synthetic data against original patterns
5. **Domain Expert Validation**: Built-in cybersecurity expertise in prompts
6. **Comprehensive Logging**: Tracks all generation and validation steps

### Quality Metrics

- **Uniqueness Ratio**: Percentage of unique samples in the dataset
- **Technical Accuracy**: LLM-assessed realism of technical details
- **Real-World Fidelity**: Alignment with original data patterns (Workflow B)
- **Sample Balance**: Distribution of malicious vs benign samples
- **Schema Consistency**: Adherence to expected data structure

## 🛠️ Extending the System

### Adding New Threat Types

1. **Update taxonomy** in `problems_init.json`
2. **Enhance prompts** for domain-specific knowledge
3. **Add validation rules** for new data structures
4. **Test with real-world data** if available

### Supporting New Data Formats

1. **Extend `realworld_processor.py`** for new file types
2. **Add format-specific** schema analysis prompts
3. **Update validation logic** for new data characteristics

### Custom Workflows

The modular design allows easy customization:
- Mix manual and real-world data sources
- Create domain-specific problem generators
- Implement custom validation metrics
- Add new synthetic data generation techniques

## 📈 Performance Optimization

### Parallel Processing
- **Default**: 4 workers, batch size 2
- **High-performance**: `--max-workers 8 --batch-size 4`
- **Memory-constrained**: `--max-workers 2 --batch-size 1`

### Duplicate Detection
- **Enabled by default** with content-based hashing
- **Configurable fields** for duplicate checking
- **Disable** with `--disable-duplicates` for speed

### Large Dataset Generation
- **Streaming processing** for memory efficiency
- **Retry logic** with exponential backoff
- **Progress tracking** with detailed logging

## 🔬 Research Applications

This toolkit supports various cybersecurity research applications:

- **Model Robustness Testing**: Generate adversarial examples with controlled difficulty
- **Class Imbalance Studies**: Create datasets with specific malicious/benign ratios
- **Cross-Domain Evaluation**: Test models on synthetic data from different threat landscapes
- **Benchmark Creation**: Standardized datasets for comparing detection algorithms
- **Data Augmentation**: Enhance small real-world datasets with synthetic examples

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

knowledgeivy01@gmail.com

---

## 🆕 What's New in Enhanced Version

### Real-World Data Integration
- **Stratified sampling** from labeled cybersecurity datasets
- **Automatic schema analysis** using LLM-based data understanding
- **Pattern preservation** in synthetic data generation
- **Enhanced validation** with real-world alignment metrics

### Improved Pipeline Architecture
- **Two distinct workflows** for different data sources
- **Converged processing** at synthetic generation step
- **Modular design** enabling easy extension and customization
- **Clear file naming** without numeric prefixes

### Enhanced Quality Assurance
- **Multi-source validation** supporting both manual and real-world derived data
- **Technical accuracy assessment** with domain-specific evaluation
- **Real-world fidelity metrics** for data alignment validation
- **Comprehensive reporting** with detailed quality statistics

## Version
0.2.1

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact
knowledgeivy01@gmail.com