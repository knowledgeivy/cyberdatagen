# CyberData: Synthetic Cybersecurity Dataset Generator

CyberData extracts patterns from real cybersecurity datasets and generates synthetic data that preserves technical accuracy and schema consistency. The system uses a three-step pipeline to ensure quality and production readiness.

**Key Features:**
- Real-world data pattern extraction
- Schema-preserving synthetic generation
- Multi-dimensional quality validation
- Large-scale production generation
- Domain-specific intelligence integration

## Workflow

### Phase 1: Real-World Analysis
1. **Domain Discovery**: Extract attack patterns and normal baselines from real data
2. **Contextual Enrichment**: Create threat landscape context and problem definitions  
3. **Schema-Aware Seeds**: Generate high-quality seed examples

### Phase 2: Scale Generation
4. **Scale Generation**: Generate large datasets using extracted intelligence
5. **Scale Validation**: Validate quality and filter for production use

## File Structure

```
cyberdata/
├── src/cyberdata/
│   ├── process/
│   │   ├── real_data_processor.py    # Step 1-3: Real-world analysis
│   │   ├── seed_validator.py         # Seed quality validation
│   │   ├── scale_generation.py       # Large-scale generation
│   │   ├── scale_validation.py       # Scale quality validation
│   │   └── data_mixer.py             # Mix real and synthetic data
│   └── utils/                        # Configuration and utilities
├── config/
│   ├── data_info.yaml               # Dataset schemas and metadata
│   ├── scale_config.yaml            # Generation parameters
│   ├── domain_discovery/            # Extracted domain patterns
│   ├── contextual_problems/         # Enriched problem contexts
│   └── prompts/                     # LLM prompt templates
├── data/
│   ├── seeds-raw/                   # Raw seed examples
│   ├── seeds-validated/             # Validated high-quality seeds
│   ├── scaled-raw/                  # Raw scale generation output
│   ├── scaled-validated/            # Production-ready datasets
│   └── scaled_validation/           # Quality reports
├── raw/                             # Input datasets
└── logs/                            # Execution logs
```

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

### Basic Workflow

```bash
cd src/cyberdata/process

# Step 1: Process real data and extract patterns
python real_data_processor.py --csv-file your_data.csv.gz --data-info-name dataset_name

# Step 2: Validate seed quality
python seed_validator.py

# Step 3: Generate scale dataset
python scale_generation.py --scale-count 1000 --malicious-ratio 0.3

# Step 4: Validate scale quality
python scale_validation.py
```

### Configuration

Configure your dataset in `config/data_info.yaml`:
```yaml
datasets:
  your_dataset:
    data_name: your_dataset
    domain: network_security
    label_column: label
    label_encoding:
      malicious: 1
      benign: 0
```

### Example Use Cases

**Generate email phishing dataset:**
```bash
python real_data_processor.py --csv-file email_data.csv.gz --data-info-name email_phishing
python seed_validator.py
python scale_generation.py --scale-count 5000 --malicious-ratio 0.15
```

**Generate network intrusion dataset:**
```bash
python real_data_processor.py --csv-file network_data.csv.gz --data-info-name nsl_kdd_rare
python seed_validator.py  
python scale_generation.py --scale-count 10000 --malicious-ratio 0.05
```

**Mix real and synthetic data:**
```bash
python data_mixer.py your_real_data.csv.gz --n-raw-samples 500 --n-synthetic-samples 500
```

## Output

Generated datasets include:
- **Production datasets**: `data/scaled-validated/` - High-quality, deployment-ready data
- **Quality reports**: `data/scaled_validation/` - Comprehensive validation metrics
- **Raw datasets**: `data/scaled-raw/` - Unfiltered generation output

## Command Line Options

### Scale Generation
```bash
python scale_generation.py \
    --scale-count 10000 \
    --malicious-ratio 0.1 \
    --max-workers 8 \
    --batch-size 10
```

### Real Data Processing  
```bash
python real_data_processor.py \
    --csv-file data.csv.gz \
    --data-info-name dataset_name \
    --samples-per-class 200
```

## To Do

- [ ] Machine learning model evaluation framework
- [ ] Class imbalance sensitivity analysis tools  


## Version

Current version: 2.1.5

## License

MIT License - see LICENSE file for details.

## Contact

knowledgeivy01@gmail.com