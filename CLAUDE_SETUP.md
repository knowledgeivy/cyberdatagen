# Claude 3.5 Haiku Setup Guide

This guide explains how to set up and run experiments using Claude 3.5 Haiku for synthetic spam data generation, maintaining consistency with existing GPT-4.1-mini experiments.

## Overview

The Claude 3.5 Haiku integration provides:
- **Cost-effective alternative**: $0.80/$4.00 per MTok (2-2.5x cost of GPT-4.1-mini)
- **Comparable performance**: Similar capability level to GPT-4.1-mini
- **Identical structure**: Same file organization, config format, and workflow
- **Parallel execution**: Full support for distributed generation

## Model Comparison

| Aspect | GPT-4.1-mini | Claude 3.5 Haiku |
|--------|--------------|------------------|
| **Pricing** | $0.40 / $1.60 per MTok | $0.80 / $4.00 per MTok |
| **Cost Ratio** | 1x (baseline) | 2x / 2.5x |
| **Context Window** | 128K tokens | 200K tokens |
| **Max Output** | 16,384 tokens | 8,192 tokens |
| **Provider** | OpenAI | Anthropic |
| **Model ID** | `gpt-4.1-mini` | `claude-3-5-haiku-20241022` |

## Installation

### 1. Install Anthropic SDK

```bash
pip install anthropic
```

### 2. Configure API Key

Edit `.env` file and add your Anthropic API key:

```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

To get an API key:
1. Visit https://console.anthropic.com/
2. Sign up or log in
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

### 3. Verify Installation

```bash
python -c "from cyberdata.utils.claude_llm_invoke import process_llm_request; print('Claude setup OK')"
```

## File Structure

The Claude integration maintains parallel structure to GPT experiments:

```
cyberdata/
├── src/cyberdata/
│   ├── utils/
│   │   ├── llm_invoke.py              # OpenAI/GPT interface
│   │   └── claude_llm_invoke.py       # Anthropic/Claude interface ✨ NEW
│   └── process/
│       ├── scale_generation.py        # GPT scale generation
│       └── claude_scale_generation.py # Claude scale generation ✨ NEW
├── config/
│   ├── scale_config.yaml              # GPT configuration
│   └── claude_scale_config.yaml       # Claude configuration ✨ NEW
├── scripts/
│   └── run_claude_scale_generation.py # Claude execution script ✨ NEW
└── data/
    ├── scaled-raw/                    # GPT generated data
    └── scaled-raw-claude/             # Claude generated data ✨ NEW
```

## Usage

### Basic Usage

Run with default settings from `config/claude_scale_config.yaml`:

```bash
python scripts/run_claude_scale_generation.py
```

### Custom Parameters

```bash
# Specify generation count and ratio
python scripts/run_claude_scale_generation.py \
    --scale-count 1000 \
    --malicious-ratio 0.5

# Adjust parallelism (recommended: 8-10 workers for Claude)
python scripts/run_claude_scale_generation.py \
    --max-workers 8 \
    --batch-size 5

# Generate specific problems only
python scripts/run_claude_scale_generation.py \
    --problems phishing spam malware

# Full custom configuration
python scripts/run_claude_scale_generation.py \
    --scale-count 2000 \
    --malicious-ratio 0.6 \
    --max-workers 10 \
    --batch-size 5 \
    --problems spam phishing
```

### Advanced Usage

#### Direct Python API

```python
from cyberdata.process.claude_scale_generation import ClaudeScaleGenerator

# Initialize generator
generator = ClaudeScaleGenerator()

# Generate data for specific problem
scale_data = generator.generate_scale_data(
    area="Email Security",
    nature="phishing",
    scale_count=1000,
    malicious_ratio=0.5
)

# Save results
scale_file = generator.save_scale_data("Email Security", "phishing", scale_data)
print(f"Generated {len(scale_data['samples'])} samples")
print(f"Saved to: {scale_file}")
```

## Configuration

### Claude-Specific Settings

Key differences in `config/claude_scale_config.yaml`:

```yaml
scale_generation:
  defaults:
    max_workers: 10      # vs 12 for GPT (lower for Claude API limits)
    batch_size: 5        # vs 5 for GPT (same)

  performance:
    api_delay: 0.3       # vs 0.2 for GPT (slightly higher)

models:
  primary:
    name: "claude-3-5-haiku-20241022"
    max_tokens: 8192     # vs 16384 for GPT
    provider: "anthropic"

  rate_limiting:
    calls_per_minute: 50 # vs 60 for GPT (more conservative)
    parallel_request_limit: 6  # vs 8 for GPT

claude_specific:
  cost_tracking:
    cost_per_million_input: 0.80
    cost_per_million_output: 4.00
```

### Performance Tuning

For optimal performance with Claude 3.5 Haiku:

- **max_workers**: 8-10 (Claude API is more rate-limited than OpenAI)
- **batch_size**: 5 (balanced for Claude's context handling)
- **api_delay**: 0.3s (slightly higher to avoid rate limits)

## Cost Estimation

For your experiment with **6000 samples** (3 prompts × 2000 samples each):

### Assumptions
- Input per sample: ~800 tokens (prompt + context)
- Output per sample: ~800 tokens (generated spam)
- Total: 1,600 tokens per sample

### Cost Calculation

```
Input tokens:  6000 × 800 × $0.80/M = $3.84
Output tokens: 6000 × 800 × $4.00/M = $19.20
Total cost:    $23.04
```

### Comparison with GPT-4.1-mini

| Metric | GPT-4.1-mini | Claude 3.5 Haiku | Difference |
|--------|--------------|------------------|------------|
| Input cost | $1.92 | $3.84 | +$1.92 (2x) |
| Output cost | $7.68 | $19.20 | +$11.52 (2.5x) |
| **Total cost** | **$9.60** | **$23.04** | **+$13.44 (2.4x)** |

**Total experiment budget**: ~$23 (vs $9.60 for GPT-4.1-mini)

## Output Data Structure

Claude-generated data maintains the same structure as GPT data:

```
data/scaled-raw-claude/
└── Email_Security/
    ├── phishing_scale_claude.json
    ├── spam_scale_claude.json
    └── malware_scale_claude.json
```

Each JSON file contains:
```json
{
  "samples": [
    {
      "sample_id": "mal_0_0",
      "generation_timestamp": 1234567890.123,
      "label": 1,
      "content": "...",
      ...
    }
  ],
  "metadata": {
    "model_name": "claude-3-5-haiku-20241022",
    "generation_method": "enhanced_claude_scale_generation",
    "actual_samples_generated": 1000,
    ...
  }
}
```

## Comparison Experiments

To run comparative experiments between GPT-4.1-mini and Claude 3.5 Haiku:

### Step 1: Generate with GPT (if not done)

```bash
python scripts/run_scale_generation.py \
    --scale-count 1000 \
    --malicious-ratio 0.5
```

### Step 2: Generate with Claude

```bash
python scripts/run_claude_scale_generation.py \
    --scale-count 1000 \
    --malicious-ratio 0.5
```

### Step 3: Compare Results

Both datasets will have identical structure, allowing direct comparison:
- `data/scaled-raw/` - GPT-4.1-mini results
- `data/scaled-raw-claude/` - Claude 3.5 Haiku results

## Troubleshooting

### API Key Issues

```
Error: ANTHROPIC_API_KEY not found
```

**Solution**: Verify `.env` file has correct key:
```bash
cat .env | grep ANTHROPIC_API_KEY
```

### Rate Limiting

```
Error: Rate limit exceeded
```

**Solution**: Reduce parallelism in config:
```yaml
scale_generation:
  defaults:
    max_workers: 6  # Reduce from 10
    batch_size: 3   # Reduce from 5
```

### Import Errors

```
ModuleNotFoundError: No module named 'anthropic'
```

**Solution**: Install the Anthropic SDK:
```bash
pip install anthropic
```

### Context Length Errors

```
Error: Maximum context length exceeded
```

**Solution**: Claude 3.5 Haiku has 8K max output tokens (vs 16K for GPT). Reduce batch size:
```yaml
scale_generation:
  defaults:
    batch_size: 3  # Reduce from 5
```

## Best Practices

1. **Start Small**: Test with `--scale-count 100` first
2. **Monitor Costs**: Track token usage in logs
3. **Use Checkpoints**: Generation supports resume from checkpoint
4. **Parallel Execution**: Recommended 8-10 workers for Claude
5. **Rate Limits**: Allow 0.3s delay between API calls

## Research Benefits

Using both models provides:

1. **Model Comparison**: Compare OpenAI vs Anthropic generation quality
2. **Robustness**: Test classifier performance across different LLM outputs
3. **Publication Value**: Demonstrates generalizability across LLM providers
4. **Cost-Performance Tradeoff**: Quantify quality vs cost relationships

## Questions?

For issues or questions:
1. Check logs in `logs/` directory
2. Review `config/claude_scale_config.yaml`
3. Compare with GPT implementation in `src/cyberdata/process/scale_generation.py`

## Summary

**Ready to run!**

```bash
# 1. Set API key in .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env

# 2. Run generation
python scripts/run_claude_scale_generation.py --scale-count 1000

# 3. Check output
ls data/scaled-raw-claude/
```

**Estimated cost for 6000 samples**: ~$23
**Estimated time**: ~15-20 minutes (with 10 workers)
