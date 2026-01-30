# DepthCharge

A domain-agnostic framework for measuring knowledge depth in Large Language Models through adaptive drilling with verified ground truth and survival statistics.

## Overview

DepthCharge measures how deeply LLMs can maintain accuracy through adaptive follow-up questioning. Unlike static benchmarks, it:

1. **Adapts to model responses**: Follow-up questions drill into concepts the model actually mentions
2. **Verifies ground truth on-demand**: Every question has a known correct answer from authoritative sources
3. **Maintains statistical power**: Always asks N questions per depth (default 30)
4. **Uses cumulative survival**: Errors compound across depths, revealing reliability under sustained questioning

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to set your API credentials and model IDs:

```python
# Set your API key
API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")

# Configure your models
MODELS = {
    "model_a": "your-model-a-id",
    "model_b": "your-model-b-id",
    # ...
    "extractor": "your-extractor-model-id",
    "fact_search": "your-fact-search-model-id",
}
```

Or set via environment variable:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
python depth_charge.py --model model_a --topic "Medicine"
```

### Full Options

```bash
python depth_charge.py \
    --model model_a \
    --topic "Quantum Computing" \
    --questions-per-depth 30 \
    --passes-per-tier 3 \
    --max-depth 15 \
    --survival-threshold 0.20 \
    --seed 42 \
    --output results/output.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `model_a` | Target model key from `config.MODELS` |
| `--topic` | `Medicine` | Domain/topic to probe |
| `-n, --questions-per-depth` | 30 | Questions asked at each depth level (N) |
| `-q, --passes-per-tier` | 3 | Depth passes within each specificity tier (Q) |
| `--max-depth` | 15 | Maximum drilling depth |
| `--survival-threshold` | 0.20 | Stop when cumulative survival falls below this |
| `--seed` | 42 | Random seed for reproducibility |
| `--output` | `results/probe_output.json` | Output file path |

## Specificity Tiers

With Q=3 passes per tier, depths map to tiers as follows:

| Depth | Tier | Knowledge Level | Fact Source |
|-------|------|-----------------|-------------|
| 1-3 | COMMON | General public | Wikipedia summaries |
| 4-6 | TEXTBOOK | University student | Wikipedia detailed |
| 7-9 | PROFESSIONAL | Practitioner | Professional sources |
| 10-12 | SPECIALIST | Expert | Peer-reviewed literature |
| 13+ | CUTTING_EDGE | Researcher | Recent publications |

## Metrics

### Expected Valid Depth (EVD)

The primary metric is EVD, computed as the area under the cumulative survival curve:

```
EVD = Σ S(d) for d = 1 to D
```

where `S(d)` is the cumulative survival at depth `d`:

```
S(d) = Π A(i) for i = 1 to d
```

and `A(i)` is the accuracy at depth `i`.

### Output Format

Results are saved as JSON with the following structure:

```json
{
  "topic": "Medicine",
  "target_model": "model_id",
  "questions_per_depth": 30,
  "survival_curve": [
    {
      "depth": 1,
      "level": "COMMON/P1",
      "questions_asked": 30,
      "correct": 28,
      "accuracy_at_depth": 0.933,
      "cumulative_survival": 0.933
    },
    ...
  ],
  "statistics": {
    "expected_valid_depth": 6.08,
    "median_depth": 5,
    "max_depth_with_correct": 11,
    "overall_accuracy": 0.82,
    "accuracy_by_tier": {
      "COMMON": 0.97,
      "TEXTBOOK": 0.76,
      "PROFESSIONAL": 0.54
    }
  },
  "all_probes": [...],
  "metrics": {
    "total_calls": 1024,
    "total_runtime_sec": 3847.2,
    "estimated_cost_usd": 0.96
  }
}
```

## Reproducing Paper Results

To reproduce the experiments from the paper:

```bash
# Run on each domain
for topic in "Medicine" "Constitutional Law" "Ancient Rome" "Quantum Computing"; do
    for model in model_a model_b model_c model_d model_e; do
        python depth_charge.py \
            --model $model \
            --topic "$topic" \
            --questions-per-depth 30 \
            --passes-per-tier 3 \
            --seed 42 \
            --output "results/${model}_${topic// /_}.json"
    done
done
```

## License

MIT License
