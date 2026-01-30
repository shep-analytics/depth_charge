"""
DepthCharge Configuration

All parameters referenced in the paper with their default values.
"""

import os

# =============================================================================
# API CONFIGURATION
# =============================================================================

# OpenRouter API key - set via environment variable or replace with your key
API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-api-key-here")

# OpenRouter base URL
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


# =============================================================================
# EVALUATION PARAMETERS (from paper Section 4.2)
# =============================================================================

# N: Number of questions per depth level
# Paper default: 30
# "We ask exactly N questions per depth (default 30)"
QUESTIONS_PER_DEPTH = 30

# Q: Number of passes per specificity tier
# Paper default: 3
# "Q passes per tier (default Q=3)"
PASSES_PER_TIER = 3

# Survival threshold: Stop when cumulative survival falls below this
# Paper default: 0.20 (20%)
# "We stop drilling when S(d) < θ (default θ = 0.20)"
SURVIVAL_THRESHOLD = 0.20

# Maximum drilling depth
# Paper default: 15
# "Maximum depth 15 (allowing progression through CUTTING_EDGE tier)"
MAX_DEPTH = 15

# Random seed for reproducibility
# Paper default: 42
# "Random seed 42 for reproducibility"
DEFAULT_SEED = 42


# =============================================================================
# SPECIFICITY TIERS (from paper Section 3.5)
# =============================================================================

# Depth-to-tier mapping with Q=3:
#   Depths 1-3:   COMMON (Wikipedia summaries)
#   Depths 4-6:   TEXTBOOK (Wikipedia detailed sections)
#   Depths 7-9:   PROFESSIONAL (clinical guidelines, professional standards)
#   Depths 10-12: SPECIALIST (peer-reviewed literature)
#   Depths 13+:   CUTTING_EDGE (recent publications from past 2 years)

TIER_NAMES = [
    "COMMON",       # General public knowledge
    "TEXTBOOK",     # University student level
    "PROFESSIONAL", # Practitioner level
    "SPECIALIST",   # Expert level
    "CUTTING_EDGE"  # Researcher level
]


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Models tested in the paper (anonymized as A, B, C, D, E)
# Replace with actual model IDs from your API provider

MODELS = {
    # Target models to evaluate
    "model_a": "your-model-a-id",  # Top performer on general benchmarks
    "model_b": "your-model-b-id",  # Strong reasoning capabilities
    "model_c": "your-model-c-id",  # Competitive performance
    "model_d": "your-model-d-id",  # Value-focused option
    "model_e": "your-model-e-id",  # Baseline comparison

    # Utility models
    "extractor": "your-extractor-model-id",  # For concept extraction, question generation, scoring
    "fact_search": "your-fact-search-model-id",  # For PROFESSIONAL+ tier fact verification
}

# Default model aliases (for command-line convenience)
DEFAULT_TARGET_MODEL = "model_a"
DEFAULT_EXTRACTOR_MODEL = "extractor"
DEFAULT_FACT_SEARCH_MODEL = "fact_search"


# =============================================================================
# COST RATES (approximate, per 1M tokens)
# =============================================================================

# Update these with your API provider's actual rates
COST_RATES = {
    "model_a": {"input": 1.00, "output": 3.00},
    "model_b": {"input": 15.00, "output": 75.00},
    "model_c": {"input": 3.00, "output": 15.00},
    "model_d": {"input": 0.27, "output": 1.10},
    "model_e": {"input": 1.25, "output": 5.00},
    "extractor": {"input": 0.075, "output": 0.30},
    "fact_search": {"input": 1.00, "output": 1.00},
}


# =============================================================================
# DOMAINS TESTED (from paper Section 4.4)
# =============================================================================

# Four diverse domains spanning different knowledge types:
#   - Medicine: Clinical and biomedical knowledge
#   - Constitutional Law: Legal domain with interpretive complexity
#   - Ancient Rome: Classical history
#   - Quantum Computing: Technical physics/CS domain

EXAMPLE_DOMAINS = [
    "Medicine",
    "Constitutional Law",
    "Ancient Rome",
    "Quantum Computing"
]


# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

# Wilson score confidence interval level
CONFIDENCE_LEVEL = 0.95

# Bootstrap samples for EVD standard error estimation
BOOTSTRAP_SAMPLES = 1000


# =============================================================================
# ABLATION STUDY PARAMETERS (from paper Section 6)
# =============================================================================

# Alternative N values tested
ABLATION_N_VALUES = [10, 30, 50]

# Alternative Q values tested
ABLATION_Q_VALUES = [1, 3, 5]

# Alternative threshold values tested
ABLATION_THRESHOLD_VALUES = [0.10, 0.20, 0.30]
