# All the Words and Phrases: Characterizing the Implicit Vocabulary of LLMs

LLMs develop internal representations that treat certain multi-token sequences as unified semantic units — an "implicit vocabulary" that extends beyond their explicit tokenizer vocabulary. This project quantifies and characterizes these non-compositional contiguous token sequences using a probe-free method based on layer-wise representation analysis.

## Key Findings

- **6,494 unique multi-token items** extracted from 150 Wikipedia articles using Mistral-7B, demonstrating a substantial implicit vocabulary
- **Named entities dominate** (38% of meaningful items), followed by technical terms (15%) and fixed expressions (12%)
- **Negative correlation** between erasure-like score and compositionality (Spearman rho = -0.182, p = 0.003): items the model "merges" more strongly tend to be less compositional
- **Continuous compositionality spectrum**: items range from fully non-compositional (named entities, idioms) to fully compositional, with no binary split
- **High-scoring items are significantly less compositional** than low-scoring items (Cohen's d = -0.60, p < 0.0001)

## How to Reproduce

### Environment Setup
```bash
uv venv && source .venv/bin/activate
uv add torch transformers spacy pandas numpy scipy matplotlib seaborn scikit-learn openai tqdm wandb
uv pip install en-core-web-sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

### Run Experiments
```bash
# Step 1: Extract implicit vocabulary (requires GPU, ~1.3 min)
CUDA_VISIBLE_DEVICES=0 python src/extract_vocabulary.py

# Step 2: Classify compositionality (requires OPENAI_API_KEY, ~5 min)
python src/classify_compositionality.py

# Step 3: Analyze and visualize (fast)
python src/analyze_and_visualize.py
```

### Requirements
- GPU with 16+ GB VRAM (tested on NVIDIA RTX A6000)
- OpenAI API key for GPT-4.1 classification
- ~$5 in API costs for classification

## File Structure
```
.
├── REPORT.md                       # Full research report with results
├── README.md                       # This file
├── planning.md                     # Research plan and methodology
├── src/
│   ├── extract_vocabulary.py       # Step 1: Extract implicit vocabulary
│   ├── classify_compositionality.py # Step 2: LLM-based classification
│   └── analyze_and_visualize.py    # Step 3: Analysis and plots
├── results/
│   ├── all_segments.csv            # All extracted multi-token spans
│   ├── implicit_vocabulary.csv     # Unique vocabulary items with scores
│   ├── classified_vocabulary.csv   # LLM-classified items
│   ├── ner_entities.csv            # spaCy NER entities
│   ├── statistical_tests.json      # All statistical test results
│   ├── summary_statistics.json     # Summary statistics
│   ├── config.json                 # Experiment configuration
│   └── plots/                      # All visualizations
├── code/footprints/                # Feucht et al. (2024) codebase
├── papers/                         # Downloaded research papers
└── literature_review.md            # Literature review
```

## References

- Feucht et al. (2024). "Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs." arXiv:2406.20086
- Liu & Neubig (2022). "Are Representations Built from the Ground Up?" arXiv:2210.03575
