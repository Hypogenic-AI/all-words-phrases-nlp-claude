# Research Plan: Characterizing the Implicit Vocabulary of LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs process text as sequences of subword tokens, yet they develop internal representations that treat certain multi-token sequences as unified semantic units — an "implicit vocabulary." Understanding this implicit vocabulary is crucial for: (1) interpretability — knowing what units the model actually operates over, (2) tokenization design — informing better tokenizer choices, and (3) understanding how models handle non-compositional language (idioms, named entities, technical terms). Currently, we have no comprehensive characterization of these implicit vocabulary items, particularly regarding the spectrum from fully compositional to fully non-compositional sequences.

### Gap in Existing Work
Feucht et al. (2024) introduced the "token erasure" method and demonstrated that LLMs have implicit vocabulary items, but their work has key limitations:
1. **Low recall** (~1.6%) — they recover only ~1800 sequences from Wikipedia for Llama-2-7b
2. **No compositionality analysis** — they don't distinguish compositional from non-compositional sequences among recovered items
3. **Limited characterization** — they categorize items as multi-token words vs. entities but don't analyze the compositionality spectrum
4. **Single method** — they use only erasure scores without complementary signals

Liu & Neubig (2022) studied compositionality but found weak correlation between model and human judgments. No work has combined erasure-based vocabulary extraction with systematic compositionality analysis.

### Our Novel Contribution
We combine the erasure score method with:
1. **Embedding-based compositionality measurement** — comparing phrase embeddings to compositional predictions from constituents
2. **LLM-based compositionality classification** — using a state-of-the-art LLM to judge whether recovered sequences are compositional or non-compositional
3. **Comprehensive characterization** — analyzing the full distribution of compositionality scores across the implicit vocabulary, identifying clusters and categories
4. **Cross-model comparison** — running on both Llama-2-7b and Llama-3-8B to see how tokenizer vocabulary size affects implicit vocabulary composition

### Experiment Justification
- **Experiment 1 (Erasure Score Extraction)**: Necessary to recover the implicit vocabulary using the established method. We need these sequences before we can analyze them.
- **Experiment 2 (Compositionality Scoring)**: Tests whether recovered sequences systematically differ in compositionality from random bigrams/trigrams. This is the core novelty.
- **Experiment 3 (LLM Classification)**: Provides interpretable ground-truth labels for compositionality categories (idiom, named entity, technical term, compositional phrase). Enables analysis of what types of sequences the model treats as vocabulary items.
- **Experiment 4 (Cross-model Comparison)**: Tests whether models with larger tokenizer vocabularies have smaller implicit vocabularies (as suggested by Feucht et al.).

## Research Question
How many non-compositional contiguous sequences of tokens in an LLM exhibit token-like properties, and how can we distinguish these from compositional phrases?

## Hypothesis Decomposition
H1: LLMs possess a substantial implicit vocabulary of multi-token sequences with high erasure scores (>1000 items).
H2: Sequences with high erasure scores are disproportionately non-compositional (named entities, idioms, technical terms) compared to random token sequences.
H3: There exists a continuous spectrum of compositionality among implicit vocabulary items, not a binary distinction.
H4: Models with larger explicit tokenizer vocabularies (Llama-3 vs Llama-2) have fewer non-compositional implicit vocabulary items.

## Proposed Methodology

### Approach
1. Extract implicit vocabulary items using the erasure score (ψ) from Feucht et al., running on Wikipedia text
2. Compute an embedding-based compositionality score for each extracted multi-token sequence
3. Use GPT-4.1 to classify sequences into categories and rate compositionality
4. Analyze the relationship between erasure scores and compositionality

### Experimental Steps

#### Step 1: Erasure Score Extraction
- Load Llama-3-8B (larger tokenizer vocab = more interesting implicit items)
- Load pre-trained probes from HuggingFace (sfeucht/footprints)
- Run readout on Wikipedia test set (500 articles)
- Extract all multi-token segments with their erasure scores
- Also run on Llama-2-7b for comparison (if time permits)

#### Step 2: Embedding Compositionality Score
- For each extracted multi-token sequence, get the final-layer embedding of the full phrase
- Get embeddings of constituent tokens individually
- Compute compositionality score as: cosine_similarity(phrase_embedding, mean(constituent_embeddings))
- Low similarity = non-compositional, high similarity = compositional

#### Step 3: LLM-Based Classification
- Sample ~500 extracted sequences stratified by erasure score
- Send to GPT-4.1 with prompt asking for:
  - Category: named_entity, idiom, technical_term, compound_word, compositional_phrase, other
  - Compositionality rating: 1 (fully non-compositional) to 5 (fully compositional)
  - Brief explanation

#### Step 4: Analysis
- Distribution of erasure scores and compositionality scores
- Correlation between erasure score and compositionality
- Category breakdown of implicit vocabulary items
- Cross-model comparison (if time permits)

### Baselines
- **Random bigrams/trigrams**: Sample random contiguous token pairs/triples from same text
- **spaCy NER entities**: Named entities detected by spaCy as known non-compositional items
- **BPE merge order**: Token pairs that merge early in BPE should be more "word-like"

### Evaluation Metrics
- Distribution statistics of erasure scores (mean, std, percentiles)
- Precision/recall of erasure method vs. spaCy NER (as approximate ground truth)
- Spearman correlation between erasure score and compositionality ratings
- Category distribution of implicit vocabulary items
- Cohen's kappa for LLM classification reliability (via duplicate items)

### Statistical Analysis Plan
- Spearman rank correlation for erasure vs. compositionality scores (α=0.05)
- Mann-Whitney U test comparing compositionality of high-ψ vs. random sequences
- Chi-squared test for category distribution differences
- Bootstrap confidence intervals for all key statistics

## Expected Outcomes
- H1: We expect to find >1000 multi-token sequences with erasure scores above a meaningful threshold
- H2: We expect high-erasure sequences to be significantly less compositional than random sequences (effect size d > 0.5)
- H3: We expect a continuous distribution of compositionality scores, not a bimodal one
- H4: We expect Llama-3-8B to have fewer non-compositional implicit items than Llama-2-7b

## Timeline and Milestones
1. Environment setup + data loading: 10 min ✓
2. Erasure score extraction pipeline: 30 min
3. Run erasure on Wikipedia (500 docs): 60 min
4. Compositionality scoring: 30 min
5. LLM classification: 20 min
6. Analysis and visualization: 30 min
7. Documentation: 30 min

## Potential Challenges
- **Model loading**: Llama models need HF auth; fallback to using smaller model or cached weights
- **Compute time**: Full readout on 500 docs may be slow; can limit to 100-200 docs
- **Probe compatibility**: Pre-trained probes may have version issues; can retrain if needed
- **Low recall**: May recover few sequences; will analyze what we do find thoroughly

## Success Criteria
1. Successfully extract ≥500 multi-token implicit vocabulary items
2. Demonstrate statistically significant difference in compositionality between implicit vocabulary items and random sequences
3. Produce a clear characterization of what types of sequences constitute the implicit vocabulary
4. Generate publication-quality visualizations of the compositionality spectrum
