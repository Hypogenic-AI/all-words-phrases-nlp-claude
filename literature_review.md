# Literature Review: Non-Compositional Token Sequences in LLMs

## Research Area Overview

This research investigates how LLMs process non-compositional, contiguous token sequences that function as single semantic units — forming what can be called an "implicit vocabulary." The core question is: how many such sequences exist, and how can we distinguish them from compositional phrases? This sits at the intersection of LLM interpretability, tokenization, and multiword expression (MWE) processing.

## Key Papers

### Paper 1: Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs
- **Authors**: Sheridan Feucht, David Atkinson, Byron C. Wallace, David Bau
- **Year**: 2024 (arXiv: 2406.20086)
- **Key Contribution**: First attempt to probe the implicit vocabulary of an LLM. Discovers "token erasure" — last token positions of multi-token words and named entities rapidly forget token-level information in early layers.
- **Methodology**:
  - Train linear probes on hidden states at each layer to predict nearby tokens
  - Observe that at last-token positions of multi-token words/entities, probe accuracy for current and previous tokens drops sharply in early layers (by layer 9)
  - Define an "erasure score" ψ that captures this forgetting pattern
  - Use Algorithm 1 (greedy document segmentation) to extract high-scoring non-overlapping segments
- **Datasets Used**: COUNTERFACT (Meng et al., 2022), Wikipedia (500 articles), The Pile
- **Results**:
  - ~1800 sequences recovered for Llama-2-7b, ~900 for Llama-3-8b on Wikipedia
  - 44.9% of recovered Llama-2-7b sequences are multi-token words or entities
  - Precision ~30% for Llama-2-7b, but recall very low (~1.6%)
  - Llama-3-8b (4x larger vocab) shows implicit vocabulary of larger multi-word expressions and code chunks
- **Code Available**: https://github.com/sfeucht/footprints, probes at HuggingFace sfeucht/footprints
- **Relevance**: **Foundational paper** — directly addresses our research question. Their erasure score provides the primary method for identifying implicit vocabulary items.

### Paper 2: Are Representations Built from the Ground Up? (Liu & Neubig, 2022)
- **Authors**: Emmy Liu, Graham Neubig
- **Year**: 2022 (arXiv: 2210.03575)
- **Key Contribution**: Studies whether LM representations are locally compositional using "tree reconstruction error" — predicting parent phrase embedding from child embeddings.
- **Methodology**: Train affine probes to predict phrase representations from constituent representations across BERT, RoBERTa, DeBERTa, GPT-2. Create CHIP dataset of idioms with matched non-idiomatic phrases.
- **Datasets Used**: Penn Treebank (823K phrases), CHIP dataset (1001 phrases with human compositionality judgments)
- **Results**:
  - Affine probes can predict phrase representations fairly well
  - Human compositionality judgments do NOT align well with model compositionality scores (best: Spearman ρ=0.19 for RoBERTa_CLS)
  - Named entities have lower compositionality scores (correctly treated as less compositional)
  - Models perform below chance on identifying which subphrase contributes more to meaning
- **Code Available**: https://github.com/nightingal3/lm-compositionality
- **Relevance**: Provides complementary methodology — measuring compositionality through reconstruction error rather than erasure. Key finding that LMs don't distinguish compositional vs. non-compositional well.

### Paper 3: Neurons in Large Language Models: Dead, N-gram, Positional (2023)
- **Authors**: Elena Voita et al.
- **Year**: 2023 (arXiv: 2309.04827)
- **Key Contribution**: Categorizes neurons in LLMs. Identifies "n-gram neurons" that activate on specific token sequences, providing evidence that models learn multi-token patterns.
- **Relevance**: N-gram neurons are a complementary mechanism to token erasure for understanding how models process multi-token units.

### Paper 4: Probing for Idiomaticity in Vector Space Models (Garcia et al., 2021)
- **Authors**: Marcos Garcia, Tiago Kramer Vieira, Carolina Scarton, Marco Idiart, Aline Villavicencio
- **Year**: 2021 (EACL)
- **Key Contribution**: Probes whether pre-trained models (BERT, etc.) can distinguish idiomatic from literal uses of expressions.
- **Datasets Used**: Noun compound datasets with compositionality ratings
- **Relevance**: Establishes probing methodology for non-compositional expressions in LMs.

### Paper 5: Semantics of Multiword Expressions in Transformer-Based Models: A Survey (2024)
- **Authors**: Various
- **Year**: 2024 (arXiv: 2401.15393)
- **Key Contribution**: Comprehensive survey of how transformer models handle MWEs.
- **Relevance**: Provides overview of the field and identifies open problems, including the challenge of detecting non-compositional semantics.

### Paper 6: Interpreting Token Compositionality in LLMs: A Robustness Analysis (2024)
- **Year**: 2024 (arXiv: 2410.12924)
- **Key Contribution**: Analyzes how robust LLM representations are to compositional vs. non-compositional token groupings.
- **Relevance**: Directly tests how models handle token composition, complementing the erasure approach.

### Paper 7: Using Shapley Interactions to Understand How Models Use Structure (2024)
- **Authors**: Singhvi et al.
- **Year**: 2024 (arXiv: 2403.13106)
- **Key Contribution**: Uses Shapley Taylor interaction indices to measure how inputs interact beyond linear superposition. Shows autoregressive models encode interactions correlating with syntactic proximity, and both autoregressive and masked models encode nonlinear interactions in idiomatic phrases.
- **Relevance**: Provides an alternative method (Shapley interactions) for detecting non-compositional semantics.

### Paper 8: Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)
- **Year**: 2020 (arXiv: 2012.14913)
- **Key Contribution**: Shows FF layers act as key-value memories, storing patterns in keys and associated distributions in values.
- **Relevance**: Helps explain the mechanism by which implicit vocabulary items might be stored and retrieved.

### Paper 9: CLCL: Non-compositional Expression Detection (2023)
- **Year**: 2023 (ACL)
- **Key Contribution**: Uses contrastive learning and curriculum learning for detecting non-compositional expressions.
- **Relevance**: Provides a detection method and dataset for non-compositional expressions.

### Paper 10: Unified Representation for Non-compositional and Compositional Expressions (PIER, 2023)
- **Year**: 2023 (arXiv: 2310.19127)
- **Key Contribution**: Proposes PIER model building on BART that creates semantically meaningful representations for potentially idiomatic expressions.
- **Relevance**: Provides approach for unified handling of compositional/non-compositional expressions.

### Paper 11: Rethinking Tokenization (Yang, 2024)
- **Year**: 2024 (arXiv: 2403.00417)
- **Key Contribution**: Proposes Less-is-Better (LiB) tokenizer that learns vocabulary of subwords, words, AND MWEs.
- **Relevance**: Shows tokenizers can be designed to incorporate multi-word expressions directly.

## Common Methodologies
- **Linear probing of hidden states** (Feucht et al., Liu & Neubig): Train probes to predict tokens or phrases from internal representations
- **Erasure/forgetting analysis** (Feucht et al.): Track how token-level information changes across layers
- **Tree reconstruction error** (Liu & Neubig): Predict parent from children embeddings
- **Shapley interactions** (Singhvi et al.): Measure non-additive interactions between inputs
- **Contrastive learning** (CLCL): Learn to distinguish compositional from non-compositional expressions

## Standard Baselines
- Multi-token word identification via whitespace splitting
- spaCy named entity recognition for entity identification
- Cosine similarity between constituent and phrase embeddings
- Simple additive/affine composition functions

## Evaluation Metrics
- Precision/recall of recovered vocabulary items vs. known multi-token words/entities
- Probe accuracy across layers (for erasure detection)
- Cosine distance/similarity (for compositionality measurement)
- Spearman correlation with human compositionality judgments
- SemEval-style idiomaticity detection accuracy

## Datasets in the Literature
- **COUNTERFACT** (Meng et al., 2022): Prompts requiring factual knowledge, used in Feucht et al.
- **The Pile** (Gao et al., 2020): 800GB diverse text corpus, used for probe training
- **Wikipedia**: Used in multiple papers for evaluation
- **Penn Treebank**: Used in Liu & Neubig for syntactic phrase analysis
- **CHIP**: Compositionality of Human-annotated Idiomatic Phrases (Liu & Neubig)
- **SemEval-2022 Task 2**: Multilingual idiomaticity detection dataset
- **Noun compound compositionality datasets** (Reddy et al., 2011; Ramisch et al., 2016)

## Gaps and Opportunities
1. **Scale**: Feucht et al. recover only ~1800 sequences with very low recall. Can we do better?
2. **Ground truth**: No authoritative ground truth for implicit vocabulary items exists
3. **Beyond Llama**: Only tested on Llama family; need cross-architecture validation
4. **Compositionality spectrum**: Current methods are binary; the spectrum from fully compositional to fully non-compositional is underexplored
5. **Combining approaches**: Token erasure + Shapley interactions + reconstruction error could be combined for better detection
6. **Frequency effects**: Relationship between token sequence frequency and "lexicality" is unclear

## Recommendations for Our Experiment
- **Recommended approach**: Extend the token erasure methodology from Feucht et al. as the primary method
- **Recommended datasets**: COUNTERFACT, Wikipedia, The Pile (all available in the footprints repo)
- **Recommended baselines**: Multi-token words (whitespace), spaCy NER, possibly BPE merge order
- **Recommended metrics**: Erasure score ψ, precision/recall vs. known multi-token words/entities, cosine similarity
- **Key tool**: nnsight library for accessing model internals, footprints codebase for erasure computation
- **Methodological considerations**:
  - Use pre-trained probes from HuggingFace (sfeucht/footprints) to avoid expensive retraining
  - Consider extending to newer models (Llama-3, Mistral)
  - Explore whether BPE merge frequency correlates with erasure score
  - Consider using multiple complementary signals (erasure + composition error + Shapley) to improve recall
