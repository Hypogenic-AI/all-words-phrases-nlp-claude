# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on non-compositional token sequences in LLMs ("implicit vocabulary"). Resources include papers, datasets, and code repositories.

## Papers
Total papers downloaded: 11

| # | Title | Authors | Year | File | Relevance |
|---|-------|---------|------|------|-----------|
| 1 | Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs | Feucht et al. | 2024 | papers/2406.20086_token_erasure_implicit_vocabulary.pdf | **Core** |
| 2 | Are Representations Built from the Ground Up? | Liu, Neubig | 2022 | papers/2210.03575_representations_ground_up.pdf | **Core** |
| 3 | Neurons in LLMs: Dead, N-gram, Positional | Voita et al. | 2023 | papers/2309.04827_neurons_dead_ngram_positional.pdf | High |
| 4 | Probing for Idiomaticity in Vector Space Models | Garcia et al. | 2021 | papers/eacl2021_probing_idiomaticity.pdf | High |
| 5 | MWE Semantics in Transformers: A Survey | Various | 2024 | papers/2401.15393_mwe_transformer_survey.pdf | High |
| 6 | CLCL: Non-compositional Expression Detection | Various | 2023 | papers/acl2023_clcl_noncompositional.pdf | High |
| 7 | Interpreting Token Compositionality in LLMs | Various | 2024 | papers/2410.12924_token_compositionality_robustness.pdf | High |
| 8 | Using Shapley Interactions for Structure | Singhvi et al. | 2024 | papers/2403.13106_shapley_interactions_structure.pdf | High |
| 9 | Rethinking Tokenization for LLMs | Yang | 2024 | papers/2403.00417_rethinking_tokenization.pdf | Medium |
| 10 | Unified Repr. for Compositional Expressions | Zeng, Bhat | 2023 | papers/2310.19127_unified_representation_compositional.pdf | Medium |
| 11 | FF Layers Are Key-Value Memories | Geva et al. | 2020 | papers/2012.14913_ff_layers_key_value_memories.pdf | Medium |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets available: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| COUNTERFACT (Expanded) | ROME/footprints | ~12K prompts, 6.7MB | Entity factual recall | code/footprints/data/counterfact_expanded.csv | Primary entity evaluation |
| The Pile (subsets) | EleutherAI | 500-1000 docs each | Probe training/eval | code/footprints/data/{train,val,test}_tiny_*.csv | Probe training data |
| Wikipedia (subsets) | Wikimedia | 500-1000 articles each | Multi-token word eval | code/footprints/data/wikipedia_*.csv | Primary word evaluation |
| Pre-trained Probes | HuggingFace | 34 checkpoints/model | Token prediction | sfeucht/footprints (HuggingFace) | Saves GPU compute |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| footprints | github.com/sfeucht/footprints | Token erasure & implicit vocabulary | code/footprints/ | Core implementation, includes all data |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service with diligent mode for 3 complementary queries
- Focused on: non-compositional token sequences, implicit vocabulary, token erasure, compositionality detection, MWE in LLMs
- 143+ papers returned across searches; filtered to top 11 by relevance and citation count

### Selection Criteria
- Prioritized papers directly addressing implicit vocabulary or non-compositional token processing in LLMs
- Selected complementary methodologies: erasure probing, composition reconstruction, Shapley interactions, contrastive learning
- Ensured coverage of both foundational work and recent advances (2020-2024)

### Challenges Encountered
- No single "ground truth" dataset exists for implicit vocabulary items
- The research question is relatively new (2024), so few papers directly address it
- Most MWE work focuses on detection rather than enumeration of a model's full implicit vocabulary

### Gaps and Workarounds
- No existing dataset maps LLM implicit vocabulary → addressed by using footprints methodology
- Limited to Llama family in core paper → can extend to other models using same probing approach
- Low recall in vocabulary extraction → combining multiple signals (erasure + composition + frequency) may help

## Recommendations for Experiment Design

1. **Primary approach**: Use the footprints codebase to replicate and extend token erasure analysis
2. **Primary dataset(s)**: Wikipedia (500 articles) and The Pile subsets — already prepared in footprints/data
3. **Baseline methods**:
   - Whitespace-separated multi-token words
   - spaCy NER entities
   - BPE merge order (tokens that merge early are more "word-like")
4. **Evaluation metrics**:
   - Erasure score ψ distribution analysis
   - Precision/recall vs. known multi-token words and entities
   - Vocabulary size estimation at different ψ thresholds
5. **Code to adapt/reuse**:
   - `footprints/scripts/segment.py` — core segmentation algorithm
   - `footprints/scripts/readout.py` — vocabulary readout across a corpus
   - Pre-trained probes from HuggingFace `sfeucht/footprints`
6. **Compute requirements**: GPU with ~16GB VRAM for Llama-2-7b inference (no probe training needed if using pre-trained probes)
